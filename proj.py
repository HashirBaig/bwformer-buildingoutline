import os
import sys
import glob
import numpy as np, cv2
from collections import defaultdict, deque
import json
import open3d as o3d
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import NearestNeighbors

def _fit_roof_plane_o3d(pc_xyz, dist_thresh=0.02, ransac_n=3, num_iters=2000):
    """Fit dominant roof plane on normalized points (0..255) using Open3D RANSAC."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_xyz.astype(np.float64))
    plane_model, inliers = pcd.segment_plane(distance_threshold=dist_thresh,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iters)
    a,b,c,d = plane_model  # ax+by+cz+d=0
    inliers = np.asarray(inliers, dtype=np.int64)
    return (a,b,c,d), inliers

def _orthonormal_frame_from_plane(a,b,c):
    """Build (u,v,n) basis given plane normal n=[a,b,c]."""
    n = np.array([a,b,c], dtype=np.float64)
    n /= (np.linalg.norm(n) + 1e-12)
    # choose a vector not parallel to n
    h = np.array([1.0,0.0,0.0]) if abs(n[0]) < 0.9 else np.array([0.0,1.0,0.0])
    u = np.cross(n, h); u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u); v /= (np.linalg.norm(v) + 1e-12)
    return u, v, n

def _project_points_to_plane_uv(pc_xyz, a,b,c,d):
    """Project points to plane coordinate system -> (u,v) in 2D."""
    u,v,n = _orthonormal_frame_from_plane(a,b,c)
    # choose an origin on plane: n·X0 + d = 0 -> X0 = -d*n
    X0 = -d * n
    # vector from origin to each point
    q = pc_xyz - X0[None,:]
    U = q @ u  # (N,)
    V = q @ v  # (N,)
    return np.stack([U,V], axis=1), (u,v,n,X0)

def _normalize_uv_to_canvas(uv, pad=8, size=256):
    """Affine normalize uv coords to 0..(size-1), keeping aspect."""
    umin,vmin = uv.min(0); umax,vmax = uv.max(0)
    du = max(umax-umin, 1e-6); dv = max(vmax-vmin, 1e-6)
    s = (size - 2*pad) / max(du, dv)
    T = np.array([ [s, 0, -(umin*s) + pad],
                   [0, s, -(vmin*s) + pad],
                   [0, 0, 1] ], dtype=np.float64)
    uv1 = np.c_[uv, np.ones(len(uv))] @ T.T
    return uv1[:,:2], T  # canvas coords (float), 3x3 affine

def _mask_by_bag_polygon_xy(pc_xy_canvas, bag_rings_canvas):
    """Keep points whose (x,y) in ANY BAG ring (in canvas coordinates)."""
    mask = np.zeros(len(pc_xy_canvas), dtype=bool)
    for ring in bag_rings_canvas:
        cnt = ring.astype(np.float32)[None,:,:]  # (1,N,2)
        inside = cv2.pointPolygonTest(cnt[0], (0,0), False)  # dummy call to warm-up
        # vectorized test via rasterization
        H=W=256
        poly = np.rint(ring).astype(np.int32)[None,:,:]
        roi = np.zeros((H,W), np.uint8); cv2.fillPoly(roi, poly, 1)
        px = np.clip(np.rint(pc_xy_canvas[:,0]).astype(int), 0, W-1)
        py = np.clip(np.rint(pc_xy_canvas[:,1]).astype(int), 0, H-1)
        mask |= (roi[py, px] > 0)
    return mask

def _largest_contour_intersecting_bag(bin_img, bag_mask):
    """Return largest contour that has positive intersection with BAG mask."""
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best = None; best_area = 0
    for c in contours:
        area = abs(cv2.contourArea(c))
        if area <= 10:  # ignore tiny blobs
            continue
        test = np.zeros_like(bin_img); cv2.drawContours(test, [c], -1, 1, -1)
        if (test & bag_mask).any():
            if area > best_area:
                best, best_area = c, area
    return best

def wireframe_from_pointcloud(pc_xyz_255, bag_rings_proj, d_k=3, dilate=1, close=3):
    """
    Build vertices/edges from POINT CLOUD under BAG guidance.
    Inputs:
      pc_xyz_255: (N,3) in your 0..255 normalized frame
      bag_rings_proj: list of rings in 0..255 frame (same canvas as your jpg)
    Returns:
      vertices_proj (V,3)  in 0..255 frame, edges_idx (E,2).
    """
    # 1) Fit dominant plane on (x,y,z) in 0..255 frame
    (a,b,c,d), inliers = _fit_roof_plane_o3d(pc_xyz_255, dist_thresh=1.0, ransac_n=3, num_iters=2000)
    roof = pc_xyz_255[inliers]  # (M,3)

    # 2) Project to plane (u,v), then normalize to canvas 0..255
    uv, (u_axis,v_axis,n_axis,X0) = _project_points_to_plane_uv(roof, a,b,c,d)
    uv_canvas, T_uv2canvas = _normalize_uv_to_canvas(uv, pad=8, size=256)

    # 3) BAG guidance: project BAG XY rings directly with the same T_uv2canvas (Option A restored)
    bag_rings_canvas = []
    for ring_xy in bag_rings_proj:
        ring_xy1 = np.c_[ring_xy[:, :2], np.ones(len(ring_xy))]
        ring_canvas = (ring_xy1 @ T_uv2canvas.T)[:, :2]
        bag_rings_canvas.append(ring_canvas)



    # 4) Keep only points inside BAG mask (safety)
    pts_canvas = uv_canvas  # (M,2)
    H=W=256
    bag_mask = np.zeros((H,W), np.uint8)
    for ring in bag_rings_canvas:
        cv2.fillPoly(bag_mask, [np.rint(ring).astype(np.int32)], 1)
    px = np.clip(np.rint(pts_canvas[:,0]).astype(int), 0, W-1)
    py = np.clip(np.rint(pts_canvas[:,1]).astype(int), 0, H-1)
    keep = bag_mask[py, px] > 0
    pts_canvas = pts_canvas[keep]
    roof_kept = roof[keep]

    if len(pts_canvas) < 20:
        # fallback: use all inliers (rare)
        pts_canvas = uv_canvas
        roof_kept = roof

    # 5) Rasterize occupancy + morphology; get largest contour intersecting BAG
    occ = np.zeros((H,W), np.uint8)
    px = np.clip(np.rint(pts_canvas[:,0]).astype(int), 0, W-1)
    py = np.clip(np.rint(pts_canvas[:,1]).astype(int), 0, H-1)
    occ[py, px] = 255
    if dilate>0: occ = cv2.dilate(occ, np.ones((dilate,dilate), np.uint8))
    if close>0:  occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((close,close), np.uint8))

    contour = _largest_contour_intersecting_bag(occ, bag_mask)
    if contour is None or len(contour) < 3:
        # fallback to convex hull of points
        hull = cv2.convexHull(np.rint(pts_canvas).astype(np.int32))
        contour = hull

    # 6) Simplify contour -> polygon in canvas; sample uniformly
    poly = contour[:,0,:].astype(np.float64)  # (K,2)
    poly = _rdp(poly, eps=2.5)                # reuse your RDP
    poly = _dedupe_ring_pixels(poly, px_eps=1.0)
    if len(poly) < 3:
        raise RuntimeError("Contour too small after simplification")

    # 7) Build vertices (with Z) and edges as ring, then OPTIONAL merges (reusing your helpers)
    z_mean = float(np.median(roof_kept[:,2]))  # or per-vertex z by nearest neighbor
    vertices_canvas = np.c_[poly, np.full((len(poly),), z_mean, dtype=np.float64)]  # (V,3)
    edges = []
    V = len(vertices_canvas)
    for i in range(V):
        edges.append([i, (i+1)%V])
    vertices_canvas, edges, _ = merge_vertices_projected(vertices_canvas, np.array(edges, dtype=np.int32), snap_to_int=True)

    # 8) Return in the original 0..255 **image XY** frame (canvas is already 0..255)
    vertices_proj = vertices_canvas.astype(np.float64)
    edges_idx = edges.astype(np.int32)
    return vertices_proj, edges_idx



def merge_adjacent_regions(labels, region_masks):
    region_areas = [np.sum(region_mask) for region_mask in region_masks]

    for label in range(1, labels):
        current_mask = region_masks[label]
        adjacent_labels = set()

        for dy, dx in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
            y, x = np.where(current_mask == 1)
            y_adjacent = np.clip(y + dy, 0, labels.shape[0] - 1)
            x_adjacent = np.clip(x + dx, 0, labels.shape[1] - 1)
            adjacent_labels.update(labels[y_adjacent, x_adjacent])

        adjacent_labels.remove(label)  

        max_area = 0
        max_area_label = None
        for adj_label in adjacent_labels:
            area = region_areas[adj_label]
            if area > max_area:
                max_area = area
                max_area_label = adj_label

        if max_area_label is not None:
            labels[labels == label] = max_area_label

    return labels
def detect_outliers(points, k=5, threshold_factor=20):
   
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(points)
    distances, _ = neigh.kneighbors(points)
    mean_distances = np.mean(distances, axis=1)
    
  
    threshold = threshold_factor * np.mean(mean_distances)

    outliers_idx = np.where(mean_distances > threshold)[0]
    
    return outliers_idx


def render(corners, edges, render_pad=0, edge_linewidth=2, corner_size=3, scale=1.):
    size = int(256 * scale)
    mask = np.ones((2, size, size)) * render_pad

    corners = np.round(corners.copy() * scale).astype(int)
    for edge_i in range(edges.shape[0]):
        a = edges[edge_i, 0]
        b = edges[edge_i, 1]
        mask[0] = cv2.line(mask[0], (int(corners[a, 0]), int(corners[a, 1])),
                           (int(corners[b, 0]), int(corners[b, 1])), 1.0, thickness=edge_linewidth)
    for corner_i in range(corners.shape[0]):
        mask[1] = cv2.circle(mask[1], (int(corners[corner_i, 0]), int(corners[corner_i, 1])), corner_size, 1.0, -1)

    return mask

def convert_annot(annot):
    corners = np.array(list(annot.keys()))
    corners_mapping = {tuple(c): idx for idx, c in enumerate(corners)}
    edges = set()
    for corner, connections in annot.items():
        idx_c = corners_mapping[tuple(corner)]
        for other_c in connections:
            idx_other_c = corners_mapping[tuple(other_c)]
            if (idx_c, idx_other_c) not in edges and (idx_other_c, idx_c) not in edges:
                edges.add((idx_c, idx_other_c))
    edges = np.array(list(edges))
    gt_data = {
        'corners': corners,
        'edges': edges
    }
    return gt_data
# load pc and wireframe
def load_files(pc_dir, train_list_path, test_list_path):
    with open(train_list_path, 'r') as f:
        train_names = [line.strip() for line in f if line.strip()]
    
    all_names = sorted(set(train_names))  # 合并并去重

    pc_files = [name + '/points' + '.ply' for name in all_names]

    return pc_files

def load_wireframe(wireframe_file):
    vertices = []
    edges = set()
    with open(wireframe_file) as f:
        for lines in f.readlines():
            line = lines.strip().split(' ')
            if line[0] == 'v':
                vertices.append(line[1:])
            else:
                if line[0] == '#':
                    continue
                obj_data = np.array(line[1:], dtype=np.int32).reshape(2) - 1
                edges.add(tuple(sorted(obj_data)))
    vertices = np.array(vertices, dtype=np.float64)
    edges = np.array(list(edges))
    return vertices, edges

def load_polygons_json(polygon_file):
    # JSON structure: { "polygons": [ [[x,y], [x,y], ...], ... ] }
    with open(polygon_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print('poly_data: ', data)
    polys = data.get('polygons', [])
    print('polys_data: ', polys)
    polygons = []
    for ring in polys:
        arr = np.array(ring, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
            raise ValueError(f"Invalid polygon ring in {polygon_file}: expected Nx2, got {arr.shape}")
        polygons.append(arr)
    return polygons

def _load_geojson_polygons_from_obj(obj):
    geom_type = obj.get('type', None)
    polygons = []
    if geom_type == 'FeatureCollection':
        features = obj.get('features', [])
        for feat in features:
            geom = feat.get('geometry', {})
            polygons.extend(_load_geojson_polygons_from_obj(geom))
    elif geom_type == 'Feature':
        geom = obj.get('geometry', {})
        polygons.extend(_load_geojson_polygons_from_obj(geom))
    elif geom_type == 'Polygon':
        # coordinates: [ [outer], [hole1], ... ]; we assume no holes
        coords = obj.get('coordinates', [])
        if len(coords) > 0:
            outer = np.array(coords[0], dtype=np.float64)
            if outer.ndim == 2 and outer.shape[1] >= 2 and outer.shape[0] >= 3:
                polygons.append(outer[:, :2])
    elif geom_type == 'MultiPolygon':
        # list of polygons; take each outer ring (index 0)
        mcoords = obj.get('coordinates', [])
        for poly in mcoords:
            if len(poly) > 0:
                outer = np.array(poly[0], dtype=np.float64)
                if outer.ndim == 2 and outer.shape[1] >= 2 and outer.shape[0] >= 3:
                    polygons.append(outer[:, :2])
    else:
        # Not a GeoJSON geometry container we recognize
        pass
    return polygons

def load_geojson_polygons(geojson_file):
    with open(geojson_file, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    return _load_geojson_polygons_from_obj(obj)

def load_any_polygons(polygon_file):
    """
    Load polygons from either our simple JSON schema or a GeoJSON file.
    Returns a list of (N_i x 2) arrays.
    """
    with open(polygon_file, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    if isinstance(obj, dict) and 'polygons' in obj:
        polys = obj.get('polygons', [])
        polygons = []
        for ring in polys:
            arr = np.array(ring, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
                raise ValueError(f"Invalid polygon ring in {polygon_file}: expected Nx2, got {arr.shape}")
            polygons.append(arr)
        return polygons
    else:
        return _load_geojson_polygons_from_obj(obj)

def polygons_to_wireframe(polygons):
    # Deduplicate vertices across rings; build edges and ring index lists
    vertex_index = {}
    vertices = []
    rings_idx = []
    edges = set()
    
    def _clean_ring(ring_arr):
        # Remove repeated closing point and consecutive duplicates
        ring = np.asarray(ring_arr, dtype=np.float64)
        if ring.shape[0] >= 2 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        cleaned = []
        for pt in ring:
            if len(cleaned) == 0 or not np.allclose(pt, cleaned[-1]):
                cleaned.append(pt)
        # If after cleaning the first and last are equal, drop last
        if len(cleaned) >= 2 and np.allclose(cleaned[0], cleaned[-1]):
            cleaned = cleaned[:-1]
        return np.array(cleaned, dtype=np.float64)

    def _simplify_ring_by_angle(ring_arr, angle_eps_deg=3.0, len_eps=1e-9, max_passes=3):
        """
        Remove near-collinear intermediate vertices while preserving corners.
        angle_eps_deg: threshold around 180 degrees; points with turn angle within this
                        tolerance are treated as collinear and removed.
        len_eps: ignore extremely small segments.
        """
        ring = np.asarray(ring_arr, dtype=np.float64)
        if ring.shape[0] <= 3:
            return ring
        def is_collinear(p_prev, p, p_next):
            v1 = p - p_prev
            v2 = p_next - p
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < len_eps or n2 < len_eps:
                return True
            u1 = v1 / n1
            u2 = v2 / n2
            # angle between u1 and u2; 180 deg means straight line (cos ~ -1)
            cos_a = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
            # convert to deviation from 180 degrees: cos(pi - a) = -cos(a)
            # We can check |sin| as indicator of deviation from straight
            sin_a = float(np.linalg.norm(np.cross(np.append(u1,0), np.append(u2,0))))
            # If sin is small, vectors nearly collinear
            from math import sin, radians
            thr = sin(radians(angle_eps_deg))
            return sin_a <= thr
        pts = ring.tolist()
        # ensure closed handling by modular indexing but store as open ring
        for _ in range(max_passes):
            keep = [True] * len(pts)
            changed = False
            m = len(pts)
            if m <= 3:
                break
            for i in range(m):
                p_prev = np.array(pts[(i - 1) % m])
                p = np.array(pts[i])
                p_next = np.array(pts[(i + 1) % m])
                if is_collinear(p_prev, p, p_next):
                    keep[i] = False
            # Ensure we don't remove all points; keep at least 3 and keep extreme if all marked
            new_pts = [pt for k, pt in zip(keep, pts) if k]
            if len(new_pts) < 3:
                # fall back to original if over-pruned
                break
            if len(new_pts) == len(pts):
                break
            pts = new_pts
            changed = True
            if not changed:
                break
        return np.array(pts, dtype=np.float64)

    def get_vid(pt):
        key = (float(pt[0]), float(pt[1]))
        if key in vertex_index:
            return vertex_index[key]
        idx = len(vertices)
        vertex_index[key] = idx
        vertices.append([pt[0], pt[1], 0.0])
        return idx

    for ring in polygons:
        ring = _clean_ring(ring)
        ring = _simplify_ring_by_angle(ring, angle_eps_deg=3.0)
        # skip degenerate rings
        if ring.shape[0] < 3:
            continue
        idxs = [get_vid(pt) for pt in ring]
        rings_idx.append(idxs)
        n = len(idxs)
        for i in range(n):
            a = idxs[i]
            b = idxs[(i + 1) % n]
            if a == b:
                continue
            edges.add(tuple(sorted((a, b))))

    vertices = np.array(vertices, dtype=np.float64)
    edges = np.array(sorted(list(edges)), dtype=np.int32)
    return vertices, edges, rings_idx

def point_in_poly(point, ring):
    # Ray casting; returns True if inside or on edge
    x, y = float(point[0]), float(point[1])
    inside = False
    n = ring.shape[0]
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
        if 0.0 <= t <= 1.0:
            projx = x1 + t * dx
            projy = y1 + t * dy
            if abs(projx - x) <= 1e-6 and abs(projy - y) <= 1e-6:
                return True
        intersect = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1)
        if intersect:
            inside = not inside
    return inside

def point_to_segment_distance(p, a, b):
    ap = p - a
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0.0:
        return float(np.linalg.norm(ap))
    t = float(np.dot(ap, ab)) / denom
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))

def label_points_projected(points_xy, rings_proj, eps_px=1.5):
    N = points_xy.shape[0]
    inside = np.zeros(N, dtype=bool)
    boundary = np.zeros(N, dtype=bool)
    instance_id = np.full(N, -1, dtype=np.int32)

    rings = [np.asarray(r, dtype=np.float64) for r in rings_proj]

    for i in range(N):
        p = points_xy[i]
        for ridx, ring in enumerate(rings):
            if point_in_poly(p, ring):
                inside[i] = True
                instance_id[i] = ridx
                break
        mind = np.inf
        for ring in rings:
            n = ring.shape[0]
            for k in range(n):
                a = ring[k]
                b = ring[(k + 1) % n]
                d = point_to_segment_distance(p, a, b)
                if d < mind:
                    mind = d
            if mind <= eps_px:
                break
        if mind <= eps_px:
            boundary[i] = True

    return {
        'inside': inside,
        'boundary': boundary,
        'instance_id': instance_id
    }

def _rdp(points, eps):
    """
    Ramer–Douglas–Peucker for 2D polyline (numpy array Nx2).
    Returns simplified points in order.
    """
    P = np.asarray(points, dtype=np.float64)
    if P.shape[0] <= 2:
        return P
    # perpendicular distance of point p to segment ab
    def _perp_dist(p, a, b):
        ab = b - a
        if np.allclose(ab, 0):
            return float(np.linalg.norm(p - a))
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))
    # recursive
    a = P[0]
    b = P[-1]
    dmax = -1.0
    idx = -1
    for i in range(1, len(P) - 1):
        d = _perp_dist(P[i], a, b)
        if d > dmax:
            idx = i
            dmax = d
    if dmax > eps:
        left = _rdp(P[: idx + 1], eps)
        right = _rdp(P[idx:], eps)
        return np.vstack([left[:-1], right])
    else:
        return np.vstack([a, b])


def _dedupe_ring_pixels(ring_xy, px_eps=1.0):
    """
    Remove consecutive vertices that fall on the same (rounded) pixel,
    including the wrap-around between the last and first vertex.
    """
    ring = np.asarray(ring_xy, dtype=np.float64)
    if ring.shape[0] < 3:
        return ring

    q = np.rint(ring[:, :2]).astype(int)
    keep = [ring[0]]
    for i in range(1, len(ring)):
        if np.linalg.norm(q[i] - q[i-1]) > px_eps:
            keep.append(ring[i])

    ring = np.array(keep, dtype=np.float64)
    if ring.shape[0] < 3:
        return ring

    # circular check: last vs first
    if np.linalg.norm(np.rint(ring[-1, :2]) - np.rint(ring[0, :2])) <= px_eps:
        ring[0] = (ring[0] + ring[-1]) / 2.0
        ring = ring[:-1]

    return ring

def _simplify_ring_projected(ring_xy, d_min_px=4.0, angle_eps_deg=6.0, rdp_tol_px=3.0):
    """
    Simplify a 2D ring in projected pixel space.
    Steps: merge-nearby (circular) -> RDP -> remove near-180° (circular) -> pixel dedupe
    """
    ring = np.asarray(ring_xy, dtype=np.float64)
    if ring.shape[0] < 3:
        return ring

    # 1) Merge consecutive points that are too close (CIRCULAR)
    tmp = [ring[0]]
    for pt in ring[1:]:
        if np.linalg.norm(pt[:2] - tmp[-1][:2]) > d_min_px:
            tmp.append(pt)
    ring = np.array(tmp, dtype=np.float64)
    if ring.shape[0] >= 2 and np.linalg.norm(ring[0, :2] - ring[-1, :2]) <= d_min_px:
        # average the endpoints to avoid a micro edge at the wrap-around
        ring[0] = (ring[0] + ring[-1]) / 2.0
        ring = ring[:-1]
    if ring.shape[0] < 3:
        return ring

    # 2) RDP simplification (on 2D)
    from shapely.geometry import LineString
    line = LineString(ring[:, :2])
    simplified_line = line.simplify(rdp_tol_px, preserve_topology=False)
    ring = np.array(simplified_line.coords, dtype=np.float64)
    if ring.shape[0] < 3:
        return ring

    # 3) Remove near-collinear vertices (angles ~180°), CIRCULAR
    def interior_angle(a, b, c):
        ba, bc = a - b, c - b
        na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
        if na == 0 or nc == 0:
            return 180.0
        cosang = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)
        return np.degrees(np.arccos(cosang))

    n = len(ring)
    keep = []
    for i in range(n):
        a = ring[(i - 1) % n, :2]
        b = ring[i, :2]
        c = ring[(i + 1) % n, :2]
        ang = interior_angle(a, b, c)
        # drop if nearly straight (within angle_eps of 180°)
        if (180.0 - ang) >= angle_eps_deg:
            keep.append(ring[i])
    ring = np.array(keep, dtype=np.float64)
    if ring.shape[0] < 3:
        return ring

    # 4) Final pixel-space dedupe (circular) to kill double corners
    ring = _dedupe_ring_pixels(ring, px_eps=1.0)

    return ring



def simplify_and_rebuild_from_projected_rings(wf_vertices_proj, rings_idx, d_min_px=4.0, angle_eps_deg=5.0):
    """
    Given projected vertices (Nx3) and ring index lists, simplify each ring in pixel space
    and rebuild a global vertex/edge list. Uses integer pixel rounding to deduplicate
    across rings while preserving ring connectivity.
    Returns (new_vertices_proj (Nx3), new_edges (Mx2), new_rings_idx).
    """
    V = np.asarray(wf_vertices_proj, dtype=np.float64)
    all_new_idxs = []
    key_to_idx = {}
    new_vertices = []
    def get_idx_from_xy(x, y, z):
        key = (int(round(x)), int(round(y)))
        if key in key_to_idx:
            return key_to_idx[key]
        idx = len(new_vertices)
        key_to_idx[key] = idx
        new_vertices.append([key[0], key[1], float(z)])
        return idx
    for idxs in rings_idx:
        ring_xy = V[np.array(idxs, dtype=np.int32), :2]
        simp = _simplify_ring_projected(ring_xy, d_min_px=d_min_px, angle_eps_deg=angle_eps_deg)
        if simp.shape[0] < 3:
            all_new_idxs.append([])
            continue
        ring_new_idx = []
        # use mean Z of original ring for z channel
        z_mean = float(np.mean(V[np.array(idxs, dtype=np.int32), 2])) if V.shape[1] >= 3 else 0.0
        for p in simp:
            ring_new_idx.append(get_idx_from_xy(p[0], p[1], z_mean))
        all_new_idxs.append(ring_new_idx)
    # Build edges
    edges = set()
    for ridx in all_new_idxs:
        n = len(ridx)
        if n < 3:
            continue
        for i in range(n):
            a = ridx[i]
            b = ridx[(i + 1) % n]
            if a == b:
                continue
            if a > b:
                a, b = b, a
            edges.add((a, b))
    new_vertices = np.array(new_vertices, dtype=np.float64)
    new_edges = np.array(sorted(list(edges)), dtype=np.int32) if len(edges) > 0 else np.zeros((0, 2), dtype=np.int32)
    return new_vertices, new_edges, all_new_idxs


def merge_vertices_projected(vertices_proj, edges_idx, snap_to_int=True):
    """
    Merge projected vertices that land on the same pixel (after optional rounding),
    and remap edges accordingly. Removes self-loops and duplicate edges.
    Returns (new_vertices, new_edges, idx_map).
    """
    if vertices_proj.size == 0:
        return vertices_proj, edges_idx, np.arange(0)
    verts = np.asarray(vertices_proj, dtype=np.float64)
    edges = np.asarray(edges_idx, dtype=np.int32)
    if snap_to_int:
        px = np.rint(verts[:, 0]).astype(int)
        py = np.rint(verts[:, 1]).astype(int)
    else:
        px = verts[:, 0]
        py = verts[:, 1]
    keys = list(zip(px.tolist(), py.tolist()))

    key_to_new = {}
    new_vertices = []
    for i, key in enumerate(keys):
        if key in key_to_new:
            continue
        key_to_new[key] = len(new_vertices)
        new_vertices.append([key[0], key[1], verts[i, 2]])

    idx_map = np.empty(len(verts), dtype=np.int32)
    for i, key in enumerate(keys):
        idx_map[i] = key_to_new[key]

    new_edges_set = set()
    for e in edges:
        a = int(idx_map[int(e[0])])
        b = int(idx_map[int(e[1])])
        if a == b:
            continue
        if a > b:
            a, b = b, a
        new_edges_set.add((a, b))

    new_vertices = np.array(new_vertices, dtype=np.float64)
    new_edges = np.array(sorted(list(new_edges_set)), dtype=np.int32) if len(new_edges_set) > 0 else np.zeros((0,2), dtype=np.int32)
    return new_vertices, new_edges, idx_map

def merge_vertices_eps(vertices_proj, edges_idx, eps_px=2.0):
    """
    Merge vertices whose projected XY are within eps_px distance.
    Uses grid hashing + union-find to cluster, then averages cluster coordinates.
    Returns (new_vertices, new_edges, idx_map).
    """
    verts = np.asarray(vertices_proj, dtype=np.float64)
    edges = np.asarray(edges_idx, dtype=np.int32)
    n = verts.shape[0]
    if n == 0:
        return verts, edges, np.arange(0)
    # Build grid bins
    gx = np.floor(verts[:, 0] / eps_px).astype(int)
    gy = np.floor(verts[:, 1] / eps_px).astype(int)
    bins = {}
    for i, (bx, by) in enumerate(zip(gx, gy)):
        bins.setdefault((bx, by), []).append(i)

    parent = list(range(n))
    rank = [0] * n

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    eps2 = float(eps_px) * float(eps_px)
    # Check within each bin and 8 neighbors
    for (bx, by), idxs in bins.items():
        neighbor_bins = [(bx+dx, by+dy) for dx in (-1,0,1) for dy in (-1,0,1)]
        for nb in neighbor_bins:
            cand = bins.get(nb)
            if not cand:
                continue
            for i in idxs:
                vi = verts[i, :2]
                for j in cand:
                    if j <= i:
                        continue
                    vj = verts[j, :2]
                    if ((vi[0]-vj[0])**2 + (vi[1]-vj[1])**2) <= eps2:
                        union(i, j)

    # Build clusters
    clusters = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, []).append(i)

    # Create new vertices as mean of cluster members (XY), Z as mean too
    new_index = {}
    new_vertices = []
    for new_id, (root, members) in enumerate(clusters.items()):
        pts = verts[members]
        mean = pts.mean(axis=0)
        new_vertices.append(mean.tolist())
        for m in members:
            new_index[m] = new_id

    idx_map = np.array([new_index[i] for i in range(n)], dtype=np.int32)

    # Remap edges, drop self-loops and duplicates
    new_edges_set = set()
    for e in edges:
        a = int(idx_map[int(e[0])])
        b = int(idx_map[int(e[1])])
        if a == b:
            continue
        if a > b:
            a, b = b, a
        new_edges_set.add((a, b))

    new_vertices = np.array(new_vertices, dtype=np.float64)
    new_edges = np.array(sorted(list(new_edges_set)), dtype=np.int32) if len(new_edges_set) > 0 else np.zeros((0,2), dtype=np.int32)
    return new_vertices, new_edges, idx_map

def collapse_short_edges(vertices_proj, edges_idx, min_len_px=4.0):
    """
    Collapse edges shorter than min_len_px by merging their endpoints (mean position).
    Uses union-find across all short edges, then rebuilds the graph.
    Returns (new_vertices, new_edges, idx_map)
    """
    V = np.asarray(vertices_proj, dtype=np.float64)
    E = np.asarray(edges_idx, dtype=np.int32)
    if V.size == 0 or E.size == 0:
        return V, E, np.arange(V.shape[0], dtype=np.int32)
    n = V.shape[0]
    parent = list(range(n))
    rank = [0] * n

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    thr2 = float(min_len_px) * float(min_len_px)
    for a, b in E:
        a = int(a); b = int(b)
        if a == b:
            continue
        dx = V[a, 0] - V[b, 0]
        dy = V[a, 1] - V[b, 1]
        if dx*dx + dy*dy <= thr2:
            union(a, b)

    # Build clusters and compute mean positions
    clusters = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, []).append(i)

    new_index = {}
    new_vertices = []
    for new_id, (root, members) in enumerate(clusters.items()):
        pts = V[members]
        mean = pts.mean(axis=0)
        new_vertices.append(mean.tolist())
        for m in members:
            new_index[m] = new_id

    idx_map = np.array([new_index[i] for i in range(n)], dtype=np.int32)

    # Remap edges, drop self-loops and duplicates
    new_edges_set = set()
    for e in E:
        a = int(idx_map[int(e[0])])
        b = int(idx_map[int(e[1])])
        if a == b:
            continue
        if a > b:
            a, b = b, a
        new_edges_set.add((a, b))

    new_vertices = np.array(new_vertices, dtype=np.float64)
    new_edges = np.array(sorted(list(new_edges_set)), dtype=np.int32) if len(new_edges_set) > 0 else np.zeros((0,2), dtype=np.int32)
    return new_vertices, new_edges, idx_map

# --- helpers ---------------------------------------------------------------
def _build_adj(E):
    from collections import defaultdict
    adj = defaultdict(list)
    for a,b in E:
        a=int(a); b=int(b)
        adj[a].append(b); adj[b].append(a)
    return adj

def _line_from_pts(p, q):
    # returns (u, v) s.t. line(t) = u + t*v
    u = p[:2].astype(float)
    v = (q[:2] - p[:2]).astype(float)
    n = np.linalg.norm(v)
    if n == 0.0:
        return u, None
    return u, v / n

def _angle_between(v1, v2):
    c = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return np.degrees(np.arccos(c))

def _intersect_lines(u1, v1, u2, v2):
    # Solve u1 + t*v1 = u2 + s*v2  ->  t = cross(u2-u1, v2) / cross(v1, v2)
    # using 2D scalar cross (z of 3D cross)
    if v1 is None or v2 is None:
        return None
    def cross(a,b): return a[0]*b[1] - a[1]*b[0]
    den = cross(v1, v2)
    if abs(den) < 1e-8:
        return None
    t = cross(u2 - u1, v2) / den
    P = u1 + t * v1
    return P  # 2D

def _choose_endpoint_by_external_length(V, E, members, adj):
    mset = set(members)
    best = members[0]
    best_score = -1.0
    for m in members:
        score = 0.0
        for n in adj.get(int(m), []):
            if n in mset:  # ignore internal cluster edges
                continue
            score += np.linalg.norm(V[m,:2] - V[n,:2])
        if score > best_score:
            best_score = score
            best = m
    return best

# --- smart adaptive short-edge collapse with intersection snapping ----------
def collapse_short_edges_adaptive(vertices_proj, edges_idx, base_px=4.0, alpha=0.20, verbose=False, snap_endpoint=True):
    """
    Adaptive short-edge collapse with geometric snapping.
    - Compute tau = max(base_px, alpha * median_edge_length)
    - Union vertices linked by edges <= tau
    - For each cluster (>1 vertex), try to snap to the intersection of two
      dominant external lines; if not possible, snap to best endpoint.
    """
    V = np.asarray(vertices_proj, dtype=np.float64)
    E = np.asarray(edges_idx, dtype=np.int32)

    if V.size == 0 or E.size == 0:
        return V, E, np.arange(V.shape[0], dtype=np.int32)

    # lengths & threshold
    L = np.linalg.norm(V[E[:,0],:2] - V[E[:,1],:2], axis=1)
    if L.size == 0:
        return V, E, np.arange(V.shape[0], dtype=np.int32)
    tau = float(max(base_px, alpha * float(np.median(L))))
    if verbose:
        print(f"[adaptive] median={np.median(L):.2f}px  tau={tau:.2f}px")

    # union-find for edges <= tau
    n = V.shape[0]
    parent = list(range(n)); rank = [0]*n
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra == rb: return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra; rank[ra] += 1

    tau2 = tau*tau
    for a,b in E:
        a=int(a); b=int(b)
        dx = V[a,0]-V[b,0]; dy = V[a,1]-V[b,1]
        if dx*dx + dy*dy <= tau2:
            union(a,b)

    # clusters
    clusters = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, []).append(i)

    adj = _build_adj(E)

    # build new vertices by snapping
    new_index = {}
    new_vertices = []
    for new_id, (root, members) in enumerate(clusters.items()):
        if len(members) == 1:
            m = members[0]
            new_vertices.append(V[m].tolist())
            new_index[m] = new_id
            continue

        # collect external neighbor directions from all members
        dirs = []   # (origin_id, neighbor_id, dir_vec(unit), length)
        mset = set(members)
        for m in members:
            for nb in adj.get(int(m), []):
                if nb in mset:  # internal
                    continue
                u, v = _line_from_pts(V[m], V[nb])
                if v is None: 
                    continue
                length = np.linalg.norm(V[m,:2] - V[nb,:2])
                dirs.append((m, nb, v, length))

        snapped_pt = None
        if len(dirs) >= 2:
            # pick two strongest non-parallel directions (by length, with angle gap)
            dirs_sorted = sorted(dirs, key=lambda x: x[3], reverse=True)
            picked = []
            for cand in dirs_sorted:
                if not picked:
                    picked.append(cand)
                else:
                    ok = True
                    for p in picked:
                        if _angle_between(cand[2], p[2]) < 12.0:  # avoid near-parallel
                            ok = False; break
                    if ok:
                        picked.append(cand)
                if len(picked) == 2:
                    break

            if len(picked) == 2:
                (m1, n1, v1, _), (m2, n2, v2, _) = picked
                u1, _ = _line_from_pts(V[m1], V[n1])
                u2, _ = _line_from_pts(V[m2], V[n2])
                P = _intersect_lines(u1, v1, u2, v2)
                if P is not None and np.all(np.isfinite(P)):
                    # keep Z from the best endpoint (longer external edges)
                    best = _choose_endpoint_by_external_length(V, E, members, adj)
                    z = V[best, 2] if V.shape[1] >= 3 else 0.0
                    snapped_pt = [float(P[0]), float(P[1]), float(z)]

        if snapped_pt is None:
            # fallback: snap to best endpoint among cluster
            best = _choose_endpoint_by_external_length(V, E, members, adj)
            snapped_pt = V[best].tolist()

        new_vertices.append(snapped_pt)
        for m in members:
            new_index[m] = new_id

    idx_map = np.array([new_index[i] for i in range(n)], dtype=np.int32)

    # remap edges, drop self-loops/dupes
    new_edges = set()
    for a,b in E:
        A = int(idx_map[int(a)]); B = int(idx_map[int(b)])
        if A == B: 
            continue
        if A > B: 
            A,B = B,A
        new_edges.add((A,B))

    new_vertices = np.array(new_vertices, dtype=np.float64)
    new_edges = np.array(sorted(list(new_edges)), dtype=np.int32) if len(new_edges) > 0 else np.zeros((0,2), dtype=np.int32)
    return new_vertices, new_edges, idx_map



def build_polygon_files(train_list_path, test_list_path, poly_dir="./"):
    with open(train_list_path, 'r') as f:
        train_names = [line.strip() for line in f if line.strip()]
    all_names = sorted(set(train_names))
    
    
    
    files = []
    for name in all_names:
        p_geo = name + '/polygon' + '.geojson'
        p_json = name + '/polygon' + '.json'
        if os.path.exists(p_geo):
            files.append(p_geo)
        elif os.path.exists(p_json):
            files.append(p_json)
        else:
            # default to .json path; will error later if missing
            files.append(p_json)
    return files

def proj_img(pc, index, output_dir):
    """
    Rasterize a max-height (roof) image from the point cloud.
    Assumes pc[:,0:3] are already normalized to [0,255] in proj.py.
    """
    H = W = 256
    image = np.zeros((H, W, 3), dtype=np.uint8)

    # pixel coords
    x_pixels = np.floor(pc[:, 0]).astype(int)
    y_pixels = np.floor(pc[:, 1]).astype(int)

    # clip to bounds
    x_pixels = np.clip(x_pixels, 0, W - 1)
    y_pixels = np.clip(y_pixels, 0, H - 1)

    # iterate and keep the **maximum** z per pixel (roof DSM)
    for i in range(len(pc)):
        y, x = y_pixels[i], x_pixels[i]
        z = int(round(pc[i, 2]))          # z is already in 0..255 after your normalization
        if z > image[y, x, 0]:             # compare with current stored height
            image[y, x] = (z, z, z)

    cv2.imwrite(os.path.join(output_dir, f"{index}.jpg"), image)
    return image


def proj_maskimg(pc, index, output_dir):

    x_pixels = np.floor(pc[:, 0]).astype(int)
    y_pixels = np.floor(pc[:, 1]).astype(int)

    image = np.zeros((256, 256, 3), dtype=np.uint8)


    for i in range(len(pc)):
        
        image[y_pixels[i], x_pixels[i]] = [pc[:, 2], pc[:, 2], pc[:, 2]]


    cv2.imwrite(os.path.join(output_dir, f"{index}.jpg"), image)

def visualize_image(image, data_dict, output_dir, index):
    image = image.copy()  # get a new copy of the original image

    for vertex, connected_vertices in data_dict.items():

        x, y, _ = vertex
        cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)  # 红色圆点

    for vertex, connected_vertices in data_dict.items():

        x, y, _ = vertex

        for connected_vertex in connected_vertices:
            x2, y2, _ = connected_vertex
            cv2.line(image, (int(x2), int(y2)), (int(x), int(y)), (255, 0, 0), 2)  # 蓝色线段

    cv2.imwrite(os.path.join(output_dir, f"{index}_vis.jpg"), image)

def main():
    pc_dir = "./"
    train_list_path = "./train_list.txt"
    test_list_path = "xxx/test_list.txt"
    
    pc_files = load_files(pc_dir, train_list_path, test_list_path)
    polygon_files = build_polygon_files(train_list_path, test_list_path, poly_dir="./")
    
    names = []
    image_id = 0

    for index, path in enumerate(pc_files):
        base_path = path[:-10]
        print('idx: ',index,' - path: ', base_path)
        image_id = image_id + 1
        pc_file = pc_files[index]
        pcd = o3d.io.read_point_cloud(pc_file)  # read the point cloud file
        point_cloud = np.asarray(pcd.points) 
        name =  os.path.splitext(os.path.basename(pc_file))[0]
        print('point cloud name >> ',name)
    
        names.append(name)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        # ------------------------------- Polygons ------------------------------
        # load polygons (no holes) from simple JSON or GeoJSON, then convert to wireframe
        polygon_file = polygon_files[index]
        
        # 1) Load BAG polygon (only for guidance)
        polygons = load_any_polygons(polygon_file)          # list of rings in world/image XY
        # bring to projected frame you already use (0..255); in your code wf_vertices later get normalized
        # We first normalize point_cloud to 0..255 as you already do (centroid/max_distance code).
        # After you compute:
        #   centroid / max_distance
        #   point_cloud[:,0:3] -> normalized to 0..255  (as in your script)
        # then map BAG rings the same way:

        centroid = np.mean(point_cloud[:, :3], axis=0)
        point_cloud[:, :3] -= centroid

        # --- scale using XY only + padding ---
        r = np.linalg.norm(point_cloud[:, :2], axis=1)
        max_distance = np.percentile(r, 99.5)

        PAD_SCALE = 0.90  # ~5% margin on each side
        point_cloud[:, :2] = (point_cloud[:, :2] / max_distance) * (127.5 * PAD_SCALE)

        # --- shift to 0..255 canvas and clip ---
        point_cloud[:, :3] += 127.5
        point_cloud[:, :2] = np.clip(point_cloud[:, :2], 0, 255)
        point_cloud[:,  2] = np.clip(point_cloud[:,  2], 0, 255)




        rings_proj = []
        for ring in polygons:
            ring = ring - centroid[:2]
            ring = (ring / max(max_distance, 1e-6)) * (127.5 * PAD_SCALE)
            ring = ring + 127.5
            ring = np.clip(ring, 0, 255)
            rings_proj.append(ring)

        # Per-point labeling (projected pixel frame)
        points_xy_proj = point_cloud[:, :2].copy()              # (N,2) in 0..255 frame
        labels = label_points_projected(points_xy_proj, rings_proj, eps_px=1.5)


        # 2) Build wireframe FROM POINTS under BAG supervision
        wf_vertices, wf_edges = wireframe_from_pointcloud(point_cloud[:,0:3], rings_proj)

        # (optional) compact/safety merges with your existing helpers
        v_before, e_before = int(wf_vertices.shape[0]), int(wf_edges.shape[0])
        wf_vertices, wf_edges, _ = merge_vertices_eps(wf_vertices, wf_edges, eps_px=4.0)
        wf_vertices, wf_edges, _ = collapse_short_edges_adaptive(wf_vertices, wf_edges, base_px=4.0, alpha=0.20, verbose=False, snap_endpoint=True)
        wf_vertices, wf_edges, _ = merge_vertices_projected(wf_vertices, wf_edges, snap_to_int=True)
        v_after, e_after = int(wf_vertices.shape[0]), int(wf_edges.shape[0])

        # 3) Build annot dict from this point-derived wireframe (unchanged)
        vertex_con = defaultdict(set)
        for a,b in wf_edges:
            a=int(a); b=int(b)
            if a!=b: vertex_con[a].add(b); vertex_con[b].add(a)
        vertex_connections1 = {}
        for v_idx, neighs in vertex_con.items():
            key = (wf_vertices[v_idx][0], wf_vertices[v_idx][1], wf_vertices[v_idx][2])
            vertex_connections1[key] = [np.array([wf_vertices[n][0], wf_vertices[n][1], wf_vertices[n][2]]) for n in sorted(list(neighs))]

        # 4) Save annotation — now **from point cloud**
        npypath = base_path + f"{name}.npy"
        combined = {
            'annot': vertex_connections1,
            'point_labels': labels,              # you already computed from rings_proj (okay to keep)
            'vertices_proj': wf_vertices,        # <- point cloud derived
            'edges_idx': wf_edges,               # <- point cloud derived
            'integrity': {
                'vertices_before_merge': v_before,
                'edges_before_merge': e_before,
                'vertices_after_merge': v_after,
                'edges_after_merge': e_after,
            },
        }
        np.save(npypath, combined)


        # show_npy(npypath)
        image = proj_img(point_cloud, name, base_path)
        
         # -------------------- Configuration --------------------
        IMG_PATH = base_path + f"{name}.jpg"       # top-down point cloud / height image
        OUT_PATH = base_path +  "wireframe_overlay.png" # where overlay will be saved
        # -------------------------------------------------------

        get_wireframe_overlay(npypath,IMG_PATH, OUT_PATH)
         
def get_wireframe_overlay(NPY_PATH,IMG_PATH, OUT_PATH):
    def collect_graph(annot_dict):
        """Convert adjacency dictionary into vertex/edge arrays."""
        verts = set()
        edges = set()
        for k, neighs in annot_dict.items():
            # ensure tuple of 3 floats
            if isinstance(k, (list, np.ndarray)):
                k = tuple(map(float, k))
            else:
                k = tuple(k)
            verts.add(k)
            for n in neighs:
                n = tuple(map(float, n))
                verts.add(n)
                a, b = k, n
                edges.add(tuple(sorted((a, b))))
        verts = list(verts)
        idx = {v: i for i, v in enumerate(verts)}
        E = [(idx[a], idx[b]) for (a, b) in edges if idx[a] != idx[b]]
        return np.array(verts, dtype=float), np.array(E, dtype=int)

    # -------------------- Load data --------------------
    obj = np.load(NPY_PATH, allow_pickle=True).item()
    V = None
    E = None
    if isinstance(obj, dict) and 'vertices_proj' in obj and 'edges_idx' in obj:
        V = np.asarray(obj['vertices_proj'], dtype=float)
        E = np.asarray(obj['edges_idx'], dtype=int)
    else:
        if isinstance(obj, dict) and "annot" in obj:
            annot = obj["annot"]
        else:
            annot = obj
        V, E = collect_graph(annot)

    # Final dedup for overlay: first epsilon-cluster in pixel space, then collapse by rounding
    if V is not None and V.size > 0:
        # Ensure V has z column
        if V.shape[1] == 2:
            V = np.concatenate([V, np.zeros((V.shape[0], 1), dtype=V.dtype)], axis=1)
        # Epsilon-based merge in overlay (robust against near duplicates)
        V, E, _ = merge_vertices_eps(V, E, eps_px=4.0)
        # Collapse short edges in overlay too
        V, E, _ = collapse_short_edges_adaptive(V, E, base_px=4.0, alpha=0.20, verbose=False, snap_endpoint=True)
        px = np.rint(V[:, 0]).astype(int)
        py = np.rint(V[:, 1]).astype(int)
        key_to_new = {}
        new_V = []
        for i, (x, y) in enumerate(zip(px, py)):
            key = (int(x), int(y))
            if key not in key_to_new:
                key_to_new[key] = len(new_V)
                # keep Z if present, else 0
                z = V[i, 2] if V.shape[1] >= 3 else 0.0
                new_V.append([x, y, z])
        idx_map = np.array([key_to_new[(int(x), int(y))] for x, y in zip(px, py)], dtype=int)
        new_edges = set()
        for a, b in E:
            a2 = int(idx_map[int(a)])
            b2 = int(idx_map[int(b)])
            if a2 == b2:
                continue
            if a2 > b2:
                a2, b2 = b2, a2
            new_edges.add((a2, b2))
        V = np.array(new_V, dtype=float)
        E = np.array(sorted(list(new_edges)), dtype=int) if len(new_edges) > 0 else np.zeros((0,2), dtype=int)

    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {IMG_PATH}")
    H, W = img.shape[:2]

    # -------------------- Overlay drawing --------------------
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def in_img(pt):
        return 0 <= pt[0] < W and 0 <= pt[1] < H

    # Draw edges
    for (a, b) in E:
        pa = (int(round(V[a, 0])), int(round(V[a, 1])))
        pb = (int(round(V[b, 0])), int(round(V[b, 1])))
        if in_img(pa) and in_img(pb):
            cv2.line(overlay, pa, pb, (255, 255, 255), 1)

    # Draw vertices
    for i in range(len(V)):
        p = (int(round(V[i, 0])), int(round(V[i, 1])))
        if in_img(p):
            cv2.circle(overlay, p, 2, (255, 255, 255), -1)

    cv2.imwrite(OUT_PATH, overlay)
    print(f"[✓] Overlay saved to: {OUT_PATH}")
    print(f"Vertices: {len(V)} | Edges: {len(E)} | Image size: {H}×{W}")


def show_npy(file_path):
    # Load the uploaded npy file again
    data = np.load(file_path, allow_pickle=True).item()

    # Extract number of vertices and edges
    num_vertices = len(data)
    edge_count = sum(len(v) for v in data.values()) // 2  # divide by 2 to avoid double counting

    print('no. of vertices: ',num_vertices,' - no of edges: ', edge_count)

        
if __name__ == '__main__':
    main()
        
