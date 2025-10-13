import os
import sys
import glob
import numpy as np
from collections import defaultdict, deque
import cv2
import json
import open3d as o3d
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import NearestNeighbors

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


def build_polygon_files(train_list_path, test_list_path, poly_dir="./"):
    with open(train_list_path, 'r') as f:
        train_names = [line.strip() for line in f if line.strip()]
    # with open(test_list_path, 'r') as f:
    #     test_names = [line.strip() for line in f if line.strip()]
    all_names = sorted(set(train_names))
    
    
    
    files = []
    for name in all_names:
        p_geo = name + '/polygon' + '.geojson'
        p_json = name + '/polygon' + '.json'
        
        print(p_geo)
        if os.path.exists(p_geo):
            files.append(p_geo)
        elif os.path.exists(p_json):
            files.append(p_json)
        else:
            # default to .json path; will error later if missing
            files.append(p_json)
    return files

def proj_img(pc, index, output_dir):

    x_pixels = np.floor(pc[:, 0]).astype(int)
    y_pixels = np.floor(pc[:, 1]).astype(int)


    image = np.zeros((256, 256, 3), dtype=np.uint8)


    for i in range(len(pc)):
        if image[y_pixels[i], x_pixels[i]][0] == 0:
            image[y_pixels[i], x_pixels[i]] = [pc[i, 2], pc[i, 2], pc[i, 2]]
        else:
            if pc[i, 2] < image[y_pixels[i], x_pixels[i]][0]:
                image[y_pixels[i], x_pixels[i]] = [pc[i, 2], pc[i, 2], pc[i, 2]]


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
    proj_dir = "./training_data/rgb"
    npy_dir = "./training_data/annot"
    train_list_path = "./train_list.txt"
    test_list_path = "xxx/test_list.txt"
    
    pc_files = load_files(pc_dir, train_list_path, test_list_path)
    polygon_files = build_polygon_files(train_list_path, test_list_path, poly_dir="./")
    
    names = []
    annotations = []
    annotation_id = 0
    images = []
    image_id = 0

    for index, path in enumerate(pc_files):
        base_path = path[:-10]
        print('idx: ',index,' - path: ', base_path)
        image_id = image_id + 1
        pc_file = pc_files[index]
        pcd = o3d.io.read_point_cloud(pc_file)  # 读取 .ply 文件
        point_cloud = np.asarray(pcd.points) 
        name =  os.path.splitext(os.path.basename(pc_file))[0]
        print('point cloud name >> ',name)
    
        names.append(name)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           

         # ------------------------------- Polygons ------------------------------
        # load polygons (no holes) from simple JSON or GeoJSON, then convert to wireframe
        polygon_file = polygon_files[index]
        polygons = load_any_polygons(polygon_file)
        wf_vertices, wf_edges, rings_idx = polygons_to_wireframe(polygons)

        
        centroid = np.mean(point_cloud[:, 0:3], axis=0)
        point_cloud[:, 0:3] -= centroid
        wf_vertices -= centroid
        max_distance = np.max(np.linalg.norm(np.vstack((point_cloud[:, 0:3], wf_vertices)), axis=1))

                
        point_cloud[:, 0:3] /= (max_distance)
        point_cloud[:, 0:3] = (point_cloud[:, 0:3] +  np.ones_like(point_cloud[:, 0:3]))  * 127.5

        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(point_cloud)

        wf_vertices /= (max_distance)

        wf_vertices = (wf_vertices +  np.ones_like(wf_vertices))  *  127.5

        # --------------------------- Per-point labeling ---------------------------
        points_xy_proj = point_cloud[:, :2].copy()
        rings_proj = []
        for idxs in rings_idx:
            ring_xy = wf_vertices[np.array(idxs, dtype=np.int32), :2]
            rings_proj.append(ring_xy)
        labels = label_points_projected(points_xy_proj, rings_proj, eps_px=1.5)

        # Integrity counts before merge
        v_before = int(wf_vertices.shape[0])
        e_before = int(wf_edges.shape[0])

        # Merge projected vertices that are within epsilon pixels to avoid duplicates
        wf_vertices, wf_edges, _ = merge_vertices_eps(wf_vertices, wf_edges, eps_px=2.0)

        # Integrity counts after merge
        v_after = int(wf_vertices.shape[0])
        e_after = int(wf_edges.shape[0])

        # Print and log integrity summary
        try:
            print(f"[integrity] {name}: V {v_before}->{v_after}, E {e_before}->{e_after}")
            log_path = os.path.join(npy_dir, "integrity_log.csv")
            if not os.path.exists(log_path):
                with open(log_path, 'w', encoding='utf-8') as lf:
                    lf.write("name,vertices_before,edges_before,vertices_after,edges_after\n")
            with open(log_path, 'a', encoding='utf-8') as lf:
                lf.write(f"{name},{v_before},{e_before},{v_after},{e_after}\n")
        except Exception:
            pass

        vertex_con = defaultdict(set)

        for edge in wf_edges:
            vertex1, vertex2 = int(edge[0]), int(edge[1])
            if vertex1 == vertex2:
                continue
            if vertex1 < 0 or vertex2 < 0 or vertex1 >= wf_vertices.shape[0] or vertex2 >= wf_vertices.shape[0]:
                continue
            vertex_con[vertex1].add(vertex2)
            vertex_con[vertex2].add(vertex1)

        vertex_connections1 = {}
        for vertex, neighbors in vertex_con.items():
            key = tuple([wf_vertices[vertex][0], wf_vertices[vertex][1], wf_vertices[vertex][2]])
            vertex_connections1[key] = []
            for edge_vertex in sorted(list(neighbors)):
                vertex_connections1[key].append(np.array([
                    wf_vertices[edge_vertex][0],
                    wf_vertices[edge_vertex][1],
                    wf_vertices[edge_vertex][2]
                ]))
        npypath = base_path + f"{name}.npy"
        combined = {
            'annot': vertex_connections1,
            'point_labels': labels,
            'vertices_proj': wf_vertices,     # projected to 0..255 coords
            'edges_idx': wf_edges,            # 0-based indices into vertices_proj
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
    if isinstance(obj, dict) and "annot" in obj:
        annot = obj["annot"]
    else:
        annot = obj

    V, E = collect_graph(annot)

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
        
