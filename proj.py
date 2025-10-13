import os
import sys
import glob
import numpy as np
from collections import defaultdict
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
    wf_dir = "xxx/wf"

    with open(train_list_path, 'r') as f:
        train_names = [line.strip() for line in f if line.strip()]
    with open(test_list_path, 'r') as f:
        test_names = [line.strip() for line in f if line.strip()]
    
    all_names = sorted(set(train_names + test_names))  # 合并并去重

    pc_files = [os.path.join(pc_dir, name + '.ply') for name in all_names]
    wireframe_files = [os.path.join(wf_dir,  name + '.obj') for name in all_names]

    return pc_files, wireframe_files

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
    polys = data.get('polygons', [])
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

def build_polygon_files(train_list_path, test_list_path, poly_dir="xxx/poly"):
    with open(train_list_path, 'r') as f:
        train_names = [line.strip() for line in f if line.strip()]
    with open(test_list_path, 'r') as f:
        test_names = [line.strip() for line in f if line.strip()]
    all_names = sorted(set(train_names + test_names))
    files = []
    for name in all_names:
        p_geo = os.path.join(poly_dir, name + '.geojson')
        p_json = os.path.join(poly_dir, name + '.json')
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
    pc_dir = "xxx/pc"
    proj_dir = "xxx/rgb"
    npy_dir = "xxx/annot"
    train_list_path = "xxx/train_list.txt"
    test_list_path = "xxx/test_list.txt"
    
    pc_files, wireframe_files = load_files(pc_dir, train_list_path, test_list_path)
    polygon_files = build_polygon_files(train_list_path, test_list_path, poly_dir="xxx/poly")
    names = []
    annotations = []
    annotation_id = 0
    images = []
    image_id = 0

    for index in range(len(pc_files)):
        print(index)
        image_id = image_id + 1
        pc_file = pc_files[index]
        pcd = o3d.io.read_point_cloud(pc_file)  # 读取 .ply 文件
        point_cloud = np.asarray(pcd.points) 
        name =  os.path.splitext(os.path.basename(pc_file))[0]
        print(name)
    
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
        npypath = os.path.join(npy_dir, f"{name}.npy")
        combined = {
            'annot': vertex_connections1,
            'point_labels': labels,
            'vertices_proj': wf_vertices,     # projected to 0..255 coords
            'edges_idx': wf_edges            # 0-based indices into vertices_proj
        }
        np.save(npypath, combined)
        image = proj_img(point_cloud, name, proj_dir)
        
if __name__ == '__main__':
    main()
        



