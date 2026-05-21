import random

import numpy as np
from scipy import ndimage
from scipy.interpolate import CubicSpline
from scipy.ndimage import binary_dilation, convolve, label
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree, distance_matrix
from skimage.morphology import ball, closing, skeletonize


def skeletonize_with_fallback(binary_map_dilated, processed_segment, voxel_threshold=100):
    skel = skeletonize(binary_map_dilated).astype(np.uint8)
    if np.count_nonzero(skel) < voxel_threshold:
        #print("Fallback triggered")
        processed_fallback = closing(processed_segment, ball(3))
        processed_fallback = binary_dilation(processed_fallback, iterations=1)
        skel_fallback = skeletonize(processed_fallback).astype(np.uint8)
        if np.count_nonzero(skel_fallback) >= voxel_threshold:
            #print("Fallback successful")
            return skel_fallback
        else:
            #print("Fallback failed")
            return None
    return skel


def generate_skeleton(binary_map, dilation_structure=(5,5,5), skeleton_dilate_iters=1):
    structure = np.ones(dilation_structure)
    binary_map_dilated = ndimage.binary_dilation(binary_map, structure=structure).astype(binary_map.dtype)
    original_skeleton = skeletonize(binary_map_dilated).astype(np.uint8)
    skeleton_dilated = binary_dilation(original_skeleton.astype(bool), iterations=skeleton_dilate_iters).astype(np.uint8)
    return skeleton_dilated

def generate_skeleton_v2(binary_map):
    """
    与原始 notebook 中的骨架提取逻辑完全一致。

    步骤：
    1. 对 binary_map 执行 2 次膨胀
    2. skeletonize_3d 得到骨架
    3. 对骨架执行 1 次膨胀

    参数:
        binary_map (np.ndarray): 输入的三维二值掩膜

    返回:
        skeleton (np.ndarray): 骨架图像（0 和 1）
    """
    binary_map_dilated = ndimage.binary_dilation(binary_map, iterations=2).astype(binary_map.dtype)
    skeleton = skeletonize(binary_map_dilated).astype(np.uint8)
    #print(f"Skeleton voxel count: {np.count_nonzero(skeleton)}")
    skeleton = ndimage.binary_dilation(skeleton.astype(bool), iterations=1).astype(np.uint8)
    return skeleton

def keep_largest_connected_component(binary_array):
    """
    保留输入 3D 二值图像中最大的连通区域，其余全部置零。

    参数:
        binary_array (np.ndarray): 输入的三维二值掩膜（通常为骨架）。

    返回:
        cleaned (np.ndarray): 仅包含最大连通区域的掩膜。
    """
    labeled_array, num_features = label(binary_array)

    if num_features < 1:
        return np.zeros_like(binary_array)  # 全部清空

    # 计算每个连通块的体素数
    counts = [(labeled_array == i).sum() for i in range(1, num_features + 1)]
    largest_label = np.argmax(counts) + 1
    #print(f"Largest component label: {largest_label}, size: {counts[largest_label-1]} voxels")

    cleaned = (labeled_array == largest_label).astype(np.uint8)
    return cleaned

# --- 端点与 peak ---

#for left and right 
def find_endpoints(centerline_img):
    points = np.argwhere(centerline_img > 0)
    if len(points) < 2:
        raise ValueError("Skeleton too small (<2 points)")

    from scipy.spatial.distance import cdist
    D = cdist(points, points)
    i, j = np.unravel_index(D.argmax(), D.shape)
    return [tuple(points[i]), tuple(points[j])]


def find_endpoints_v2(centerline_img, k=50, max_distance=10.0):
    """
    从中心线图像中找到两端点（start, end），用于路径提取。
    若正常 endpoint 数量不足，则回退为最远点对。

    参数:
        centerline_img (np.ndarray): 二值化的骨架图像。
        k (int): 邻居数量，用于构建 kNN 图。
        max_distance (float): 距离阈值，构建稀疏图的边。

    返回:
        List[Tuple[int, int, int]]: 选中的起点和终点。
    """
    points = np.argwhere(centerline_img > 0)
    if len(points) < 2:
        raise ValueError("Skeleton too small (<2 points)")

    # 计算度为 1 的端点
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    neighbor_count = convolve((centerline_img > 0).astype(np.uint8), kernel, mode='constant', cval=0)
    endpoint_mask = (centerline_img > 0) & (neighbor_count == 2)
    endpoints = np.argwhere(endpoint_mask)
    if len(endpoints) < 2:
        endpoints = points  # fallback

    # 构建图
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=k)
    mask = distances <= max_distance

    rows, cols, data = [], [], []
    for i in range(len(points)):
        valid = mask[i, 1:]
        if np.any(valid):
            rows.extend([i] * np.sum(valid))
            cols.extend(indices[i, 1:][valid])
            data.extend(distances[i, 1:][valid])

    adjacency = csr_matrix((data, (rows, cols)), shape=(len(points), len(points)))

    # 查找最远的 endpoint pair
    longest_path_len = -1
    best_start, best_end = None, None

    tree_points = cKDTree(points)
    endpoint_indices = [tree_points.query(ep)[1] for ep in endpoints]

    for i in range(len(endpoint_indices)):
        for j in range(i + 1, len(endpoint_indices)):
            start_idx, end_idx = endpoint_indices[i], endpoint_indices[j]
            dist_matrix, _ = dijkstra(adjacency, indices=start_idx, return_predecessors=True)
            path_len = dist_matrix[end_idx]
            if np.isfinite(path_len) and path_len > longest_path_len:
                longest_path_len = path_len
                best_start = start_idx
                best_end = end_idx

    if best_start is None or best_end is None:
        # fallback: 直接找最远点对
        D = distance_matrix(points, points)
        idx = np.unravel_index(np.argmax(D), D.shape)
        start = tuple(points[idx[0]])
        end = tuple(points[idx[1]])
    else:
        start = tuple(points[best_start])
        end = tuple(points[best_end])

    return [start, end]

# for main aorta
def find_endpoints_aorta(centerline_img):
    z, y, x = np.nonzero(centerline_img)
    points = np.column_stack((z, y, x))
    
    endpoints = []
    for point in points:
        z, y, x = point
        neighborhood = centerline_img[
            max(0, z - 1):min(centerline_img.shape[0], z),
            max(0, y - 20):min(centerline_img.shape[1], y + 20),
            max(0, x - 20):min(centerline_img.shape[2], x + 20)
        ]
        
        if np.sum(neighborhood) == 0:
            endpoints.append(point)
    
    if len(endpoints) < 2:
        raise ValueError("Not enough endpoints found for the centerline segmentation.")
    elif len(endpoints) > 2:
        unique_z_coords = {point[0] for point in endpoints}
        selected_endpoints = []
        for z in unique_z_coords:
            same_z_points = [point for point in endpoints if point[0] == z]
            selected_endpoints.append(random.choice(same_z_points))
            if len(selected_endpoints) == 2:
                break
        endpoints = selected_endpoints
    return endpoints

def find_peak_point(points):
    peak_index = np.argmax(points[:, 0])
    peak_point = points[peak_index]
    return peak_point
# --- 图与最短路径 ---
def create_graph(binary_image, segmentation_array):
    points = np.argwhere(binary_image > 0)
    tree = cKDTree(points)
    k = 100
    distances, indices = tree.query(points, k=k)
    
    max_distance = np.percentile(distances[:, 1:], 75) * 2.5
    mask = distances <= max_distance
    
    rows = []
    cols = []
    data = []
    
    for i in range(len(points)):
        valid_connections = mask[i, 1:]
        if np.any(valid_connections):
            rows.extend([i] * np.sum(valid_connections))
            cols.extend(indices[i, 1:][valid_connections])
            data.extend(distances[i, 1:][valid_connections])
    
    adjacency = csr_matrix((data, (rows, cols)), shape=(len(points), len(points)))
    
    #print(f"Total points: {len(points)}, Total connections: {len(data)}")
    #print(f"Average connections per point: {len(data)/len(points):.2f}")
    
    return points, adjacency
def create_graph_v2(binary_image, segmentation_array):
    points = np.argwhere(binary_image > 0)
    tree = cKDTree(points)
    num_voxels = np.count_nonzero(binary_image)
    if num_voxels > 150:
        k = 50
    elif num_voxels > 88:
        k=30
    else:
        k = 20
    print(f"Selected k: {k}")
    distances, indices = tree.query(points, k=k)
    print(f"  Suggestion: Try threshold = {np.percentile(distances[:, 1:], 50) * 1.5:.3f}")

    max_distance = np.percentile(distances[:, 1:], 75) * 1.5
    mask = distances <= max_distance
    
    rows = []
    cols = []
    data = []
    
    for i in range(len(points)):
        valid_connections = mask[i, 1:]
        if np.any(valid_connections):
            rows.extend([i] * np.sum(valid_connections))
            cols.extend(indices[i, 1:][valid_connections])
            data.extend(distances[i, 1:][valid_connections])
    
    adjacency = csr_matrix((data, (rows, cols)), shape=(len(points), len(points)))
    
    print(f"Total points: {len(points)}, Total connections: {len(data)}")
    print(f"Average connections per point: {len(data)/len(points):.2f}")
    
    return points, adjacency
def get_centerline_dijkstra(binary_image, start_point, end_point, segmentation_array):
    points, adjacency = create_graph(binary_image, segmentation_array)
    tree = cKDTree(points)
    start_idx = tree.query(start_point)[1]
    end_idx = tree.query(end_point)[1]
    
    dist_matrix, predecessors = dijkstra(adjacency, indices=start_idx, return_predecessors=True)
    
    path = []
    current = end_idx
    while current != start_idx and current != -9999:
        path.append(points[current])
        current = predecessors[current]
    path.append(points[start_idx])
    path = np.array(path[::-1])
    
    #print(f"Path length: {len(path)}")
    #print(f"Path z range: {path[:,0].min()} to {path[:,0].max()}")
    
    return path
def get_centerline_dijkstra_v2(binary_image, start_point, end_point, segmentation_array):
    points, adjacency = create_graph_v2(binary_image, segmentation_array)
    tree = cKDTree(points)
    start_idx = tree.query(start_point)[1]
    end_idx = tree.query(end_point)[1]
    
    dist_matrix, predecessors = dijkstra(adjacency, indices=start_idx, return_predecessors=True)
    
    path = []
    current = end_idx
    while current != start_idx and current != -9999:
        path.append(points[current])
        current = predecessors[current]
    path.append(points[start_idx])
    path = np.array(path[::-1])
    
    #print(f"Path length: {len(path)}")
    #print(f"Path z range: {path[:,0].min()} to {path[:,0].max()}")
    
    return path

# --- 曲线拟合与切向量 ---

#for main aorta
def fit_polynomial_robust_aorta(points, num_points=200, is_connection_point=None):
    """改进的多项式拟合函数，增强连接处的平滑性"""
    if len(points) < 4:
        return np.repeat(points.mean(axis=0).reshape(1, -1), num_points, axis=0), np.zeros((num_points, 3))
    
    # 计算弧长参数化
    diffs = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    t = np.concatenate(([0], np.cumsum(diffs)))
    
    # 检查并修复重复的t值
    eps = 1e-10
    for i in range(1, len(t)):
        if t[i] <= t[i-1]:
            t[i] = t[i-1] + eps
    
    t = t / t[-1]
    t_smooth = np.linspace(0, 1, num_points)
    
    curve_points = np.zeros((num_points, 3))
    tangents = np.zeros((num_points, 3))
    
    for i in range(3):
        try:
            # 根据是否是连接点选择不同的边界条件
            if is_connection_point == 1.0:  # 末端连接点
                cs = CubicSpline(t, points[:, i], bc_type=((2, 0), (2, 0)))
            elif is_connection_point == 0.0:  # 起始连接点
                cs = CubicSpline(t, points[:, i], bc_type=((2, 0), (2, 0)))
            else:
                cs = CubicSpline(t, points[:, i], bc_type='natural')
            
            curve_points[:, i] = cs(t_smooth)
            tangents[:, i] = cs.derivative()(t_smooth)
        except ValueError as e:
            print(f"Warning: Error fitting dimension {i}, using linear interpolation")
            from scipy.interpolate import interp1d
            f = interp1d(t, points[:, i], kind='linear')
            curve_points[:, i] = f(t_smooth)
            tangents[:, i] = np.gradient(curve_points[:, i], t_smooth[1] - t_smooth[0])
    
    # 归一化切向量
    tangents_norm = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents_norm[tangents_norm == 0] = 1
    tangents = tangents / tangents_norm
    
    return curve_points, tangents

#for left and right arota branch
def fit_polynomial_robust(points, num_points=100, is_connection_point=None):
    """改进的多项式拟合函数，增强连接处的平滑性"""
    if len(points) < 4:
        return np.repeat(points.mean(axis=0).reshape(1, -1), num_points, axis=0), np.zeros((num_points, 3))
    
    # 计算弧长参数化
    diffs = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    t = np.concatenate(([0], np.cumsum(diffs)))
    
    # 检查并修复重复的t值
    eps = 1e-10
    for i in range(1, len(t)):
        if t[i] <= t[i-1]:
            t[i] = t[i-1] + eps
    
    t = t / t[-1]
    t_smooth = np.linspace(0, 1, num_points)
    
    curve_points = np.zeros((num_points, 3))
    tangents = np.zeros((num_points, 3))
    
    for i in range(3):
        try:
            # 根据是否是连接点选择不同的边界条件
            if is_connection_point == 1.0:  # 末端连接点
                cs = CubicSpline(t, points[:, i], bc_type=((2, 0), (2, 0)))
            elif is_connection_point == 0.0:  # 起始连接点
                cs = CubicSpline(t, points[:, i], bc_type=((2, 0), (2, 0)))
            else:
                cs = CubicSpline(t, points[:, i], bc_type='natural')
            
            curve_points[:, i] = cs(t_smooth)
            tangents[:, i] = cs.derivative()(t_smooth)
        except ValueError as e:
            print(f"Warning: Error fitting dimension {i}, using linear interpolation")
            from scipy.interpolate import interp1d
            f = interp1d(t, points[:, i], kind='linear')
            curve_points[:, i] = f(t_smooth)
            tangents[:, i] = np.gradient(curve_points[:, i], t_smooth[1] - t_smooth[0])
    
    # 归一化切向量
    tangents_norm = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents_norm[tangents_norm == 0] = 1
    tangents = tangents / tangents_norm
    
    return curve_points, tangents

def smooth_tangents(tangents, window_size=5):
    """使用滑动窗口平滑切向量"""
    smoothed = np.zeros_like(tangents)
    pad_size = window_size // 2
    
    # 填充边界
    padded = np.pad(tangents, ((pad_size, pad_size), (0, 0)), mode='edge')
    
    for i in range(len(tangents)):
        window = padded[i:i + window_size]
        # 计算局部平均，保持方向一致性
        mean_vec = np.mean(window, axis=0)
        # 确保与前一个切向量方向基本一致
        if i > 0 and np.dot(mean_vec, smoothed[i-1]) < 0:
            mean_vec = -mean_vec
        smoothed[i] = mean_vec / np.linalg.norm(mean_vec)
    
    return smoothed

def check_tangent_continuity(tangents, threshold=30):
    """检查相邻切向量的角度变化"""
    angles = np.degrees(np.arccos(np.clip(np.sum(
        tangents[1:] * tangents[:-1], axis=1), -1.0, 1.0)))
    
    problematic = angles > threshold
    if np.any(problematic):
        print(f"Warning: Large angle changes detected at indices: "
              f"{np.where(problematic)[0]}")
        print(f"Maximum angle change: {np.max(angles):.2f} degrees")
    
    return angles
# --- curve mask ---
def create_binary_curve_image(centerline_img, curve_points):
    curve_img = np.zeros_like(centerline_img, dtype=np.uint8)
    
    for point in curve_points:
        z, y, x = int(round(point[0])), int(round(point[1])), int(round(point[2]))
        if (0 <= z < curve_img.shape[0] and 
            0 <= y < curve_img.shape[1] and 
            0 <= x < curve_img.shape[2]):
            curve_img[z, y, x] = 1
    
    return curve_img