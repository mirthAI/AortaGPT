## clear unnecesary libraries
# zone 11 only keep the cutting plane that alll voxel from zone 11. (22,23)
# we can define the valid percentage for each zone
# used for super resolution compare with low and high resolution 
import numpy as np
import os
import random
import time
import traceback
import matplotlib.pyplot as plt
import gc
import torch
import torch.nn.functional as F
import SimpleITK as sitk
from skimage.morphology import skeletonize, ball, closing, opening
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree, distance_matrix
from scipy.interpolate import CubicSpline, interp1d
from scipy.ndimage import label, binary_closing, binary_opening, binary_fill_holes,binary_dilation
from skimage.measure import regionprops, find_contours, EllipseModel
from scipy.spatial.transform import Rotation as R
from matplotlib.path import Path
import itertools
import sys
from concurrent.futures import ThreadPoolExecutor

from cupyx.scipy.ndimage import convolve as convolve
import cupy
import cvxpy as cp
from cupyx.scipy.ndimage import label as cp_label
from collections import deque
from scipy.sparse import coo_matrix
from pathlib import Path as path_vtk  
import vtk
from vtk.util import numpy_support


class AortaAnalysis:    
    def __init__(self):
        self.pad_percentage = 0.3 ## Padding percentage for the segmentation map
        self.radius = 2 ## Radius for binary preprocessing maybe 3
        self.filter_size = 3 ## size of the median filter to fix broken edges
        self.binary_map = None
        self.blend_window = 5
        self.constant_classes = [[1, 3, 5, 7, 8, 9, 10, 12, 14, 17], [18,22], [19,23]] ## Classes to keep in the segmentation map
        self.constant_classes_list = [1, 3, 5, 7, 8, 9, 10, 12, 14, 17, 18, 19, 22, 23] ## Classes to keep in the segmentation map
        self.zone_num_of_sample_points = [10, 5, 5, 5, 10, 15, 5, 5, 5, 10, 5, 5, 15, 15]
        self.crop_size = 81
        # self.percentage_threshold = [1.0,0.5,0.5,0.5,0.5,1.0,0.5,0.5,0.5,1.0,1.0,1.0,1.0,1.0]
        self.percentage_threshold = {
        1: 1.0, 3: 0.5, 5: 0.5, 7: 0.5, 8: 0.5,
        9: 1.0, 10: 0.5, 12: 0.5, 14: 0.5, 17: 1.0,
        18: 1.0, 19: 1.0, 22: 1.0, 23: 1.0
    }
        self.device = 'cuda'
        self.angles = [-20, -10, 0,10, 20]
    
    def calculate_all_diameters(self, mask):
        total_start = time.time()
        # step 1:
        mask = torch.tensor(mask, dtype=torch.float32, device=self.device)

        step1_start = time.time()
        multiclass_map, binary_processed_segment = self.segmentation_to_binary(mask, self.constant_classes)
        multiclass_map = self._to_cpu(multiclass_map)
        binary_processed_segment = [self._to_cpu(i) for i in binary_processed_segment]
        step1_time = time.time() - step1_start
        print(f"⏱️  step1 - segment_preprocess: {step1_time:.2f}seconds")
 
        self.multiclass_map_gpu = torch.from_numpy(multiclass_map).to(self.device, dtype=torch.float32)

        #step 2: skeleton
        step2_start = time.time()
        original_skeleton_dilated = list(ThreadPoolExecutor().map(self.generate_skeleton, binary_processed_segment))
        
        step2_time = time.time() - step2_start
        print(f"⏱️  step2 - Skeleton generation: {step2_time:.2f}seconds")

        #step 3： centerline extraction
        step3_start = time.time()
        curve_points, tangents = self.get_all_centerline_points_and_tangents(original_skeleton_dilated)
        step3_time = time.time() - step3_start
        sampled_points, sampled_tangents = self.sample_centerline_points_for_diameter_measurement(curve_points, tangents, multiclass_map)
        print(f"⏱️  step3 - centerline extraction: {step3_time:.2f}seconds")

        #step 5: diameter measurement
        step4_start = time.time()
        diameters, centers, tangents, mc_map, zone_slice_data = self.calculate_diameters_by_angle_search(
    multiclass_map, sampled_points, sampled_tangents, self.device
)
        step4_time = time.time() - step4_start
        print(f"⏱️  step4 - diameter_measurement: {step4_time:.2f}seconds")
        return diameters, centers, tangents, mc_map, zone_slice_data, curve_points, sampled_points,original_skeleton_dilated

    def keep_largest_connected_component(self, binary_array):
        # 1) push to GPU via true CuPy
        arr_gpu = cupy.asarray(binary_array, dtype=cupy.uint8)
        # 2) label on GPU
        labeled_gpu, num_features = cp_label(arr_gpu)
        if num_features < 1:
            return binary_array * 0

        # 3) count voxels per label (bincount on GPU)
        counts = cupy.bincount(labeled_gpu.ravel())
        counts[0] = 0                           # ignore background
        largest = int(cupy.argmax(counts))      # get largest component label

        # 4) extract it and bring back to CPU
        cleaned_gpu = (labeled_gpu == largest).astype(cupy.uint8)
        return cupy.asnumpy(cleaned_gpu)

    def centerline_for_main_aorta(self, original_skeleton_dilated):      
        endpoints = self.find_endpoints(original_skeleton_dilated)
        B, A = sorted(endpoints, key=lambda p: p[0])
        original_path = self.get_centerline_dijkstra(original_skeleton_dilated, A, B)

        split_idx = np.argmax(original_path[:, 0])
        ascending_part = original_path[:split_idx+1]
        descending_part = original_path[split_idx:]

        overlap_point = original_path[split_idx]
        ascending_part = np.vstack([ascending_part, overlap_point])
        descending_part = np.vstack([overlap_point, descending_part])
        ascending_curve, ascending_tangents = self.fit_polynomial(ascending_part, num_points=400, is_connection_point=1.0)
        descending_curve, descending_tangents = self.fit_polynomial(descending_part, num_points=800, is_connection_point=0.0)

        for i in range(self.blend_window):
            alpha = i / self.blend_window
            blended_tangent = (1 - alpha) * ascending_tangents[-self.blend_window + i] + alpha * descending_tangents[i]
            blended_tangent /= np.linalg.norm(blended_tangent)
            ascending_tangents[-self.blend_window + i] = blended_tangent
            descending_tangents[i] = blended_tangent


        ascending_tangents = self.smooth_tangents(ascending_tangents)
        descending_tangents = self.smooth_tangents(descending_tangents)
      
        curve_points_origin = np.vstack((ascending_curve, descending_curve))
        tangents_origin = np.vstack((ascending_tangents, descending_tangents))

        curve = curve_points_origin[:,[2,1,0]]#zyx——》xyz
        tangents = tangents_origin[:,[2,1,0]]#zyx——》xyz

        return curve, tangents

    def get_all_centerline_points_and_tangents(self, original_skeleton_dilated):
        curve_points_origin_0, tangents_origin_0 = self.centerline_for_main_aorta(original_skeleton_dilated[0])
        curve_points_origin_1, tangents_origin_1 = self.centerline(original_skeleton_dilated[1])
        curve_points_origin_2, tangents_origin_2 = self.centerline(original_skeleton_dilated[2])

        curve_points_origin = np.concatenate([curve_points_origin_0.astype(np.int16),
                                              curve_points_origin_1.astype(np.int16),
                                              curve_points_origin_2.astype(np.int16)], axis=0)

        tangents_origin = np.concatenate([tangents_origin_0, tangents_origin_1, tangents_origin_2], axis=0)

        return curve_points_origin, tangents_origin



    def centerline(self, original_skeleton_dilated, num_points=400):      
        """
        Original Dijkstra-based centerline, instrumented to show per-step timings.
        """
        t0 = time.time()
        # 1) find endpoints
        endpoints = self.find_endpoints(original_skeleton_dilated)
        t1 = time.time()

        # 2) sort to get start/end
        B, A = sorted(endpoints, key=lambda p: p[0])
        t2 = time.time()

        # 3) Dijkstra shortest path
        original_path = self.get_centerline_dijkstra(original_skeleton_dilated, A, B)
        t3 = time.time()

        # 4) fit polynomial to get smooth curve & tangents
        curve, tangents = self.fit_polynomial(original_path, num_points=num_points)
        t4 = time.time()

        # 5) reorder axes from (z,y,x)→(x,y,z)
        curve    = curve   [:, [2,1,0]]
        tangents = tangents[:, [2,1,0]]
        t5 = time.time()

        return curve, tangents

    
    def sample_centerline_points_for_diameter_measurement(self, curve_points, tangents, multiclass_map):

        ### compute index range, number of points in each zone 

        zone_ranges, point_zone = self.analyze_centerline_zones(curve_points, multiclass_map)

        sampled_points = []
        sampled_tangents = []
        
        for zone_id, num_samples in zip(self.constant_classes_list, self.zone_num_of_sample_points):

            indices = np.where(point_zone == zone_id)[0]
        
            if len(indices) == 0:
                continue  # skip zones with no points
        
            if len(indices) <= num_samples:
                chosen = indices  # keep all available
            else:
                chosen = indices[np.linspace(0, len(indices) - 1, num_samples, dtype=int)]
        
            sampled_points.append(curve_points[chosen])
            sampled_tangents.append(tangents[chosen])
        
        sampled_points = np.concatenate(sampled_points, axis=0)
        sampled_tangents = np.concatenate(sampled_tangents, axis=0)

        return sampled_points, sampled_tangents
    
    
    def segmentation_to_binary(self, mask, classes):
        mask_interim = self.preprocess_keep_classes(mask, self.constant_classes_list)
        cropped_array = self.crop_image_array(mask_interim)
        multiclass_map_origin = self.pad_with_percentage(cropped_array)           
        multiclass_array_origin = [self.preprocess_keep_classes(multiclass_map_origin, c) for c in classes]
        
        self.binary_map = [i.clone() for i in multiclass_array_origin]
        binary_processed_segment = list(ThreadPoolExecutor().map(self.binary_preprocess, self.binary_map))
     
        return self._to_cpu(multiclass_map_origin), binary_processed_segment ## list of arrays


    def binary_preprocess(self, binary_map):
        binary_map[binary_map > 0] = 1
        processed_segment = self.preprocess_binary_segment(binary_map)
        return processed_segment
    
#     def preprocess_binary_segment(self, binary_segment):
#         selem = self.make_ball(self.radius) 
#         opened = self.opening3d(binary_segment.unsqueeze(0).unsqueeze(0), selem) 
#         processed_segment = self.closing3d(opened, selem)

#         return processed_segment.squeeze(0).squeeze(0)
    def preprocess_binary_segment(self,binary_segment):
        x = binary_segment.unsqueeze(0).unsqueeze(0)
        selem_erode  = self.make_ball(1).to(x.device)  # erosion 小核
        selem_dilate = self.make_ball(2).to(x.device)  # dilation 大核

        # opening: 小核腐蚀 → 大核膨胀
        x = self.morphological_erosion3d(x, selem_erode)
        x = self.morphological_dilation3d(x, selem_dilate)

        # closing: 大核膨胀 → 小核腐蚀
        x = self.morphological_dilation3d(x, selem_dilate)
        x = self.morphological_erosion3d(x, selem_erode)

        return x.squeeze(0).squeeze(0)
    def make_ball(self, radius: int):
        """Creates a 3D ball (spherical) structuring element as a binary tensor."""
        size = 2 * radius + 1
        z, y, x = torch.meshgrid(
            torch.arange(size, device=self.device),
            torch.arange(size, device=self.device),
            torch.arange(size, device=self.device),
            indexing='ij'  # for PyTorch ≥1.10
        )
        center = radius
        dist = ((z - center)**2 + (y - center)**2 + (x - center)**2).sqrt()
        ball = (dist <= radius).float()
        return ball
    def morphological_erosion3d(self, volume, kernel):
        k = kernel.sum()
        padding = kernel.shape[0] // 2  # assumes odd-sized kernel
        volume = volume.float()  # ensure float for conv
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        return (F.conv3d(volume, kernel, padding=padding) == k).float()
        
    def morphological_dilation3d(self, volume, kernel):
        padding = kernel.shape[0] // 2
        volume = volume.float()
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        return (F.conv3d(volume, kernel, padding=padding) > 0).float()
    
    def opening3d(self, volume, kernel):
        eroded = self.morphological_erosion3d(volume, kernel)
        return self.morphological_dilation3d(eroded, kernel)

    def closing3d(self, volume, kernel):
        dilated = self.morphological_dilation3d(volume, kernel)
        return self.morphological_erosion3d(dilated, kernel)        

    def preprocess_keep_classes(self, arr, keep_classes):
        arr_new = arr.clone()
        mask = ~torch.isin(arr, torch.tensor(keep_classes, device=self.device))
        arr_new[mask] = 0
        return arr_new
    def crop_image_array(self, array):
        nonzero_coords = torch.nonzero(array)
        min_z, min_y, min_x = nonzero_coords.min(axis=0).values
        max_z, max_y, max_x = nonzero_coords.max(axis=0).values
        return array[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]

    def pad_with_percentage(self, volume):
        depth, height, width = volume.shape
        pad_depth = int(depth * self.pad_percentage)
        pad_height = int(height * self.pad_percentage)
        pad_width = int(width * self.pad_percentage)
        new_shape = (depth + 2 * pad_depth, height + 2 * pad_height, width + 2 * pad_width)
        new_array = torch.zeros(new_shape, dtype=volume.dtype, device=self.device)
        new_array[pad_depth:pad_depth + depth, pad_height:pad_height + height, pad_width:pad_width + width] = volume
        return new_array
    
    def generate_skeleton(self, binary_map, dilation_structure=(3,3,3), skeleton_dilate_iters=2):
        skeleton = skeletonize(binary_map).astype(np.uint8)

        skeleton = binary_dilation(skeleton.astype(bool), iterations=skeleton_dilate_iters).astype(np.uint8)
        skeleton = self.keep_largest_connected_component(skeleton)
        return skeleton
 
    def find_endpoints(self, centerline_img, k=26, max_distance=2.0):
        #points = np.argwhere(centerline_img > 0)

        skel_t = torch.from_numpy((centerline_img > 0).astype(np.uint8)).to(self.device)
        pts_t  = torch.nonzero(skel_t, as_tuple=False)   # (M,3) on GPU

        # 2) bring back the small (M×3) array to CPU
        points = pts_t.cpu().numpy()                     # dtype=int64
  

        mask_gpu = cupy.asarray(centerline_img > 0, dtype=cupy.uint8)
        kernel_gpu = cupy.ones((3, 3, 3), dtype=cupy.uint8)
        neighbor_count_gpu = convolve(mask_gpu, kernel_gpu, mode='constant', cval=0)
        neighbor_count_gpu = cupy.where(mask_gpu > 0, neighbor_count_gpu, cupy.uint8(255))
        neighbor_count = cupy.asnumpy(neighbor_count_gpu)


        flat = neighbor_count.ravel()
        s = min(50, len(points))
        flat_indices = np.argpartition(flat, s)[:s]
        sorted_flat_indices = flat_indices[np.argsort(flat[flat_indices])]
        # Convert flat indices to (z, y, x) coordinates and stack as (N, 3) array
        endpoints = np.column_stack(np.unravel_index(sorted_flat_indices, neighbor_count.shape))

        # 构建图
        tree = cKDTree(points)
        distances, indices = tree.query(points, k=k)
        mask = distances <= max_distance


        # Vectorized assembly of the sparse adjacency
        valid = mask[:, 1:]                 # drop self-distance column, shape (N, k-1)
        i_idx, j_off = np.nonzero(valid)    # all (point_idx, neighbor_offset) pairs

        rows = i_idx                         # source point indices
        cols = indices[i_idx, j_off + 1]     # neighbor point indices (shift by +1)
        data = distances[i_idx, j_off + 1]   # corresponding edge weights

        adjacency = csr_matrix(
            (data, (rows, cols)),
            shape=(len(points), len(points)))


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

    def get_centerline_dijkstra(self, binary_image, start_point, end_point):
        points, adjacency = self.create_graph(binary_image)
        
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
        return path

    def create_graph(self, binary_image):
        """
        Builds a graph for skeleton voxels, with per-step timing:
          1) points extraction
          2) KD-tree build
          3) k-NN query
          4) threshold & mask computation
          5) edge list assembly
          6) sparse adjacency creation
        """
        # 1) extract points
        t0 = time.time()
        # points = np.argwhere(binary_image > 0)

        skel_t = torch.from_numpy((binary_image > 0).astype(np.uint8)).to(self.device)
        pts_t  = torch.nonzero(skel_t, as_tuple=False)   # (M,3) on GPU

        # 2) bring back the small (M×3) array to CPU
        points = pts_t.cpu().numpy()                     # dtype=int64


        t1 = time.time()
        
        # 2) build KD-tree
        tree = cKDTree(points)
        t2 = time.time()

        # 3) k-NN query (including self at index 0)
        k = 100
        distances, indices = tree.query(points, k=k)
        t3 = time.time()

        # 4) compute max_distance threshold & mask
        max_distance = np.percentile(distances[:, 1:], 75) * 2.5
        mask = distances <= max_distance
        t4 = time.time()


        valid = mask[:, 1:]           # (N, k-1) bool
        # get all (i, j_offset) where valid is True
        i_idx, j_off = np.nonzero(valid)  # both 1D arrays of length M

        # build rows, cols, data *as numpy arrays*
        rows = i_idx
        cols = indices[i_idx, j_off + 1]
        data = distances[i_idx, j_off + 1]

        t5 = time.time()

        N = len(points) 
        adjacency = coo_matrix((data, (rows, cols)), shape=(N, N))

        t6 = time.time()

        return points, adjacency

    def fit_polynomial(self, points, num_points=400, is_connection_point=None):
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

    def smooth_tangents(self, tangents, window_size=5):
        """使用滑动窗口平滑切向量"""
        smoothed = np.zeros_like(tangents)
        pad_size = window_size // 2
        
        # 填充边界
        padded = np.pad(tangents, ((pad_size, pad_size), (0, 0)), mode='edge')
        
        for i in range(len(tangents)):
            window = padded[i:i + window_size]
            
            mean_vec = np.mean(window, axis=0)
            
            if i > 0 and np.dot(mean_vec, smoothed[i-1]) < 0:
                mean_vec = -mean_vec
            smoothed[i] = mean_vec / np.linalg.norm(mean_vec)
        
        return smoothed        
    
    def create_binary_curve_image(self, centerline_img, curve_points):
        curve_img = np.zeros_like(centerline_img, dtype=np.uint8)
        
        for point in curve_points:
            z, y, x = int(round(point[0])), int(round(point[1])), int(round(point[2]))
            if (0 <= z < curve_img.shape[0] and 
                0 <= y < curve_img.shape[1] and 
                0 <= x < curve_img.shape[2]):
                curve_img[z, y, x] = 1
        
        return curve_img    

    def analyze_centerline_zones(self, centerline_points, segmentation_map):
        """
        分析中心线点所在的区域，将点按zone分类
        
        参数:
        - centerline_points: 中心线点坐标数组 shape=(N, 3)
        - segmentation_map: 分区图像 shape=(D, H, W)
        
        返回:
        - zone_ranges: dict, 每个zone对应的点索引范围
        - point_zones: list, 每个点对应的zone值
        """
        point_zones = []  # 存储每个点对应的zone
        zone_points = {}  # 存储每个zone包含的点的索引
        
        # 遍历所有中心线点
        for i, point in enumerate(centerline_points):
            x, y, z = np.round(point).astype(int)
            
            # 确保点在图像范围内
            if (0 <= z < segmentation_map.shape[0] and 
                0 <= y < segmentation_map.shape[1] and 
                0 <= x < segmentation_map.shape[2]):
                
                zone = segmentation_map[z, y, x]
                point_zones.append(zone)
                
                # 将点索引添加到对应zone的列表中
                if zone not in zone_points:
                    zone_points[zone] = []
                zone_points[zone].append(i)
        
        # 计算每个zone的点索引范围
        zone_ranges = {}
        for zone in sorted(zone_points.keys()):
            indices = sorted(zone_points[zone])
            if indices:  # 确保该zone有点
                zone_ranges[zone] = {
                    'start': indices[0],
                    'end': indices[-1],
                    'total_points': len(indices),
                    'indices': indices  # 保存所有索引点
                }
        
        return zone_ranges, np.array(point_zones)



    def rotate_tangent_batch(self, sampled_tangents, device=None):
        if device is None:
            device = self.device
        # Rotation angles in degrees
        angles = self.angles
        
        # Generate all combinations of (x, y, z) angles — 125 total
        all_angle_combinations = list(itertools.product(angles, repeat=3))
    
        # Convert input to PyTorch tensor and move to device
        if not isinstance(sampled_tangents, torch.Tensor):
            sampled_tangents = torch.tensor(sampled_tangents, dtype=torch.float32, device=self.device)
        else:
            sampled_tangents = sampled_tangents.to(device)
    
        B = sampled_tangents.shape[0]
        rotated_results = []
    
        for angle_x_deg, angle_y_deg, angle_z_deg in all_angle_combinations:
            # Convert to radians
            ax = torch.deg2rad(torch.tensor(angle_x_deg, device=device, dtype=torch.float32))
            ay = torch.deg2rad(torch.tensor(angle_y_deg, device=device, dtype=torch.float32))
            az = torch.deg2rad(torch.tensor(angle_z_deg, device=device, dtype=torch.float32))
    
            # Rotation matrices
            Rx = torch.tensor([
                [1, 0, 0],
                [0, torch.cos(ax), -torch.sin(ax)],
                [0, torch.sin(ax),  torch.cos(ax)]
            ], device=device)
    
            Ry = torch.tensor([
                [ torch.cos(ay), 0, torch.sin(ay)],
                [0, 1, 0],
                [-torch.sin(ay), 0, torch.cos(ay)]
            ], device=device)
    
            Rz = torch.tensor([
                [torch.cos(az), -torch.sin(az), 0],
                [torch.sin(az),  torch.cos(az), 0],
                [0, 0, 1]
            ], device=device)
    
            # Combined rotation: R = Rz * Ry * Rx
            R = Rz @ Ry @ Rx  # [3, 3]
    
            # Rotate all vectors in batch: [B, 3] @ [3, 3]^T => [B, 3]
            rotated = sampled_tangents @ R.T
            rotated_results.append(rotated)
    
        # Stack into [125, B, 3]
        return torch.stack(rotated_results, dim=0)

    def calculate_diameters_by_angle_search(self, multiclass_map, sampled_points, sampled_tangents, device):
        sampled_tangents = torch.as_tensor(sampled_tangents, dtype=torch.float32, device=device) 
        rotated = self.rotate_tangent_batch(sampled_tangents)  # (A, P, 3)
        A, P, _ = rotated.shape
        all_diams = np.zeros((A, P), dtype=float)

        # 保存所有的 Rodrigues 初始旋转矩阵（每个点 1 个）
        initial_rots = self.batch_rodrigues(sampled_tangents, device=self.device).detach().cpu().numpy()  # (P, 3, 3)

        # 所有角度组合（可选项）
        angle_combinations = list(itertools.product(self.angles, repeat=3))

        # --- 缓存所有切片图像 ---
        all_original_slices = []

        # --- 角度搜索 ---
        for ai in range(A):
            original_slices = self.batch_compute_cuts(
                multiclass_map, sampled_points, rotated[ai], device, crop_size=self.crop_size
            )
            all_diams[ai] = self.compute_diameters_from_2D_cuts(original_slices)
            if isinstance(original_slices, np.ndarray):
                all_original_slices.append(torch.from_numpy(original_slices))
            else:
                all_original_slices.append(original_slices.detach().cpu())

        all_original_slices = torch.stack(all_original_slices, dim=0)  # (A, P, H, W)

        # 选择最优角度索引
        masked = np.where(all_diams == 0, np.inf, all_diams)
        best_diams = np.min(masked, axis=0)  # (P,)
        best_angle_idx = np.argmin(masked, axis=0)  # (P,)
        best_rotated_tangent_all = rotated[best_angle_idx, np.arange(P)]  # (P, 3)

        # Zone assignment
        _, point_zones = self.analyze_centerline_zones(sampled_points, multiclass_map)
        zones = sorted(np.unique(point_zones))

        diameters, centers, tangents = [], [], []
        zone_slice_data = {}

        for zone in zones:
            idxs = np.where(point_zones == zone)[0]
            zone_diams = best_diams[idxs]
            valid = (zone_diams > 0) & np.isfinite(zone_diams)
            if not np.any(valid):
                diameters.append(0.0)
                centers.append(None)
                tangents.append(None)
                continue

            # 找出该 zone 最大直径的点
            valid_idxs = np.where(valid)[0]
            best_local = valid_idxs[np.argmax(zone_diams[valid])]
            global_pt = idxs[best_local]

            # 找出对应旋转矩阵
            angle_idx = best_angle_idx[global_pt]
            single_best_tangent = rotated[angle_idx, global_pt].unsqueeze(0)
            final_rot_matrix = self.batch_rodrigues(single_best_tangent, device=device)[0].detach().cpu().numpy()

            # 提取该点对应角度下的切片
            cutting_plane = all_original_slices[angle_idx, global_pt].numpy()
            #cy, cx = cutting_plane.shape[0] // 2, cutting_plane.shape[1] // 2
            
            # 保存结果
            diameters.append(float(zone_diams[best_local]))
            centers.append(tuple(sampled_points[global_pt]))
            tangents.append(tuple(sampled_tangents[global_pt]))

            zone_slice_data[zone] = {
                'centerline_index': int(global_pt),
                'initial_data': {
                    'rotation_matrix': initial_rots[global_pt],
                    'diameter': float(zone_diams[best_local]),
                    'centerline_index': int(global_pt),
                    'center': tuple(sampled_points[global_pt])
                },
                'final_data': {
                    'rotation_matrix': final_rot_matrix,
                    'diameter': float(zone_diams[best_local]),
                    'centerline_index': int(global_pt),
                    'center': tuple(sampled_points[global_pt]), 
                    'cutting_plane': cutting_plane  # ✅ 加入切片图像
                }
            }
        return diameters, centers, tangents, np.transpose(multiclass_map.astype(int), (2, 1, 0)), zone_slice_data


    def calculate_diameters_for_sampled_points(self, multiclass_map, sampled_points, sampled_tangents, device):
        original_slices = self.batch_compute_cuts(multiclass_map, sampled_points, sampled_tangents, device, crop_size=self.crop_size)
        diameters = self.compute_diameters_from_2D_cuts(original_slices)
        return diameters

    def batch_rodrigues(self, ref_vectors: torch.Tensor, device=None, eps=1e-8):
        """
        ref_vectors: (B,3) tensor of target tangent directions
        returns    : (B,3,3) tensor of rotation matrices that rotate each ref_vector
                    onto the Z axis [0,0,1], choosing the shortest path.
        """
        if device is None:
            device = ref_vectors.device
        B = ref_vectors.shape[0]
        
        # 1) normalize inputs
        t = ref_vectors / (ref_vectors.norm(dim=1, keepdim=True).clamp(min=eps))  # (B,3)
        
        # 2) choose per‐sample reference = ±Z so angle ≤ 90°
        z = torch.tensor([0.0,0.0,1.0], device=device).view(1,3).repeat(B,1)       # (B,3)
        cos = (t*z).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)                      # (B,1)
        # for those with cos<0 (angle>90°), flip z
        mask = (cos < 0).view(-1)
        z[mask] *= -1
        cos = (t*z).sum(dim=1).clamp(-1.0, 1.0)                                    # (B,)
        angle = torch.acos(cos).view(B,1,1)                                       # (B,1,1)
        
        # 3) rotation axis = cross(t, z), normalized
        axis = torch.cross(t, z, dim=1)                                          # (B,3)
        axis_norm = axis.norm(dim=1, keepdim=True).clamp(min=eps)                # (B,1)
        axis = axis / axis_norm                                                  # (B,3)
        
        # 4) build skew‐symmetric K matrices
        ax, ay, az = axis[:,0], axis[:,1], axis[:,2]                             # each (B,)
        zeros = torch.zeros(B, device=device)
        K = torch.stack([
            torch.stack([ zeros, -az,    ay], dim=1),
            torch.stack([ az,    zeros, -ax], dim=1),
            torch.stack([-ay,    ax,    zeros], dim=1),
        ], dim=1)                                                                 # (B,3,3)
        
        # 5) Rodrigues’ formula: R = I + sin(θ)K + (1–cos(θ)) K²
        I = torch.eye(3, device=device).unsqueeze(0).expand(B,3,3)                # (B,3,3)
        sin = torch.sin(angle)                                                   # (B,1,1)
        cos = cos.view(B,1,1)                                                    # (B,1,1)
        R = I + sin * K + (1 - cos) * (K @ K)                                    # (B,3,3)
        return R


   #gpu——version 
    def batch_compute_cuts(
        self,
        multiclass_map,
        sampled_points,
        sampled_tangents,
        device,
        crop_size=101  # 可选参数，默认 crop 121 个 slice
    ):

        rotation_matrices = self.batch_rodrigues(sampled_tangents, device=self.device)

        batch_size = rotation_matrices.shape[0]

        # ==============================
        # 先 crop segmentation_map 以减少计算
        cropped_segs = torch.zeros((batch_size, crop_size, crop_size, crop_size), device=self.device)
        local_center = torch.zeros((batch_size, 3), device=self.device)

        for i in range(batch_size):
            z = int(round(sampled_points[i][2]))
            y = int(round(sampled_points[i][1]))
            x = int(round(sampled_points[i][0]))
            
            half = crop_size // 2
            
            z_start = max(0, z - half)
            z_end = min(multiclass_map.shape[0], crop_size + z_start)

            y_start = max(0, y - half)
            y_end = min(multiclass_map.shape[1], crop_size + y_start)

            x_start = max(0, x - half)
            x_end = min(multiclass_map.shape[2], crop_size + x_start)
            
            # 用 torch.from_numpy() 转为 tensor 再拷贝：
            if hasattr(self, 'multiclass_map_gpu') and self.multiclass_map_gpu is not None:
                seg_crop = self.multiclass_map_gpu[z_start:z_end, y_start:y_end, x_start:x_end]
            else:
                seg_crop = torch.from_numpy(multiclass_map[z_start:z_end, y_start:y_end, x_start:x_end]).to(device)
            actual_z, actual_y, actual_x = seg_crop.shape
            cropped_segs[i, :actual_z, :actual_y, :actual_x] = seg_crop
            
            local_center[i][2] = sampled_points[i][2] - z_start
            local_center[i][1] = sampled_points[i][1] - y_start
            local_center[i][0] = sampled_points[i][0] - x_start

        local_center = local_center.to(dtype=torch.float32)

        # 更新形状信息
        _, depth, height, width = cropped_segs.shape
        seg_tensor = cropped_segs 
        rot_matrices_torch = rotation_matrices

        rot_matrices_inv = torch.inverse(rot_matrices_torch)
        center_pixels = local_center 
        
        # Create a standard coordinate grid
        z_indices = torch.arange(depth, device=self.device)
        y_indices = torch.arange(height, device=self.device)
        x_indices = torch.arange(width, device=self.device)
        grid_z, grid_y, grid_x = torch.meshgrid(
            z_indices, y_indices, x_indices, indexing="ij"
        )
        
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        grid = grid.unsqueeze(0)               
        grid = grid.repeat(batch_size, 1, 1, 1, 1)  
        grid = grid - center_pixels.view(batch_size, 1, 1, 1, 3) ## sample grids

        B, D, H, W, _ = grid.shape
        
        # ✅ Ensure rotation matrices match grid dtype
        rot_matrices_inv = rot_matrices_inv.to(dtype=grid.dtype)
        
        # ✅ Step 1: Flatten the grid for batched matmul
        grid_flat = grid.view(B, -1, 3)  # shape: (B, D*H*W, 3)
        
        # ✅ Step 2: Apply inverse rotation (batched)
        rotated_flat = torch.bmm(grid_flat, rot_matrices_inv.transpose(1, 2))  # shape: (B, D*H*W, 3)
        
        # ✅ Step 3: Reshape back to (B, D, H, W, 3)
        rotated_grid = rotated_flat.view(B, D, H, W, 3)
        
        # ✅ Step 4: Translate by center_pixel
        rotated_grid = rotated_grid + center_pixels.view(B, 1, 1, 1, 3)
        
        # ✅ Step 5: Normalize to [-1, 1] for grid_sample
        norm_factor = torch.tensor([W - 1, H - 1, D - 1], device=device, dtype=grid.dtype).view(1, 1, 1, 1, 3)
        normalized_grid = 2 * (rotated_grid / norm_factor) - 1  # shape: (B, D, H, W, 3)
        
        # Now `normalized_grid` is ready for F.grid_sample
        normalized_grid = normalized_grid.to(dtype=seg_tensor.dtype) 
        seg_tensor = seg_tensor.to(dtype=normalized_grid.dtype)
        seg_tensor = seg_tensor.unsqueeze(1)  # (B, 1, D, H, W)

        # Apply sampling
        rotated_batch = F.grid_sample(
            seg_tensor,
            normalized_grid,
            mode="nearest",
            align_corners=True,
            padding_mode="zeros"
        )
        
        # Remove channel dim
        rotated_batch = rotated_batch.squeeze(1)  # (B, D, H, W)
        original_slices = []
        for i in range(batch_size):
            slice_idx = int(round(center_pixels[i, 2].item()))
            slice_idx = max(0, min(slice_idx, depth - 1))
            original_slices.append(rotated_batch[i, slice_idx, :, :])
        original_slices = torch.stack(original_slices, dim=0)


        return original_slices.cpu().numpy()

 
    
#     def compute_diameters_from_2D_cuts(self, original_slices):
#         """
#         Estimate diameters via moment‐based ellipse, but only when
#         at least self.percentage_threshold of the CC’s pixels at
#         center belong to the center_class.
#         """
#         batch_size, H, W = original_slices.shape
#         diameters = np.zeros(batch_size, dtype=float)

#         for i in range(batch_size):
#             slice_img = original_slices[i]               # float labels
#             binary    = (slice_img > 0).astype(np.uint8) # any foreground
#             lbl, _    = label(binary)                   # label CCs

#             # 1) find CC at the center pixel
#             cy, cx = H // 2, W // 2
#             center_lbl = lbl[cy, cx]
#             if center_lbl == 0:
#                 continue

#             # 2) isolate that CC
#             comp = (lbl == center_lbl)

#             # 3) percentage‐of‐center‐class check
#             original_masked = slice_img.copy()
#             original_masked[~comp] = 0
#             center_id   = slice_img[cy, cx]
#             total_px    = comp.sum()
#             valid_px    = np.sum(original_masked == center_id)
#             pct         = valid_px / total_px if total_px > 0 else 0.0
#             if pct < self.percentage_threshold:
#                 # too many mis‐classified pixels → skip
#                 continue

#             # 4) compute ellipse on this cleaned CC
#             #    regionprops expects a labeled image, so we re‐label just comp
#             comp_lbl = comp.astype(np.uint8)
#             props    = regionprops(comp_lbl)
#             if not props:
#                 continue

#             # major_axis_length = 2× semi‐major axis (in pixels)
#             diameters[i] = props[0].major_axis_length

#         return diameters
    def compute_diameters_from_2D_cuts(self, original_slices):
        """
        Estimate diameters via moment‐based ellipse, but only when
        at least self.percentage_threshold of the CC’s pixels at
        center belong to the center_class.
        Each zone has a specific threshold.
        """
        batch_size, H, W = original_slices.shape
        diameters = np.zeros(batch_size, dtype=float)

        for i in range(batch_size):
            slice_img = original_slices[i]  # float labels
            binary = (slice_img > 0).astype(np.uint8)  # any foreground
            lbl, _ = label(binary)  # label CCs

            # 1) Find CC at the center pixel
            cy, cx = H // 2, W // 2
            center_lbl = lbl[cy, cx]
            if center_lbl == 0:
                continue

            # 2) Isolate that CC
            comp = (lbl == center_lbl)

            # 3) Calculate the center class (zone) and its threshold
            center_class = slice_img[cy, cx]
            threshold = self.percentage_threshold[center_class]  # No default, must exist

            # 4) Percentage-of-center-class check
            original_masked = slice_img.copy()
            original_masked[~comp] = 0
            total_px = comp.sum()
            valid_px = np.sum(original_masked == center_class)
            pct = valid_px / total_px if total_px > 0 else 0.0
            if pct < threshold:
                # too many misclassified pixels → skip
                continue

            # 5) Compute ellipse on this cleaned CC
            comp_lbl = comp.astype(np.uint8)
            props = regionprops(comp_lbl)
            if not props:
                continue

            # major_axis_length = 2× semi‐major axis (in pixels)
            diameters[i] = props[0].major_axis_length

        return diameters

    
    def _to_cpu(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        elif isinstance(tensor, np.ndarray):
            return tensor
        else:
            raise TypeError(f"Unsupported type: {type(tensor)}. Expected torch.Tensor or np.ndarray.")       
    def _to_gpu(self, array):
        if isinstance(array, np.ndarray):
            return torch.tensor(array, dtype=torch.float32, device=self.device)
        elif isinstance(array, torch.Tensor):
            return array.to(self.device)
        else:
            raise TypeError(f"Unsupported type: {type(array)}. Expected np.ndarray or torch.Tensor.")

    def export_vtk_with_cutting_planes(
        self,
        multiclass_map_origin,
        curve_point_segments,  # List of 3 arrays: [segment0, segment1, segment2]
        sampled_points, 
        zone_slice_data,
        output_dir="vtk_output"
    ):
        print("Starting export...")
        output_dir = path_vtk(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1️⃣ segmentation map
        print("Processing segmentation data...")
        seg_array = multiclass_map_origin  # already (x, y, z)

        vtk_data = vtk.vtkImageData()
        vtk_data.SetDimensions(seg_array.shape)
        vtk_data.SetSpacing((1, 1, 1))  # Replace with real spacing if available

        flat_array = seg_array.ravel(order="F").astype(np.uint8)
        vtk_array = numpy_support.numpy_to_vtk(flat_array, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_array.SetName("Labels")
        vtk_data.GetPointData().SetScalars(vtk_array)

        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(str(output_dir / "aorta.vti"))
        writer.SetInputData(vtk_data)
        writer.Write()

        # 2️⃣ centerline (multi-segment support)
        print("Processing centerline (multi-segment)...")
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        point_id = 0

        for segment in curve_point_segments:
            n = len(segment)
            if n < 2:
                continue  # Skip invalid segments
            polyLine = vtk.vtkPolyLine()
            polyLine.GetPointIds().SetNumberOfIds(n)
            for i in range(n):
                p_xyz =segment[i]#xyz
                points.InsertNextPoint(*p_xyz)
                polyLine.GetPointIds().SetId(i, point_id)
                point_id += 1
            lines.InsertNextCell(polyLine)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(output_dir / "centerline.vtp"))
        writer.SetInputData(polydata)
        writer.Write()
        # 3️⃣ cutting planes
        print("Processing cutting planes...")
        for zone, data in zone_slice_data.items():
            zone_dir = output_dir / f"zone_{zone}"
            zone_dir.mkdir(exist_ok=True)

            grid_points = 40

            for key, dat in zip(['initial', 'final'], [data['initial_data'], data['final_data']]):
                c = np.array(dat['center']) 
                center_point = np.array([c[0], c[1], c[2]])  #  xyz

                diameter = dat['diameter']
                plane_size = diameter * 3
                x = np.linspace(-plane_size/2, plane_size/2, grid_points)
                y = np.linspace(-plane_size/2, plane_size/2, grid_points)
                xv, yv = np.meshgrid(x, y)
                zv = np.zeros_like(xv)
                plane_pts = np.column_stack((xv.ravel(), yv.ravel(), zv.ravel()))
                rot = dat['rotation_matrix']
                plane_pts_rot = np.dot(np.linalg.inv(rot), plane_pts.T).T + center_point
                #plane_pts_rot = np.dot(rot, plane_pts.T).T + center_point


                vtk_pts = vtk.vtkPoints()
                for pt in plane_pts_rot:
                    vtk_pts.InsertNextPoint(pt)
                poly = vtk.vtkPolyData()
                poly.SetPoints(vtk_pts)
                delaunay = vtk.vtkDelaunay2D()
                delaunay.SetInputData(poly)
                delaunay.Update()

                writer = vtk.vtkXMLPolyDataWriter()
                writer.SetFileName(str(zone_dir / f"{key}_cutting_plane_zone{zone}.vtp"))
                writer.SetInputData(delaunay.GetOutput())
                writer.Write()
        # ✅ collect after all files are written
        print("Collecting all final cutting planes into one folder...")
        final_dir = output_dir / "all_zone_final"
        final_dir.mkdir(exist_ok=True)

        target_zones = {1, 3, 5, 7, 8, 9, 10, 12, 14, 17, 18, 19, 22, 23}

        import shutil
        for zone in target_zones:
            # 构造路径
            zone_str = f"{zone:.1f}"  # zone 可能是 float/int，需要确保是字符串
            src_file = output_dir / f"zone_{zone_str}" / f"final_cutting_plane_zone{zone_str}.vtp"
            dst_file = final_dir / f"final_cutting_plane_zone{zone_str}.vtp"

            # 执行复制
            if src_file.exists():
                shutil.copy2(src_file, dst_file)
            else:
                print(f"⚠️  Warning: {src_file} does not exist and was skipped.")

        print("✅ Export finished")

if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print("Usage: python aorta_mes_to_optimize_for_GPU_v1_0729.py <file_path>")
        sys.exit(1)
    file_path = sys.argv[1]
    #Step 1: Load segmentation mask

    img = sitk.ReadImage(file_path)
    spacing = img.GetSpacing()[0]  # voxel size (mm)
    mask = sitk.GetArrayFromImage(img)

    # Step 2: Initialize model & set device
    aorta_analysis = AortaAnalysis()
    aorta_analysis.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Step 3: Run diameter calculation
    start_time = time.time() 

    #diameters, centers, tangents, mc_map = aorta_analysis.calculate_all_diameters(mask)
    diameters, centers, tangents, mc_map, zone_slice_data,sampled_points, curve_points, original_skeleton_dilated = aorta_analysis.calculate_all_diameters(mask)

    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    # Step 4: Convert to mm and print
    #diameters_mm = {zone: d * spacing for zone, d in diameters.items()}
    diameters_mm = {
        zone: diameters[i] * spacing
        for i, zone in enumerate(zone_slice_data.keys())
    }

    print("\nZone-wise diameters (in mm):")
    for zone, dia in diameters_mm.items():
        print(f"Zone {zone}: {dia:.2f} mm")
    # skeleton_image1 = sitk.GetImageFromArray(original_skeleton_dilated[1].astype(np.uint8))
    # sitk.WriteImage(skeleton_image1, "skeleton_1.nii.gz")
    # skeleton_image2 = sitk.GetImageFromArray(original_skeleton_dilated[2].astype(np.uint8))
    # sitk.WriteImage(skeleton_image2, "skeleton_2.nii.gz")
#   
#     curve_points_0, tangents_0 = aorta_analysis.centerline_for_main_aorta(original_skeleton_dilated[0])
#     curve_points_1, tangents_1 = aorta_analysis.centerline(original_skeleton_dilated[1])
#     curve_points_2, tangents_2 = aorta_analysis.centerline(original_skeleton_dilated[2])

#     curve_point_segments = [curve_points_0, curve_points_1, curve_points_2]

#     aorta_analysis.export_vtk_with_cutting_planes(
#         multiclass_map_origin=mc_map,
#         curve_point_segments=curve_point_segments,
#         sampled_points=centers,
#         zone_slice_data=zone_slice_data,
#         output_dir="vtk_output_pipeline"
#     )