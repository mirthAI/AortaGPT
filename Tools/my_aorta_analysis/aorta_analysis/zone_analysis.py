import time

import numpy as np
import torch

# Import 依赖的 utility 函数
from .diameter_utils import (
    batch_compute_cuts,
    batch_compute_cuts_v2,
    compute_aorta_diameter,
    compute_initial_rodrigues_rotation,
    get_2d_cut_with_min_diameter,
)


def analyze_centerline_zones(centerline_points, segmentation_map):
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
    
    print("\nAnalyzing centerline points zones...")
    
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
            
            #print(f"\nZone {zone}:")
            #print(f"  Range: {indices[0]} - {indices[-1]}")
            #print(f"  Total points: {len(indices)}")
    
    return zone_ranges, point_zones

def analyze_slice_zones(slice_img):
    """
    分析切片中包含的zone
    
    参数:
    - slice_img: 2D切片图像
    
    返回:
    - unique_zones: 切片中包含的不同zone值
    """
    # 获取非零区域的唯一值
    unique_zones = np.unique(slice_img[slice_img > 0])
    return unique_zones

def calculate_zone_diameters(segmentation_map, centerline_points, tangent_vectors, zone_ranges, step=10):
    """GPU 批量旋转 + CPU 直径与 zone 判定"""
    start_time = time.time()
    zone_max_diameters = {}
    zone_slice_data = {}
    zone_boundary_handoff = {}

    point_cache = {}
    sorted_zones = sorted(zone_ranges.keys())

    # --- 预先收集所有要处理的 unique indices ---
    all_indices_to_process = set()
    for zone_idx, zone in enumerate(sorted_zones):
        indices = zone_ranges[zone]['indices']
        total_points = len(indices)

        if total_points > 100:
            actual_step = 9
        elif total_points > 50:
            actual_step = 7
        elif total_points > 20:
            actual_step = 6
        elif total_points > 10:
            actual_step = 4
        elif total_points > 6:
            actual_step = 2
        else:
            actual_step = 1

        boundary_check_points = 3
        initial_check_points = 3
        start_idx = initial_check_points if zone_idx != 0 else 0

        if zone_idx != 0:
            all_indices_to_process.update(indices[:initial_check_points])

        selected_indices = (
            indices[start_idx:-boundary_check_points:actual_step] if zone_idx != len(sorted_zones)-1 else indices[start_idx::actual_step]
        )
        all_indices_to_process.update(selected_indices)

        if zone_idx != len(sorted_zones)-1:
            all_indices_to_process.update(indices[-boundary_check_points:])

    unique_indices = sorted(all_indices_to_process)
    print(f"\nCollected {len(unique_indices)} unique points to process across all zones.")

    # --- GPU 预处理旋转 ---
    print(f"Starting GPU rotation pre-processing...")
    rotated_slices_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for idx in unique_indices:
        point = centerline_points[idx]
        tangent = tangent_vectors[idx]

        initial_rot, *_ = compute_initial_rodrigues_rotation(tangent)
        batch_rot = np.expand_dims(initial_rot, axis=0)

        original_slices, binary_slices = batch_compute_cuts(
            segmentation_map, point, batch_rot, device
        )

        rotated_slices_dict[idx] = {
            "binary_slice": binary_slices[0].cpu().numpy(),
            "original_slice": original_slices[0].cpu().numpy(),
            "point": point
        }

    print(f"GPU rotation finished for {len(rotated_slices_dict)} points.")

    # --- 修改后的 process_point() ---
    def process_point(idx):
        if idx in point_cache:
            return point_cache[idx]

        if idx not in rotated_slices_dict:
            return None, None

        rotated = rotated_slices_dict[idx]
        binary_slice = rotated["binary_slice"]
        point = rotated["point"]

        diameter, contour_len, mask = compute_aorta_diameter(binary_slice, point, torch.device("cpu"))

        if diameter and diameter > 0:
            initial_data = {"slice": binary_slice}
            final_data = {
                "diameter": diameter,
                "slice_info": rotated["original_slice"]  # 注意这里 slice_info 是 binary_slice 用于 zone 判断
            }
            point_cache[idx] = (initial_data, final_data)
            return initial_data, final_data

        return None, None

    # --- 保持原有准确性逻辑不变 ---
    for zone_idx, zone in enumerate(sorted_zones):
        is_first_zone = (zone_idx == 0)
        is_last_zone = (zone_idx == len(sorted_zones) - 1)
        indices = zone_ranges[zone]['indices']
        total_points = len(indices)

        valid_diameters = []
        valid_diameter_indices = []
        valid_slice_data_list = []

        boundary_check_points = 3
        initial_check_points = 3
        start_idx = initial_check_points if not is_first_zone else 0

        if not is_first_zone and zone in zone_boundary_handoff:
            handoff_data = zone_boundary_handoff[zone]
            handoff_data.sort(key=lambda x: x['index'])
            for point_data in handoff_data:
                slice_array = point_data['slice_data']['final_data']['slice_info']
                total_pixels = np.sum(slice_array > 0)
                zone_pixels = {z: (np.sum(slice_array == z) / total_pixels) * 100 for z in analyze_slice_zones(slice_array)}
                if not zone_pixels:
                    continue
                max_zone = max(zone_pixels.items(), key=lambda x: x[1])[0]
                if max_zone == zone:
                    valid_diameters.append(point_data['diameter'])
                    valid_diameter_indices.append(point_data['index'])
                    valid_slice_data_list.append(point_data['slice_data'])

        if not is_first_zone:
            for idx in indices[:initial_check_points]:
                initial_data, final_data = process_point(idx)
                if final_data:
                    slice_array = final_data['slice_info']
                    total_pixels = np.sum(slice_array > 0)
                    zone_pixels = {z: (np.sum(slice_array == z) / total_pixels) * 100 for z in analyze_slice_zones(slice_array)}
                    if not zone_pixels:
                        continue
                    max_zone_id = max(zone_pixels.items(), key=lambda x: x[1])[0]
                    if max_zone_id == zone:
                        valid_diameters.append(final_data['diameter'])
                        valid_diameter_indices.append(idx)
                        valid_slice_data_list.append({
                            'initial_data': initial_data,
                            'final_data': final_data,
                            'centerline_index': idx,
                            'zone': zone
                        })

        selected_indices = (
            indices[start_idx:-boundary_check_points] if not is_last_zone else indices[start_idx:]
        )
        for idx in selected_indices:
            initial_data, final_data = process_point(idx)
            if final_data:
                valid_diameters.append(final_data['diameter'])
                valid_diameter_indices.append(idx)
                valid_slice_data_list.append({
                    'initial_data': initial_data,
                    'final_data': final_data,
                    'centerline_index': idx,
                    'zone': zone
                })

        if not is_last_zone:
            for idx in indices[-boundary_check_points:]:
                initial_data, final_data = process_point(idx)
                if final_data:
                    slice_array = final_data['slice_info']
                    total_pixels = np.sum(slice_array > 0)
                    zone_pixels = {z: (np.sum(slice_array == z) / total_pixels) * 100 for z in analyze_slice_zones(slice_array)}
                    if not zone_pixels:
                        continue
                    max_zone_id = max(zone_pixels.items(), key=lambda x: x[1])[0]
                    point_data = {
                        'index': idx,
                        'diameter': final_data['diameter'],
                        'slice_data': {
                            'initial_data': initial_data,
                            'final_data': final_data,
                            'centerline_index': idx,
                            'zone': max_zone_id
                        }
                    }
                    if max_zone_id == zone:
                        valid_diameters.append(final_data['diameter'])
                        valid_diameter_indices.append(idx)
                        valid_slice_data_list.append(point_data['slice_data'])
                    else:
                        if max_zone_id in zone_boundary_handoff:
                            zone_boundary_handoff[max_zone_id].append(point_data)
                        else:
                            zone_boundary_handoff[max_zone_id] = [point_data]

        if valid_diameters:
            max_idx = np.argmax(valid_diameters)
            zone_max_diameters[zone] = {
                'max_diameter': valid_diameters[max_idx],
                'index': valid_diameter_indices[max_idx],
                'valid_points': len(valid_diameters),
            }
            zone_slice_data[zone] = valid_slice_data_list[max_idx]
        else:
            zone_max_diameters[zone] = {
                'max_diameter': float('nan'),
                'index': None,
                'valid_points': 0,
            }

    total_time = time.time() - start_time
    print(f"\nTotal Zone Calculation Time: {total_time:.2f} seconds")
    return zone_max_diameters, zone_slice_data

def calculate_zone_diameters_v2(segmentation_map, centerline_points, tangent_vectors, zone_ranges, step=10):
    """GPU 批量旋转 + CPU 直径与 zone 判定"""
    start_time = time.time()
    zone_max_diameters = {}
    zone_slice_data = {}
    zone_boundary_handoff = {}
    ZONE_RATIO_THRESHOLD = 90  # 区域像素占比阈值（单位 %）

    point_cache = {}
    sorted_zones = sorted(zone_ranges.keys())

    # --- 预先收集所有要处理的 unique indices ---
    all_indices_to_process = set()
    for zone_idx, zone in enumerate(sorted_zones):
        indices = zone_ranges[zone]['indices']
        total_points = len(indices)

        if total_points > 100:
            actual_step = 9
        elif total_points > 50:
            actual_step = 7
        elif total_points > 20:
            actual_step = 6
        elif total_points > 10:
            actual_step = 4
        elif total_points > 6:
            actual_step = 2
        else:
            actual_step = 1

        boundary_check_points = 3
        initial_check_points = 3
        start_idx = initial_check_points if zone_idx != 0 else 0

        if zone_idx != 0:
            all_indices_to_process.update(indices[:initial_check_points])

        selected_indices = (
            indices[start_idx:-boundary_check_points:actual_step] if zone_idx != len(sorted_zones)-1 else indices[start_idx::actual_step]
        )
        all_indices_to_process.update(selected_indices)

        if zone_idx != len(sorted_zones)-1:
            all_indices_to_process.update(indices[-boundary_check_points:])

    unique_indices = sorted(all_indices_to_process)
    print(f"\nCollected {len(unique_indices)} unique points to process across all zones.")

    # --- GPU 预处理旋转 ---
    print(f"Starting GPU rotation pre-processing...")
    rotated_slices_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # === 并行提取所有点、切片角度 ===
    all_points = []
    all_tangents = []
    all_indices = []

    for idx in unique_indices:
        all_points.append(centerline_points[idx])
        all_tangents.append(tangent_vectors[idx])
        all_indices.append(idx)

    # === 批量旋转矩阵 ===
    rot_matrices = np.stack(
        [compute_initial_rodrigues_rotation(tangent)[0] for tangent in all_tangents],
        axis=0
    )

    # === 批量切片 ===
    # 由于你的 batch_compute_cuts 目前接收的是单个 center_point（不是批量），
    # 我们做个轻微重写，让 center_point 扩展为 batch 版本：
    center_points = np.array(all_points)  # shape = (B, 3)
    original_slices, binary_slices = batch_compute_cuts_v2(
        segmentation_map,
        center_points=center_points,
        batch_rotation_matrices=rot_matrices,
        device=device
    )
    # === 保存每个切片 ===
    for i, idx in enumerate(all_indices):
        rotated_slices_dict[idx] = {
            "binary_slice": binary_slices[i].cpu().numpy(),
            "original_slice": original_slices[i].cpu().numpy(),
            "point": all_points[i],
        }

    print(f"GPU rotation finished for {len(rotated_slices_dict)} points.")

    # --- 修改后的 process_point() ---
    def process_point(idx):
        if idx in point_cache:
            return point_cache[idx]

        if idx not in rotated_slices_dict:
            return None, None

        rotated = rotated_slices_dict[idx]
        binary_slice = rotated["binary_slice"]
        point = rotated["point"]

        diameter, contour_len, mask = compute_aorta_diameter(binary_slice, point, torch.device("cpu"))

        if diameter and diameter > 0:
            initial_data = {"slice": binary_slice}
            final_data = {
                "diameter": diameter,
                "slice_info": rotated["original_slice"]  # 注意这里 slice_info 是 binary_slice 用于 zone 判断
            }
            point_cache[idx] = (initial_data, final_data)
            return initial_data, final_data

        return None, None

    # --- 保持原有准确性逻辑不变 ---
    for zone_idx, zone in enumerate(sorted_zones):
        is_first_zone = (zone_idx == 0)
        is_last_zone = (zone_idx == len(sorted_zones) - 1)
        indices = zone_ranges[zone]['indices']
        total_points = len(indices)

        valid_diameters = []
        valid_diameter_indices = []
        valid_slice_data_list = []

        boundary_check_points = 3
        initial_check_points = 3
        start_idx = initial_check_points if not is_first_zone else 0

        if not is_first_zone and zone in zone_boundary_handoff:
            handoff_data = zone_boundary_handoff[zone]
            handoff_data.sort(key=lambda x: x['index'])
            for point_data in handoff_data:
                slice_array = point_data['slice_data']['final_data']['slice_info']
                total_pixels = np.sum(slice_array > 0)
                zone_pixels = {
                    z: (np.sum(slice_array == z) / total_pixels) * 100
                    for z in analyze_slice_zones(slice_array)
                }
                if not zone_pixels:
                    continue
                zone_ratio = zone_pixels.get(zone, 0)
                if zone_ratio >= ZONE_RATIO_THRESHOLD:
                    valid_diameters.append(point_data['diameter'])
                    valid_diameter_indices.append(point_data['index'])
                    valid_slice_data_list.append(point_data['slice_data'])


        if not is_first_zone:
            for idx in indices[:initial_check_points]:
                    initial_data, final_data = process_point(idx)
                    if final_data:
                        slice_array = final_data['slice_info']
                        total_pixels = np.sum(slice_array > 0)
                        zone_pixels = {z: (np.sum(slice_array == z) / total_pixels) * 100 for z in analyze_slice_zones(slice_array)}
                        if not zone_pixels:
                            continue
                        zone_ratio = zone_pixels.get(zone, 0)
                        if zone_ratio >= ZONE_RATIO_THRESHOLD:
                            valid_diameters.append(final_data['diameter'])
                            valid_diameter_indices.append(idx)
                            valid_slice_data_list.append({
                                'initial_data': initial_data,
                                'final_data': final_data,
                                'centerline_index': idx,
                                'zone': zone
                            })


        selected_indices = (
            indices[start_idx:-boundary_check_points] if not is_last_zone else indices[start_idx:]
        )
        for idx in selected_indices:
            initial_data, final_data = process_point(idx)
            if final_data:
                valid_diameters.append(final_data['diameter'])
                valid_diameter_indices.append(idx)
                valid_slice_data_list.append({
                    'initial_data': initial_data,
                    'final_data': final_data,
                    'centerline_index': idx,
                    'zone': zone
                })

        if not is_last_zone:
            for idx in indices[-boundary_check_points:]:
                initial_data, final_data = process_point(idx)
                if final_data:
                    slice_array = final_data['slice_info']
                    total_pixels = np.sum(slice_array > 0)
                    zone_pixels = {z: (np.sum(slice_array == z) / total_pixels) * 100 for z in analyze_slice_zones(slice_array)}
                    if not zone_pixels:
                        continue
                    zone_ratio = zone_pixels.get(zone, 0)
                    point_data = {
                        'index': idx,
                        'diameter': final_data['diameter'],
                        'slice_data': {
                            'initial_data': initial_data,
                            'final_data': final_data,
                            'centerline_index': idx,
                            'zone': zone
                        }
                    }
                    if zone_ratio >= ZONE_RATIO_THRESHOLD:
                        valid_diameters.append(final_data['diameter'])
                        valid_diameter_indices.append(idx)
                        valid_slice_data_list.append(point_data['slice_data'])
                    else:
                        if zone in zone_boundary_handoff:
                            zone_boundary_handoff[zone].append(point_data)
                        else:
                            zone_boundary_handoff[zone] = [point_data]

        if valid_diameters:
            max_idx = np.argmax(valid_diameters)
            baseline_max_diameter = valid_diameters[max_idx]
            baseline_max_index = valid_diameter_indices[max_idx]
            baseline_slice_data = valid_slice_data_list[max_idx] 
            zone_max_diameters[zone] = {
                'max_diameter': baseline_max_diameter,
                'index': baseline_max_index,
                'valid_points': len(valid_diameters),
            }
            zone_slice_data[zone] = {
                    'initial_data': baseline_slice_data['initial_data'],
                    'final_data': baseline_slice_data['final_data'],
                    'centerline_index': baseline_max_index,
                    'zone': zone
                }
            # === 这里开始加 refine 逻辑 ===
            # 取 top 3 diameter 对应的 index
            #top3_indices_in_valid = np.argsort(valid_diameters)[-3:]  # index in valid_diameters list
            top3_indices_in_valid = (
    np.argsort(valid_diameters)[-3:] if len(valid_diameters) >= 3 else np.argsort(valid_diameters)
)

            refined_results = []
            for idx_in_valid in top3_indices_in_valid:
                candidate_idx = valid_diameter_indices[idx_in_valid]

                init_data, final_data = get_2d_cut_with_min_diameter(
                    segmentation_map,
                    centerline_points,
                    tangent_vectors,
                    candidate_idx
                )

                if final_data is not None:
                    refined_results.append({
                        'diameter': final_data['diameter'],
                        'index': candidate_idx,
                        'init_data': init_data,
                        'final_data': final_data
                    })

            if refined_results:
                # 取 refine 后 diameter 的最大值（zone 的最终 max diameter）
                refined_best = max(refined_results, key=lambda x: x['diameter'])
                refined_diameter = refined_best['diameter']
                refined_index = refined_best['index']

                print(f"Zone {zone}: refined max diameter from {baseline_max_diameter:.2f} to {refined_diameter:.2f}")

                zone_max_diameters[zone] = {
                    'max_diameter': refined_diameter,
                    'index': refined_index,
                    'valid_points': len(valid_diameters),
                }
                zone_slice_data[zone] = {
                    'initial_data': refined_best['init_data'],
                    'final_data': refined_best['final_data'],
                    'centerline_index': refined_index,
                    'zone': zone
                }


    total_time = time.time() - start_time
    print(f"\nTotal Zone Calculation Time: {total_time:.2f} seconds")
    return zone_max_diameters, zone_slice_data
