from aorta_analysis.visualization import (visualize_3d_skeleton,plot_centerline_and_fitted_curve)
from aorta_analysis.io_utils import (
    read_segmentation, preprocess_keep_classes, 
    crop_image_array, pad_with_percentage, 
    preprocess_binary_segment
)
from aorta_analysis.skeleton_utils import (
     skeletonize_with_fallback, generate_skeleton, find_endpoints, get_centerline_dijkstra,
    fit_polynomial_robust, smooth_tangents, create_binary_curve_image
)
from aorta_analysis.diameter_utils import (
    compute_aorta_diameter,
    compute_initial_rodrigues_rotation,
    batch_compute_cuts,
    get_2d_cut_with_min_diameter
)
from aorta_analysis.zone_analysis import (
    analyze_centerline_zones,
    analyze_slice_zones,
    calculate_zone_diameters
)

from aorta_analysis.downsample_utils import downsample_centerline_and_segmentation
import numpy as np
import os
import random
import time
import traceback
from datetime import timedelta
import gc
from contextlib import contextmanager
import sys
import torch

if torch.cuda.is_available():
    _ = torch.zeros(1).to("cuda")

def main(filepath):
    total_start_time = time.time()
    # 1️⃣ 读 segmentation + mask + crop + pad（
    #filepath = "Input/Zone_seg/0301_AI_predict_61_100/images_extracted/predicted_labels/62_label.mha"
    arr = read_segmentation(filepath)
    keep_classes = [18]  # LCIA 需要保留的 class
    multiclass_array_origin = preprocess_keep_classes(arr, keep_classes)
    cropped_array = crop_image_array(multiclass_array_origin)
    #print(f"Original shape: {multiclass_array_origin.shape}, Cropped shape: {cropped_array.shape}")

    multiclass_map_origin = pad_with_percentage(cropped_array, padding_percentage=0.2)
    #print(f"After padding shape: {multiclass_map_origin.shape}")
    # 2️⃣ Binary 预处理
    binary_start_time = time.time()
    binary_map = multiclass_map_origin.copy()
    binary_map[binary_map > 0] = 1
    processed_segment = preprocess_binary_segment(binary_map)
    print(f"Binary preprocessing done in {time.time() - binary_start_time:.2f} s")
    # 3️⃣ Skeleton 生成
    dilation_start_time = time.time()
    original_skeleton_dilated = generate_skeleton(processed_segment)
    print(f"Skeleton generation done in {time.time() - dilation_start_time:.2f} s")

    # 4️⃣ 可视化（可选）
    visualize_3d_skeleton(original_skeleton_dilated)
    # 5️⃣ Centerline extraction and fitting
    endpoints = find_endpoints(original_skeleton_dilated)
    B, A = endpoints

    original_path = get_centerline_dijkstra(original_skeleton_dilated, A, B, binary_map)
    curve, tangents = fit_polynomial_robust(original_path)
    tangents = smooth_tangents(tangents)

    fitted_curve_img = create_binary_curve_image(original_skeleton_dilated, curve)

    plot_centerline_and_fitted_curve(original_path, curve, curve, tangents, tangents, special_index=30)

    # 6️⃣坐标调整
    # Final result transformation
    curve_points_origin = curve[:, [2, 1, 0]]
    tangents_origin = tangents[:, [2, 1, 0]]

    print("\nFinal Results:")
    print(f"Total curve points: {len(curve_points_origin)}")
    print(f"Total tangent vectors: {len(tangents_origin)}")
    print(f"Curve z range: {curve_points_origin[:, 2].min():.2f} to {curve_points_origin[:, 2].max():.2f}")
    
    #7️⃣ segmentation 和 centerline/tangent downsample
    downsample_factor = 2
    curve_points, tangents, multiclass_map, multiclass_map_sitk = \
        downsample_centerline_and_segmentation(
            curve_points_origin, tangents_origin, multiclass_map_origin, downsample_factor
        )
    # 8️⃣分析中心线点的zone分布
    zone_ranges, point_zones = analyze_centerline_zones(curve_points, multiclass_map)

    # 9️⃣  计算每个zone的最大直径
    zone_max_diameters, zone_slice_data = calculate_zone_diameters(
        multiclass_map, curve_points, tangents, zone_ranges, step=10
    )
    # 打印结果摘要
    print("=" * 60)
    print("\nZone Analysis Summary:")
    
    zone_diameter_dict = {}  # 新增 dictionary
    
    for zone, info in zone_max_diameters.items():
        diameter_corrected = info['max_diameter'] * 2  # 乘以 2 补偿 downsample
        print(f"\nZone {zone}:")
        print(f"  Maximum Diameter: {diameter_corrected:.2f}")
        print(f"  At Index: {info['index']}")
        print(f"  Valid Points Analyzed: {info['valid_points']}")

        zone_diameter_dict[zone] = round(diameter_corrected, 2)  # 加入 dict
    
        # 输出 dictionary
    print("\nZone Diameter Dictionary (corrected for downsample):")
    print(zone_diameter_dict)

    total_time = time.time() - total_start_time
    print(f"\nTotal execution time: {timedelta(seconds=int(total_time))}")

    print(f"Total time: {total_time:.2f} seconds")
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main_rcia.py <file_path>")
        sys.exit(1)
    file_path = sys.argv[1]
    main(file_path)
