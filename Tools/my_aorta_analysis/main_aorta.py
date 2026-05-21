import gc
import os
import random
import time
import traceback
from contextlib import contextmanager
from datetime import timedelta

import numpy as np
import torch
from aorta_analysis.diameter_utils import (
    batch_compute_cuts,
    compute_aorta_diameter,
    compute_initial_rodrigues_rotation,
    get_2d_cut_with_min_diameter,
)
from aorta_analysis.downsample_utils import downsample_centerline_and_segmentation
from aorta_analysis.io_utils import (
    crop_image_array,
    pad_with_percentage,
    preprocess_binary_segment,
    preprocess_keep_classes,
    read_segmentation,
)
from aorta_analysis.skeleton_utils import create_binary_curve_image
from aorta_analysis.skeleton_utils import find_endpoints_aorta as find_endpoints
from aorta_analysis.skeleton_utils import (
    fit_polynomial_robust_aorta as fit_polynomial_robust,
)
from aorta_analysis.skeleton_utils import (
    generate_skeleton,
    get_centerline_dijkstra,
    skeletonize_with_fallback,
    smooth_tangents,
)
from aorta_analysis.visualization import (
    plot_centerline_and_fitted_curve_aorta as plot_centerline_and_fitted_curve,
)
from aorta_analysis.visualization import visualize_3d_skeleton
from aorta_analysis.zone_analysis import (
    analyze_centerline_zones,
    analyze_slice_zones,
    calculate_zone_diameters,
)

if torch.cuda.is_available():
    _ = torch.zeros(1).to("cuda")


def main(filepath):
    total_start_time = time.time()

    # 1️⃣ 读 segmentation + mask + crop + pad
    # filepath = "Input/Zone_seg/0301_AI_predict_61_100/images_extracted/predicted_labels/62_label.mha"
    arr = read_segmentation(filepath)
    keep_classes = [1, 3, 5, 7, 8, 9, 10, 12, 14, 17]  # 主动脉 zone classes
    multiclass_array_origin = preprocess_keep_classes(arr, keep_classes)
    cropped_array = crop_image_array(multiclass_array_origin)
    # print(f"Original shape: {multiclass_array_origin.shape}, Cropped shape: {cropped_array.shape}")

    multiclass_map_origin = pad_with_percentage(cropped_array, padding_percentage=0.2)
    # print(f"After padding shape: {multiclass_map_origin.shape}")

    # 2️⃣ Binary 预处理
    binary_map = multiclass_map_origin.copy()
    binary_map[binary_map > 0] = 1
    processed_segment = preprocess_binary_segment(binary_map)
    # print("Binary preprocessing done.")

    # 3️⃣ Skeleton 生成
    original_skeleton_dilated = generate_skeleton(processed_segment)
    # print("Skeleton generation done.")

    # 4️⃣ 可视化（可选）
    # visualize_3d_skeleton(original_skeleton_dilated)

    # 5️⃣ Centerline extraction and fitting for AORTA
    endpoints = find_endpoints(original_skeleton_dilated)
    B, A = sorted(endpoints, key=lambda p: p[0])

    original_path = get_centerline_dijkstra(original_skeleton_dilated, A, B, binary_map)
    split_idx = np.argmax(original_path[:, 0])

    ascending_part = original_path[: split_idx + 1]
    descending_part = original_path[split_idx:]

    # 确保连接点重叠
    overlap_point = original_path[split_idx]
    ascending_part = np.vstack([ascending_part, overlap_point])
    descending_part = np.vstack([overlap_point, descending_part])

    # print(f"Ascending points: {len(ascending_part)}, Descending points: {len(descending_part)}")

    # 分段拟合
    ascending_curve, ascending_tangents = fit_polynomial_robust(
        ascending_part, is_connection_point=1.0
    )
    descending_curve, descending_tangents = fit_polynomial_robust(
        descending_part, is_connection_point=0.0
    )

    # 在连接处平滑
    blend_window = 5
    for i in range(blend_window):
        alpha = i / blend_window
        blended_tangent = (1 - alpha) * ascending_tangents[
            -blend_window + i
        ] + alpha * descending_tangents[i]
        blended_tangent /= np.linalg.norm(blended_tangent)
        ascending_tangents[-blend_window + i] = blended_tangent
        descending_tangents[i] = blended_tangent

    # 额外平滑
    ascending_tangents = smooth_tangents(ascending_tangents)
    descending_tangents = smooth_tangents(descending_tangents)

    # 生成 binary mask for fitted curve
    ascending_curve_img = create_binary_curve_image(
        original_skeleton_dilated, ascending_curve
    )
    descending_curve_img = create_binary_curve_image(
        original_skeleton_dilated, descending_curve
    )
    fitted_curve_img = np.maximum(ascending_curve_img, descending_curve_img)

    # 合并
    curve_points_origin = np.vstack((ascending_curve, descending_curve))
    tangents_origin = np.vstack((ascending_tangents, descending_tangents))

    # 可视化
    plot_centerline_and_fitted_curve(
        original_path,
        ascending_curve,
        descending_curve,
        ascending_tangents,
        descending_tangents,
        special_index=30,
    )

    # 6️⃣ 坐标调整
    curve_points_origin = curve_points_origin[:, [2, 1, 0]]
    tangents_origin = tangents_origin[:, [2, 1, 0]]

    # print("\nFinal Results:")
    # print(f"Total curve points: {len(curve_points_origin)}")
    # print(f"Total tangent vectors: {len(tangents_origin)}")

    # 7️⃣ downsample
    downsample_factor = 2
    curve_points, tangents, multiclass_map, multiclass_map_sitk = (
        downsample_centerline_and_segmentation(
            curve_points_origin,
            tangents_origin,
            multiclass_map_origin,
            downsample_factor,
        )
    )

    # 8️⃣ 分析 zone
    zone_ranges, point_zones = analyze_centerline_zones(curve_points, multiclass_map)

    # 9️⃣ 计算最大直径
    zone_max_diameters, zone_slice_data = calculate_zone_diameters(
        multiclass_map, curve_points, tangents, zone_ranges, step=10
    )

    # 输出摘要
    print("=" * 60)
    print("\nZone Analysis Summary:")
    zone_diameter_dict = {}

    for zone, info in zone_max_diameters.items():
        diameter_corrected = info["max_diameter"] * 2  # 乘 2 补偿 downsample
        print(f"\nZone {zone}:")
        print(f"  Maximum Diameter: {diameter_corrected:.2f}")
        print(f"  At Index: {info['index']}")
        print(f"  Valid Points Analyzed: {info['valid_points']}")
        zone_diameter_dict[zone] = round(diameter_corrected, 2)

    print("\nZone Diameter Dictionary (corrected for downsample):")
    print(zone_diameter_dict)

    total_time = time.time() - total_start_time
    print(f"\nTotal execution time: {timedelta(seconds=int(total_time))}")
    print(f"Total time: {total_time:.2f} seconds")


import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main_aorta.py <file_path>")
        sys.exit(1)
    file_path = sys.argv[1]
    main(file_path)
