import time

import cvxpy as cp
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.path import Path
from scipy.ndimage import binary_fill_holes
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from skimage.measure import find_contours


def compute_aorta_diameter(binary_slice, center_point, device):
    """
    计算主动脉直径，并返回包含中心点的有效轮廓mask

    参数:
      binary_slice: 二值化的切片图像
      center_point: 中心点坐标 (x, y, z)
      device: 计算设备 (CPU/GPU)

    返回:
      max_diameter: 内切椭圆长轴直径（最大直径）
      contour_length: 轮廓长度（点数）
      valid_mask: 有效区域的mask
    """
    non_zero_points = np.where(binary_slice > 0)
    if len(non_zero_points[0]) == 0:
        return None, None, None

    # 找到所有轮廓
    contours = find_contours(binary_slice, 0.0)
    if not contours:
        return None, None, None

    # 确定中心点坐标（注意 x,y 顺序转换：center_point (x,y,z) -> (row, col) = (y, x)）
    center_2d = (int(round(center_point[1])), int(round(center_point[0])))
    selected_contour = None

    # 找到包含中心点的轮廓
    for contour in contours:
        path = Path(np.column_stack((contour[:, 0], contour[:, 1])))
        if path.contains_point((center_2d[0], center_2d[1])):
            selected_contour = contour
            break

    if selected_contour is None:
        return None, None, None

    # 创建mask
    mask = np.zeros_like(binary_slice, dtype=np.int32)
    contour_points = np.round(selected_contour).astype(int)
    rows = np.clip(contour_points[:, 0], 0, binary_slice.shape[0] - 1)
    cols = np.clip(contour_points[:, 1], 0, binary_slice.shape[1] - 1)
    mask[rows, cols] = 1
    mask = binary_fill_holes(mask).astype(np.int32)

    # 使用半正定规划求最大内切椭圆
    try:
        # 将 selected_contour 坐标从 (row, col) 转换为 (x, y) 格式：x=col, y=row
        points_xy = np.column_stack((selected_contour[:, 1], selected_contour[:, 0]))
        hull = ConvexHull(points_xy)
        # hull.equations: 每行 [a, b, c] 满足 a*x + b*y + c = 0，
        # 内部满足 a*x + b*y + c <= 0，即 a*x <= -c.
        A_poly = hull.equations[:, :2]   # shape (m,2)
        b_poly = -hull.equations[:, 2]     # shape (m,)

        # 定义变量：椭圆中心 c (2d) 和 2x2 对称正定矩阵 P，
        # 内切椭圆表示为 { x | ||P^{-1}(x - c)||_2 <= 1 }.
        c_var = cp.Variable(2)
        P_var = cp.Variable((2, 2), symmetric=True)

        # 引入一个小松弛变量 epsilon 来确保严格内切（可选）
        epsilon = 1e-6

        constraints = [P_var >> 1e-6 * np.eye(2)]
        for i in range(A_poly.shape[0]):
            ai = A_poly[i, :]
            # 加入松弛量：要求 a^T c + ||P^T a||_2 <= b - epsilon
            constraints.append(ai @ c_var + cp.norm(P_var.T @ ai) <= b_poly[i] - epsilon)

        # 目标函数：最大化 log(det(P))，面积正比于 det(P)
        objective = cp.Maximize(cp.log_det(P_var))
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)

        if P_var.value is None:
            raise Exception("CVXPY 求解失败")
        P_value = P_var.value
        # 对 P_value 进行特征分解，特征值即为椭圆半轴长度（注意是单位圆经过 P 映射后的尺度）
        eigvals = np.linalg.eigvalsh(P_value)
        major_axis = np.max(eigvals)
        max_diameter = 2 * major_axis

    except Exception as e:
        print("Convex optimization for inscribed ellipse failed, fallback to original method:", e)
        # fallback 方法：计算轮廓所有点两两之间的最大距离
        contour_tensor = torch.tensor(selected_contour, device=device)
        n_points = len(selected_contour)
        i_indices = torch.arange(n_points, device=device)
        j_indices = torch.arange(n_points, device=device)
        i, j = torch.meshgrid(i_indices, j_indices, indexing='ij')
        mask_indices = j > i
        points_i = contour_tensor[i[mask_indices]]
        points_j = contour_tensor[j[mask_indices]]
        distances = torch.norm(points_i - points_j, dim=1)
        max_diameter = distances.max().item()

    contour_length = len(selected_contour)
    return max_diameter, contour_length, mask

def compute_initial_rodrigues_rotation(tangent_vector):
    assert tangent_vector.shape == (3,)
    
    tangent_vector_norm = np.linalg.norm(tangent_vector)
    if tangent_vector_norm < 1e-10:
        return np.eye(3), tangent_vector, None, None

    tangent_vector = tangent_vector / tangent_vector_norm
    reference_xyz = np.array([0, 0, 1])
    cos_angle = np.dot(tangent_vector, reference_xyz)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    if angle > np.pi / 2:
        reference_xyz = np.array([0, 0, -1])
        cos_angle = np.dot(tangent_vector, reference_xyz)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    rotation_axis = np.cross(tangent_vector, reference_xyz)
    rotation_axis_norm = np.linalg.norm(rotation_axis)

    if rotation_axis_norm < 1e-10:
        return np.eye(3), tangent_vector, angle, rotation_axis

    rotation_axis = rotation_axis / rotation_axis_norm
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])
    rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    rotated_tangent = np.dot(rotation_matrix, tangent_vector)

    return rotation_matrix, rotated_tangent, angle, rotation_axis

def batch_compute_cuts(
    segmentation_map: np.ndarray,
    center_point: np.ndarray,
    batch_rotation_matrices: np.ndarray,
    device: torch.device,
    z_size=61  # 可选参数，默认 crop 61 个 slice
):
    batch_size = batch_rotation_matrices.shape[0]

    # ==============================
    # 先 crop segmentation_map 以减少计算
    z = int(round(center_point[2]))
    z_half = z_size // 2
    z_start = max(0, z - z_half)
    z_end = min(segmentation_map.shape[0], z + z_half)
    cropped_seg = segmentation_map[z_start:z_end, :, :]
    local_center_z = center_point[2] - z_start

    # 更新形状信息
    depth, height, width = cropped_seg.shape
    # ==============================

    # Move data to GPU
    seg_tensor = (
        torch.from_numpy(cropped_seg).float().to(device).unsqueeze(0).unsqueeze(0)
    )
    rot_matrices_torch = torch.from_numpy(batch_rotation_matrices).float().to(device)
    rot_matrices_inv = torch.inverse(rot_matrices_torch)
    center_pixel = torch.tensor(
        [center_point[0], center_point[1], local_center_z],
        dtype=torch.float32,
        device=device,
    )

    # Create a standard coordinate grid
    z_indices = torch.arange(depth, device=device)
    y_indices = torch.arange(height, device=device)
    x_indices = torch.arange(width, device=device)
    grid_z, grid_y, grid_x = torch.meshgrid(
        z_indices, y_indices, x_indices, indexing="ij"
    )

    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
    grid = grid.view(1, depth, height, width, 3) - center_pixel.view(1, 1, 1, 1, 3)

    rotated_grid = grid.reshape(1, -1, 3) @ rot_matrices_inv.transpose(1, 2)
    rotated_grid = rotated_grid.view(
        batch_size, depth, height, width, 3
    ) + center_pixel.view(1, 1, 1, 1, 3)

    norm_factor = torch.tensor([width - 1, height - 1, depth - 1], device=device).view(
        1, 1, 1, 1, 3
    )
    normalized_grid = 2 * (rotated_grid / norm_factor) - 1

    rotated_batch = F.grid_sample(
        seg_tensor.expand(batch_size, -1, -1, -1, -1),
        normalized_grid,
        mode="nearest",
        align_corners=True,
        padding_mode="zeros",
    )

    # 注意：这里 slice_index 需要用局部坐标
    slice_index = int(round(local_center_z))
    original_slices = rotated_batch[:, 0, slice_index, :, :]
    binary_slices = (original_slices > 0).int()

    return original_slices, binary_slices

def batch_compute_cuts_v2(
    segmentation_map: np.ndarray,
    center_points: np.ndarray,                # shape: (B, 3) — 每个为 (x, y, z)
    batch_rotation_matrices: np.ndarray,      # shape: (B, 3, 3)
    device: torch.device,
    z_size: int = 61,                         # crop 的深度范围
):
    """
    对多个中心点并行进行旋转切片提取。

    返回：
    - original_slices: torch.Tensor, shape (B, H, W)
    - binary_slices: torch.Tensor, shape (B, H, W)
    """
    batch_size = center_points.shape[0]
    all_original_slices = []
    all_binary_slices = []

    for i in range(batch_size):
        center_point = center_points[i]
        rot_matrix = batch_rotation_matrices[i]

        # === 局部 crop（在 z 轴方向上裁剪） ===
        z = int(round(center_point[2]))
        z_half = z_size // 2
        z_start = max(0, z - z_half)
        z_end = min(segmentation_map.shape[0], z + z_half)
        cropped_seg = segmentation_map[z_start:z_end, :, :]
        local_center_z = center_point[2] - z_start

        depth, height, width = cropped_seg.shape

        # === 准备数据并传到 GPU ===
        seg_tensor = torch.from_numpy(cropped_seg).float().to(device).unsqueeze(0).unsqueeze(0)
        rot_matrix_torch = torch.from_numpy(rot_matrix).float().to(device).unsqueeze(0)
        rot_inv = torch.inverse(rot_matrix_torch)  # shape: (1, 3, 3)
        center_pixel = torch.tensor(
            [center_point[0], center_point[1], local_center_z],
            dtype=torch.float32, device=device
        ).unsqueeze(0)

        # === 构造坐标网格并旋转 ===
        z_indices = torch.arange(depth, device=device)
        y_indices = torch.arange(height, device=device)
        x_indices = torch.arange(width, device=device)
        grid_z, grid_y, grid_x = torch.meshgrid(z_indices, y_indices, x_indices, indexing="ij")
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (D, H, W, 3)
        grid = grid.view(-1, 3) - center_pixel  # (N, 3)

        rotated_grid = grid @ rot_inv.squeeze(0).T + center_pixel  # (N, 3)

        norm_factor = torch.tensor([width - 1, height - 1, depth - 1], device=device).view(1, 3)
        norm_grid = 2 * (rotated_grid / norm_factor) - 1
        norm_grid = norm_grid.view(depth, height, width, 3).unsqueeze(0)

        # === grid_sample 采样 ===
        rotated_volume = F.grid_sample(
            seg_tensor,
            norm_grid,
            mode="nearest",
            align_corners=True,
            padding_mode="zeros"
        )

        # === 提取中心切片 ===
        slice_index = int(round(local_center_z))
        original_slice = rotated_volume[0, 0, slice_index, :, :]  # shape: (H, W)
        binary_slice = (original_slice > 0).int()

        all_original_slices.append(original_slice)
        all_binary_slices.append(binary_slice)

    # === 堆叠所有点的切片 ===
    all_original_slices = torch.stack(all_original_slices, dim=0)  # shape: (B, H, W)
    all_binary_slices = torch.stack(all_binary_slices, dim=0)      # shape: (B, H, W)

    return all_original_slices, all_binary_slices

def get_2d_cut_with_min_diameter(
    segmentation_map: np.ndarray,
    centerline_points: np.ndarray,
    tangent_vectors: np.ndarray,
    index: int,
):
    """
    获取最小直径的2D切面，使用GPU批处理加速角度搜索。
    """
    try:
        start_time = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        point = centerline_points[index]
        tangent_vector = tangent_vectors[index]

        # 1. 计算初始旋转矩阵，将切线对齐到z轴
        initial_rotation_matrix, _, _, _ = compute_initial_rodrigues_rotation(
            tangent_vector
        )
        if initial_rotation_matrix is None:
            return None, None

        # 2. 定义角度搜索空间并生成批次
        angle_range = torch.arange(-20, 21, 10, device=device).float()
        angles_x, angles_y, angles_z = torch.meshgrid(
            angle_range, angle_range, angle_range, indexing="ij"
        )
        angles_flat = torch.stack(
            [angles_x.flatten(), angles_y.flatten(), angles_z.flatten()], dim=-1
        )

        # 3. 创建批量的精调旋转矩阵
        fine_tune_rotations = R.from_euler(
            "xyz", angles_flat.cpu().numpy(), degrees=True
        )
        fine_tune_batch_np = fine_tune_rotations.as_matrix()

        # 4. 组合得到最终的旋转矩阵批次
        batch_rotations_np = fine_tune_batch_np @ initial_rotation_matrix

        # 5. 一次性计算所有切片（当前 point 重复 batch 次）
        center_points_batch = np.repeat(point[np.newaxis, :], batch_rotations_np.shape[0], axis=0)

        all_original_slices, all_binary_slices = batch_compute_cuts(
            segmentation_map,
            center_point = point,
            batch_rotation_matrices = batch_rotations_np,
            device=device,
        )

        # 将结果移至CPU进行后续处理
        all_original_slices_np = all_original_slices.cpu().numpy()
        all_binary_slices_np = all_binary_slices.cpu().numpy()

        # 6. 在CPU上循环查找最优结果
        min_diameter = float("inf")
        best_result_idx = -1
        results = []

        center_pixel_2d = (int(round(point[1])), int(round(point[0])))  # (row, col)

        for i in range(batch_rotations_np.shape[0]):
            binary_slice = all_binary_slices_np[i]

            # 检查中心点是否在切片内
            if (
                not (
                    0 <= center_pixel_2d[0] < binary_slice.shape[0]
                    and 0 <= center_pixel_2d[1] < binary_slice.shape[1]
                )
                or binary_slice[center_pixel_2d] == 0
            ):
                continue

            diameter, contour_len, valid_mask = compute_aorta_diameter(
                binary_slice, point, device
            )

            if diameter is not None:
                original_slice = all_original_slices_np[i]
                masked_slice = (
                    original_slice * valid_mask
                    if valid_mask is not None
                    else original_slice
                )

                # 记录所有有效结果
                results.append(
                    {
                        "diameter": diameter,
                        "binary_slice": binary_slice,
                        "masked_slice": masked_slice,
                        "original_slice": original_slice,
                        "valid_mask": valid_mask,
                        "contour_len": contour_len,
                        "rotation_matrix": batch_rotations_np[i],
                        "angles_deg": angles_flat[i].cpu().numpy().tolist(),
                    }
                )
                # 追踪最小直径
                if diameter < min_diameter:
                    min_diameter = diameter
                    best_result_idx = len(results) - 1

        if not results:
            return None, None

        # 7. 整理初始和最终数据
        # 初始数据是角度为(0,0,0)的结果
        initial_idx_search = np.where((angles_flat.cpu() == 0).all(axis=1))[0]
        initial_data_raw = None
        if len(initial_idx_search) > 0:
            initial_idx_val = initial_idx_search[0]
            for res in results:
                if res["angles_deg"] == [0.0, 0.0, 0.0]:
                    initial_data_raw = res
                    break

        # 如果(0,0,0)角度没有有效结果，则无法提供initial_data
        if initial_data_raw is None:
            # Fallback: use the best result as initial if no (0,0,0) result found
            initial_data_raw = results[best_result_idx]

        initial_data = {
            "rotation_matrix": initial_data_raw["rotation_matrix"],
            "slice": initial_data_raw["binary_slice"],
            "slice_info": initial_data_raw["masked_slice"],
            "original_slice_info": initial_data_raw["original_slice"],
            "diameter": initial_data_raw["diameter"],
            "center_point": point,
            "contour_length": initial_data_raw["contour_len"],
            "valid_mask": initial_data_raw["valid_mask"],
        }

        # 最终数据是直径最小的结果
        final_data_raw = results[best_result_idx]
        final_data = {
            "rotation_matrix": final_data_raw["rotation_matrix"],
            "slice": final_data_raw["binary_slice"],
            "slice_info": final_data_raw["masked_slice"],
            "original_slice_info": final_data_raw["original_slice"],
            "diameter": final_data_raw["diameter"],
            "center_point": point,
            "angles": tuple(final_data_raw["angles_deg"]),
            "contour_length": final_data_raw["contour_len"],
            "valid_mask": final_data_raw["valid_mask"],
        }

        # print(
        #     f"Point {index} | Initial Ø: {initial_data['diameter']:.2f}, Best Ø: {final_data['diameter']:.2f} at angles {final_data['angles']} | Time: {time.time() - start_time:.2f}s"
        # )
        return initial_data, final_data

    except Exception as e:
        import traceback

        print(f"Error in get_2d_cut_with_min_diameter for point {index}: {e}")
        traceback.print_exc()
        return None, None