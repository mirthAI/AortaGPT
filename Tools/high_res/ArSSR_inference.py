from typing import Dict, Tuple
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as nd
import torch
from torch import autocast
from tqdm import tqdm
from itertools import product


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def normalise_volume(vol: np.ndarray) -> Tuple[np.ndarray, float, float]:
    vol = vol.astype(np.float32)
    vol_max = vol.max().item()
    vol_min = vol.min().item()
    denom = vol_max - vol_min
    if denom == 0:
        # Return zero-normalised volume along with original stats
        return np.zeros_like(vol, dtype=np.float32), vol_max, vol_min
    return (vol - vol_min) / denom, vol_max, vol_min

def make_coord(shape, ranges=None, flatten=True):
    """
    Input:
        shape: tuple, the shape of the image
        ranges: tuple, range of data in image normally [-1, 1] for normalized data
        flatten: bool, whether to flatten the image
    Output:
        ret: numpy array, the coordinates of the image
    Note:
    """

    # this already works for non-cubic data but doesn't support using different ranges for each dimension
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])  # (depth*height*width, 3)
    return ret

def gaussian_window_3d(shape, sigma_scale: float = 0.125) -> np.ndarray:
    """Return a separable 3-D Gaussian window with max value 1.

    Args:
        shape (tuple): (D, H, W) of the patch.
        sigma_scale (float): sigma = sigma_scale × patch_size
    """
    d, h, w = shape
    # Use the largest dimension as reference for sigma so that borders of all
    # axes are tapered similarly even for anisotropic patches.
    ref = max(shape)
    sigma = ref * sigma_scale

    def g(length):
        coords = np.arange(length)
        center = (length - 1) / 2.0
        return np.exp(-((coords - center) ** 2) / (2 * sigma ** 2))

    gz = g(d)
    gy = g(h)
    gx = g(w)
    window = (gz[:, None, None] * gy[None, :, None] * gx[None, None, :]).astype(
        np.float32
    )
    window /= window.max()
    return window

def extract_patches_single_image(
    image: sitk.Image,
    scale: float = 4.0,
    patch_size: int = 40,
    stride: int | None = None,
):
    """
    Generate LR/HR patch pairs from a single 3-D image.

    Args:
        image: sitk.Image. The high-resolution volume.
        scale: Down-sampling factor used to generate the LR cube.
        patch_size: Edge length of the cubic HR patches.
        stride: Sliding-window stride. Defaults to patch_size (non-overlapping).
    Returns:
        List[Tuple[np.ndarray, torch.Tensor, Tuple[int,int,int], torch.Tensor, int, Dict]]
        Each entry is identical to the output of PatchTestDataset.__getitem__.
    """
    image_np = sitk.GetArrayFromImage(image)
    image_stats = dict(
        spacing=image.GetSpacing(),
        origin=image.GetOrigin(),
        direction=image.GetDirection(),
    )
    stride = patch_size if stride is None else stride

    # Normalise full volume once
    vol_norm, vol_max, vol_min = normalise_volume(image_np)
    D, H, W = image_np.shape

    def _compute_starts(dim_len: int, window: int, step: int) -> list[int]:
        window = max(1, int(window))
        step = max(1, int(step))
        if dim_len <= window:
            return [0]
        starts = list(range(0, dim_len - window + 1, step))
        if starts[-1] != dim_len - window:
            starts.append(dim_len - window)
        return starts

    # Depth-only scaling: derive LR depth window/stride from HR ones
    z_window_lr = max(1, int(np.ceil(patch_size / float(scale))))
    z_stride_lr = max(1, int(np.floor(stride / float(scale))))

    # Height/Width are not scaled
    y_window = patch_size
    x_window = patch_size
    y_stride = stride
    x_stride = stride

    z_starts = _compute_starts(D, z_window_lr, z_stride_lr)
    y_starts = _compute_starts(H, y_window, y_stride)
    x_starts = _compute_starts(W, x_window, x_stride)

    patches = []
    total = len(z_starts) * len(y_starts) * len(x_starts)
    for z0, y0, x0 in product(z_starts, y_starts, x_starts):
    # for z0, y0, x0 in tqdm(product(z_starts, y_starts, x_starts), total=total, desc="Extract patches", position=0, leave=True):
        # Extract LR patch (depth window is LR-sized)
        patch_lr_np = vol_norm[
            z0 : z0 + z_window_lr,
            y0 : y0 + y_window,
            x0 : x0 + x_window,
        ]

        # Target HR patch shape (anisotropic scaling in depth only)
        hr_d_nominal = int(round(patch_lr_np.shape[0] * float(scale)))
        hr_d = min(patch_size, max(1, hr_d_nominal))
        target_hr_shape = (hr_d, min(y_window, patch_lr_np.shape[1]), min(x_window, patch_lr_np.shape[2]))

        # HR coordinates for implicit decoder
        dhw_hr = make_coord(target_hr_shape, flatten=True)

        # HR-space origin indices
        origin_hr = (int(round(z0 * float(scale))), y0, x0)

        image_stats = dict(
            size=image_np.shape,
            spacing=image.GetSpacing(),
            origin=image.GetOrigin(),
            direction=image.GetDirection(),
            vol_max=vol_max,
            vol_min=vol_min,
        )

        patches.append(
            (
                patch_lr_np.astype(np.float32),  # LR cube
                dhw_hr,                          # flattened HR coords
                origin_hr,                       # HR origin indices
                torch.empty(target_hr_shape, dtype=torch.float32),  # only shape is used
                image_stats,
            )
        )

    return patches


def super_resolve(model, image, scaling_factor: float, patch_size: int, stride: int):

    torch.set_float32_matmul_precision("high")

    # Pre-compute default Gaussian window for weighting (numpy array of shape patch_size³)
    gaussian_win_default = gaussian_window_3d((patch_size, patch_size, patch_size))

    model.eval()
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Buffers for reconstruction
    img_idx = 0  # single volume
    preds_sum: Dict[int, np.ndarray] = {}
    weight_sum: Dict[int, np.ndarray] = {}
    stats_dict: Dict[int, dict] = {}

    with torch.no_grad():
        patch_data = extract_patches_single_image(image, scaling_factor, patch_size, stride)
        for patch_lr, dhw_hr, (z0, y0, x0), patch_hr, image_stats in patch_data:
        # for patch_lr, dhw_hr, (z0, y0, x0), patch_hr, image_stats in tqdm(patch_data, desc="Predict patches", position=0, leave=True):
            # prepare inputs
            patch_lr_tensor = torch.from_numpy(patch_lr).unsqueeze(0).unsqueeze(1).to(DEVICE)  # 1x1xD_lrxHxW
            dhw_hr_tensor = dhw_hr.view(1, -1, 3).to(DEVICE).float()  # 1xQx3

            # forward
            with autocast(device_type=DEVICE.type, dtype=torch.float32):
                patch_pred = (
                    model(patch_lr_tensor.float(), dhw_hr_tensor)
                    .view(*patch_hr.shape)
                    .float()  # ensure fp32
                    .cpu()
                    .numpy()
                )  # (D_hr, H, W)

            # Gaussian weighting. Handle borders or volumes smaller than patch_size.
            if patch_hr.shape == gaussian_win_default.shape:
                window = gaussian_win_default
            else:
                window = gaussian_window_3d(patch_hr.shape)
            weighted_pred = patch_pred * window

            d, h, w = patch_hr.shape

            # allocate full-volume buffers on first use
            if img_idx not in preds_sum:
                # Allocate HR-sized output volume (depth scaled by 'scaling_factor')
                size_x, size_y, size_z = image.GetSize()
                D_hr = int(round(size_z * float(scaling_factor)))
                preds_sum[img_idx] = np.zeros((D_hr, size_y, size_x), dtype=np.float32)
                weight_sum[img_idx] = np.zeros((D_hr, size_y, size_x), dtype=np.float32)
                stats_dict[img_idx] = image_stats

            # accumulate
            # Safe accumulate with boundary clamping (handles rounding at the end)
            z1 = min(z0 + d, preds_sum[img_idx].shape[0])
            y1 = min(y0 + h, preds_sum[img_idx].shape[1])
            x1 = min(x0 + w, preds_sum[img_idx].shape[2])
            dz, dy, dx = z1 - z0, y1 - y0, x1 - x0
            if dz > 0 and dy > 0 and dx > 0:
                preds_sum[img_idx][z0:z1, y0:y1, x0:x1] += weighted_pred[:dz, :dy, :dx]
                weight_sum[img_idx][z0:z1, y0:y1, x0:x1] += window[:dz, :dy, :dx]

    # Reconstruct full-resolution prediction
    denom = np.maximum(weight_sum[img_idx], 1e-8)
    pred_np = preds_sum[img_idx] / denom

    # Un-normalise back to original intensity range if stats available
    if "vol_max" in stats_dict[img_idx] and "vol_min" in stats_dict[img_idx]:
        v_max = float(stats_dict[img_idx]["vol_max"])
        v_min = float(stats_dict[img_idx]["vol_min"])
        pred_to_save = pred_np * (v_max - v_min) + v_min
    else:
        pred_to_save = pred_np

    # Convert to SimpleITK image and copy spatial metadata
    img_out = sitk.GetImageFromArray(pred_to_save.astype(np.float32))
    if stats_dict[img_idx].get("direction") is not None:
        img_out.SetDirection([float(x) for x in stats_dict[img_idx]["direction"]])
    if stats_dict[img_idx].get("spacing") is not None:
        sx, sy, sz = [float(x) for x in stats_dict[img_idx]["spacing"]]
        img_out.SetSpacing([sx, sy, sz / float(scaling_factor)])
    if stats_dict[img_idx].get("origin") is not None:
        img_out.SetOrigin([float(x) for x in stats_dict[img_idx]["origin"]])

    return img_out


 