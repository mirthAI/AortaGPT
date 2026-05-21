import SimpleITK as sitk
import numpy as np
from skimage.morphology import ball, opening, closing
from scipy.ndimage import median_filter,binary_erosion, distance_transform_edt

def read_segmentation(filepath):
    img = sitk.ReadImage(filepath)
    arr = sitk.GetArrayFromImage(img)
    return arr

def preprocess_keep_classes(arr, keep_classes):
    arr_new = arr.copy()
    mask = ~np.isin(arr, keep_classes)
    arr_new[mask] = 0
    return arr_new

def crop_image_array(array):
    nonzero_coords = np.argwhere(array)
    min_z, min_y, min_x = nonzero_coords.min(axis=0)
    max_z, max_y, max_x = nonzero_coords.max(axis=0)
    return array[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]

def pad_with_percentage(volume, padding_percentage=0.2):
    depth, height, width = volume.shape
    pad_depth = int(depth * padding_percentage)
    pad_height = int(height * padding_percentage)
    pad_width = int(width * padding_percentage)
    new_shape = (depth + 2 * pad_depth, height + 2 * pad_height, width + 2 * pad_width)
    new_array = np.zeros(new_shape, dtype=volume.dtype)
    new_array[pad_depth:pad_depth + depth, pad_height:pad_height + height, pad_width:pad_width + width] = volume
    return new_array

def preprocess_binary_segment(binary_segment, radius=2, filter_size=3):
    selem = ball(radius)
    opened = opening(binary_segment, selem)
    closed = closing(opened, selem)
    processed_segment = median_filter(closed, size=filter_size)
    return processed_segment

def compute_fatness_ratio(binary_segment):
    coords = np.argwhere(binary_segment > 0)
    if coords.shape[0] == 0:
        return 0
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0) + 1  # inclusive
    bbox_volume = np.prod(bbox_max - bbox_min)
    mask_volume = np.count_nonzero(binary_segment)
    return mask_volume / bbox_volume if bbox_volume > 0 else 0

def preprocess_binary_segment_hybrid(binary_segment, radius=2, filter_size=3,
                                     erosion_iter_adaptive=0, erosion_iter_fallback=2,
                                     fatness_threshold=0.2, min_thresh=1.5):
    fatness = compute_fatness_ratio(binary_segment)
    print(f"💡 Fatness ratio: {fatness:.2f}")

    # Common smoothing
    selem = ball(radius)
    opened = opening(binary_segment, selem)
    closed = closing(opened, selem)
    smoothed = median_filter(closed, size=filter_size)

    if fatness < fatness_threshold:
        print("✅ Slim shape detected: using light smoothing")
        processed_segment = smoothed.astype(np.uint8)
    else:
        print("✅ Fat shape detected: applying hybrid thinning strategy")
        dist = distance_transform_edt(smoothed)
        dist_max = dist.max()
        adaptive_thresh = max(min_thresh, 0.2 * dist_max)
        print(f"  Adaptive dist_thresh = {adaptive_thresh:.2f} (min_thresh = {min_thresh}, max dist = {dist_max:.2f})")

        center_channel = dist > adaptive_thresh
        thinned = binary_erosion(center_channel, iterations=erosion_iter_adaptive) if erosion_iter_adaptive > 0 else center_channel

        voxel_ratio = np.count_nonzero(thinned) / max(1, np.count_nonzero(binary_segment))
        print(f"  Post-adaptive voxel ratio: {voxel_ratio:.2f}")

        if voxel_ratio > 0.4:
            print("⚠️ Adaptive thinning not sufficient, fallback to legacy thinning (dist>2 + erosion=2)")
            center_channel_fallback = dist > 2
            thinned_fallback = binary_erosion(center_channel_fallback, iterations=erosion_iter_fallback)
            processed_segment = thinned_fallback.astype(np.uint8)
        else:
            processed_segment = thinned.astype(np.uint8)

    return processed_segment