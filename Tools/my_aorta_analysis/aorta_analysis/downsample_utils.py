import numpy as np
import SimpleITK as sitk

def downsample_centerline_and_segmentation(curve_points_origin, tangents_origin, segmentation, downsample_factor=2, origin=None, spacing=None):
    if isinstance(segmentation, np.ndarray):
        segmentation_sitk = sitk.GetImageFromArray(segmentation)
        segmentation_sitk.SetSpacing(spacing or (1.0, 1.0, 1.0))
        segmentation_sitk.SetOrigin(origin or (0.0, 0.0, 0.0))
    else:
        segmentation_sitk = segmentation

    original_spacing = segmentation_sitk.GetSpacing()
    original_size = segmentation_sitk.GetSize()
    new_size = [int(s / downsample_factor) for s in original_size]
    new_spacing = [sp * downsample_factor for sp in original_spacing]

    resample = sitk.ResampleImageFilter()
    resample.SetSize(new_size)
    resample.SetOutputSpacing(new_spacing)
    resample.SetOutputOrigin(segmentation_sitk.GetOrigin())
    resample.SetOutputDirection(segmentation_sitk.GetDirection())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    resample.SetTransform(sitk.Transform())

    downsampled_seg_sitk = resample.Execute(segmentation_sitk)
    downsampled_seg_array = sitk.GetArrayFromImage(downsampled_seg_sitk)

    downsampled_curve_points = curve_points_origin / downsample_factor
    downsampled_tangents = tangents_origin

    return downsampled_curve_points, downsampled_tangents, downsampled_seg_array, downsampled_seg_sitk
