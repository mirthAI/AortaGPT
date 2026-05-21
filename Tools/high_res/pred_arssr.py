from model import ArSSR
from ArSSR_inference import super_resolve
import torch
from pathlib import Path
import SimpleITK as sitk

# initial predictor

import numpy as np
import matplotlib.pyplot as plt

def visualize_sitk_image(img: sitk.Image,
                         title: str = "Super-resolved image",
                         pct_clip=(1, 99)) -> plt.Figure:
    """
    Show orthogonal mid-slices (axial, coronal, sagittal) from a SimpleITK image.
    Returns a matplotlib Figure you can hand to Gradio or save.

    Args:
        img: SimpleITK.Image (2D or 3D)
        title: Figure title
        pct_clip: intensity clipping percentiles for display (tuple)
    """
    # Convert to numpy in (z, y, x) order
    arr = sitk.GetArrayFromImage(img)  # (Z, Y, X) for 3D; (Y, X) for 2D

    # If 2D, just show it
    if arr.ndim == 2:
        a = arr.astype(np.float32)
        lo, hi = np.percentile(a, pct_clip)
        a = np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)
        fig = plt.figure(figsize=(4, 4), dpi=120)
        ax = plt.subplot(1,1,1)
        ax.imshow(a, cmap="gray", origin="lower")
        ax.set_title(title)
        ax.axis("off")
        fig.tight_layout()
        return fig

    # 3D: pick mid slices
    z, y, x = arr.shape
    zi, yi, xi = z // 2, y // 2, x // 2

    # Robust normalization
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a, pct_clip)
    a = np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)

    axial   = a[zi, :, :]             # (Y, X)
    coronal = a[:, yi, :]             # (Z, X)
    sagittal= a[:, :, xi]             # (Z, Y)

    fig = plt.figure(figsize=(10, 3.5), dpi=120)
    axs = [plt.subplot(1,3,i+1) for i in range(3)]
    imgs = [axial, np.flipud(coronal), np.flipud(sagittal.T)]  # light orientation polish
    titles = [f"Axial z={zi}", f"Coronal y={yi}", f"Sagittal x={xi}"]

    for ax, im, t in zip(axs, imgs, titles):
        ax.imshow(im, cmap="gray", origin="lower", interpolation="nearest")
        ax.set_title(t, fontsize=10)
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def load_state_dict_flexible(model, path, device):
    raw_state = torch.load(path, map_location=device)
    new_state = {k.split("_orig_mod.")[-1].split("module.")[-1]: v for k, v in raw_state.items()}
    model.load_state_dict(new_state)
    return model

if __name__ == "__main__":
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("Loading model...")

    try:
        model = ArSSR(
            encoder_name="RDN",
            feature_dim=128,
            decoder_depth=4,
            decoder_width=256
            ).to(DEVICE)

        # model = torch.compile(model)

        # load model
        model_dir = "/home/chojnowski.h/weishao/chojnowski.h/ArSSR/model/model_param_anisotropic_2-4x_3000.pkl"
        load_state_dict_flexible(model, model_dir, DEVICE)
    except:
        print("Model loading failed")
        exit()

    print("Model loaded")
    print("----------------------------------")
    print("Loading image...")

    try:
    # method 1: only one file
        img_path = Path("/blue/weishao/gates/AortaAgent/subject003_CTA.nii.gz")
        # img_path = Path("data/infrenence/subject057_CTA_0.25x.nii.gz")
        out_path = Path("data/infrenence/subject057_CTA_0.25-1.0x.nii.gz")
        # read image
        image = sitk.ReadImage(str(img_path))
    except Exception as e:
        print(f"Image loading failed: {e}")
        raise SystemExit(1)

    print("Image loaded")
    print("----------------------------------")
    print("Inferring...")
    try:

        # inference
        # this is a simpleITK image
        """
        This function prompts the model to super resolve a given 3D volume
        anisotropically at any arbitrary scale though it will work best [2, 4] since 
        it was trained from U~(2,4) upscaling factors.

        The patch size and stride are sampling parameters.
        I would keep the patch size as is but the lower the the better the results.
        I would stay between 32 and 64 with a hard limit of 64.

        The goal is to get the spacing of the image to 1mm on all axes.
        So simply take the original depth size in mm and make that the scaling factor.
        """
        super_resolved_image = super_resolve(model=model, image=image, scaling_factor=2.0, patch_size=64, stride=64)
        # super_resolved_image = super_resolve_batched(model=model, image=image, scaling_factor=4.0, patch_size=64, stride=32)
    except:
        print("Inference failed")
        exit()



 