import monai.transforms as mtf
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from Tools.CIS_UNET.CIS_UNet import CIS_UNet


class AortaSegmentation:
    def __init__(self, model_path, device="cuda"):
        self.model = self.initial_model(model_path, device)
        self.device = device

    def initial_model(self, model_path, device):
        model = CIS_UNet(
            spatial_dims=3,
            in_channels=1,
            num_classes=24,
            encoder_channels=[64, 128, 256, 512],
            feature_size=48,
        )

        state_dict = torch.load(model_path)

        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        model.eval()
        return model

    def segment_image(self, image_path):
        image_data = self.pre_process_image(image_path)

        with torch.no_grad():
            output = sliding_window_inference(
                inputs=image_data,
                roi_size=(128, 128, 128),
                sw_batch_size=4,
                predictor=self.model,
            )
            output = torch.argmax(output, dim=1).squeeze(0)

        output = output.cpu().numpy()
        output = np.transpose(output, (2, 1, 0))
        output = np.ascontiguousarray(output[:, ::-1, ::-1])

        return output

    def pre_process_image(self, image_path):
        pre_process = mtf.Compose(
            [
                mtf.LoadImage(image_only=True, ensure_channel_first=True),
                mtf.ScaleIntensityRange(
                    a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
                ),
                mtf.CropForeground(source_key=None, allow_smaller=True),
                mtf.Orientation(axcodes="RAS"),
                mtf.Spacing(pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
                mtf.ToTensor(dtype=torch.float32, device=self.device, track_meta=False),
            ]
        )

        return pre_process(image_path).unsqueeze(0)
