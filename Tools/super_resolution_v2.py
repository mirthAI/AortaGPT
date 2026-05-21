from Tools.high_res.model import ArSSR
from Tools.high_res.ArSSR_inference import super_resolve
import torch
import SimpleITK as sitk


class SuperResolution:
    def __init__(
        self,
        model_path,
        device="cuda",
        use_compile=True,
        use_amp=True,
        amp_dtype="fp16",   # "fp16" or "bf16"
    ):
        self.device = device
        self.use_amp = bool(use_amp)
        self.amp_dtype = amp_dtype

        self.model = self.initial_model(model_path, device)

        # ✅ Always eval for inference
        self.model.eval()

        # ✅ torch.compile once (kernel fusion)
        self.use_compile = bool(use_compile)
        if self.use_compile:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("[SR] torch.compile enabled")
            except Exception as e:
                print(f"[SR] torch.compile failed, falling back to eager: {e}")
                self.use_compile = False

    def initial_model(self, model_path, device):
        model = ArSSR(
            encoder_name="RDN",
            feature_dim=128,
            decoder_depth=4,
            decoder_width=256
        ).to(device)

        model = self.load_state_dict_flexible(model, model_path, device)
        return model

    def load_state_dict_flexible(self, model, path, device):
        raw_state = torch.load(path, map_location=device)
        new_state = {k.split("_orig_mod.")[-1].split("module.")[-1]: v for k, v in raw_state.items()}
        model.load_state_dict(new_state)
        return model

    def _autocast_ctx(self):
        """Small helper so enhance_resolution stays clean."""
        if not (self.use_amp and str(self.device).startswith("cuda")):
            return torch.autocast("cuda", enabled=False)

        dtype = torch.float16 if self.amp_dtype == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def enhance_resolution(self, image_path, scale=1):
        image = sitk.ReadImage(str(image_path))

        # ✅ best practice for inference
        with torch.inference_mode():
            with self._autocast_ctx():
                output = super_resolve(
                    model=self.model,
                    image=image,
                    scaling_factor=scale,
                    patch_size=64,
                    stride=56
                )

        return output

