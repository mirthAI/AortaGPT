from Tools.nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import nibabel as nib
import numpy as np
import torch
from Tools.nnUNet.nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from pathlib import Path



# class AortaSegmentation:
#     def __init__(self, model_path, device="cuda"):
#         self.predictor = self.initial_model(model_path, device)
#         self.device = device

#     def initial_model(self, model_path, device):
#         model = nnUNetPredictor(
#         tile_step_size=0.5,
#         use_gaussian=True,
#         use_mirroring=False,
#         perform_everything_on_device=True,
#         device=torch.device('cuda'),
#         verbose=False
#     )
#         model.initialize_from_trained_model_folder(
#             model_training_output_dir=model_path,
#             use_folds="all", 
#             checkpoint_name= 'checkpoint_final.pth'
#         )
#         # model = model.to(device)
#         # model.eval()
#         return model

#     def segment_image(self, image_path):
#         image, props = SimpleITKIO().read_images([image_path])

#         with torch.no_grad():
#             output = self.predictor.predict_single_npy_array(
#                 input_image=image,
#                 image_properties=props
#             )
          
#         # output = np.transpose(output, (2, 1, 0))
#         # output = np.ascontiguousarray(output[:, ::-1, ::-1])

#         return output

# class AortaSegmentation:
#     def __init__(self, model_path, device="cuda", use_amp=True, amp_dtype="fp16"):
#         self.device = torch.device(device)  # "cuda" or "cpu"
#         self.use_amp = bool(use_amp) and (self.device.type == "cuda")
#         self.amp_dtype = amp_dtype  # "fp16" or "bf16"
#         self.predictor = self.initial_model(model_path)

#     def _autocast_ctx(self):
#         if not self.use_amp:
#             return torch.autocast(device_type="cpu", enabled=False)

#         dtype = torch.float16
#         if str(self.amp_dtype).lower() in ("bf16", "bfloat16"):
#             dtype = torch.bfloat16

#         return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)

#     def initial_model(self, model_path):
#         predictor = nnUNetPredictor(
#             tile_step_size=0.5,
#             use_gaussian=True,
#             use_mirroring=False,
#             perform_everything_on_device=True,
#             device=self.device,          # ✅ use chosen device
#             verbose=False
#         )

#         predictor.initialize_from_trained_model_folder(
#             model_training_output_dir=model_path,
#             use_folds="all",
#             checkpoint_name="checkpoint_final.pth"
#         )
#         return predictor

#     def segment_image(self, image_path):
#         image, props = SimpleITKIO().read_images([image_path])

#         # ✅ inference_mode is best for inference
#         with torch.inference_mode():
#             # ✅ optional AMP for speed on GPU
#             with self._autocast_ctx():
#                 output = self.predictor.predict_single_npy_array(
#                     input_image=image,
#                     image_properties=props
#                 )

#         return output

class AortaSegmentation:
    def __init__(self, model_path, device="cuda", use_compile=True):
        self.device = device
        self.use_compile = bool(use_compile)
        self.predictor = self.initial_model(model_path, device)

    def initial_model(self, model_path, device):
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=True,
            device=torch.device(device),
            verbose=False
        )
        predictor.initialize_from_trained_model_folder(
            model_training_output_dir=model_path,
            use_folds="all",
            checkpoint_name="checkpoint_final.pth"
        )

        # ✅ Try to compile the underlying torch module (name differs by nnUNet version)
        if self.use_compile:
            net = None
            for attr in ("network", "model", "net"):
                if hasattr(predictor, attr):
                    cand = getattr(predictor, attr)
                    if isinstance(cand, torch.nn.Module):
                        net = cand
                        break

            if net is not None:
                net.eval()
                try:
                    compiled = torch.compile(net, mode="reduce-overhead")
                    setattr(predictor, attr, compiled)
                    print(f"[SEG] torch.compile enabled on predictor.{attr}")
                except Exception as e:
                    print(f"[SEG] torch.compile failed, falling back to eager: {e}")
            else:
                print("[SEG] Could not find torch.nn.Module inside predictor to compile")

        return predictor

    def segment_image(self, image_path):
        image, props = SimpleITKIO().read_images([image_path])
        with torch.inference_mode():
            output = self.predictor.predict_single_npy_array(
                input_image=image,
                image_properties=props
            )
        return output