import os
import cv2
import sys
import torch
import numpy as np

project_root = os.path.abspath(os.path.dirname(__file__))

if project_root not in sys.path:
    sys.path.append(project_root)

from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionAdapterPipeline,
    StableDiffusionPipeline,
    ControlNetModel,
    UNet2DConditionModel,
    T2IAdapter,
)
from annotator.hed import SOFT_HEDdetector
from annotator.lineart import LineartDetector
from huggingface_hub import snapshot_download
from ip_adapter import StyleShot, StyleContentStableDiffusionControlNetPipeline

model_dir = os.path.join(project_root, "pretrained_models")
device = "cuda"


def pil_to_tensor(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.convert("RGB")
    img = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0)  # [1, H, W, 3]


def comfyui_tensor_to_pil(img_tensor):
    img = img_tensor[0].cpu().numpy()  # [H, W, C]
    img = (img * 255).clip(0, 255).astype('uint8')
    return Image.fromarray(img)


class StyleShotApply:
    def __init__(self):
        self._pipeline_cache = None
        self._pipeline_args = None
        self.styleshot = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    ["text_driven", "image_driven", "controlnet", "t2i-adapter"],
                    {"default": "text_driven"},
                ),
                "style_image": ("IMAGE", {"default": None}),
            },
            "optional": {
                "condition_image": ("IMAGE", {"default": None}),
                "prompt": ("STRING", {"default": ""}),
                "preprocessor": (["Contour", "Lineart"], {"default": "Contour"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    OUTPUT_NODE = True

    CATEGORY = "StyleShot"

    def load_pipeline(
        self,
        mode,
        base_model_path,
        transformer_block_path,
        styleshot_model_path,
        controlnet_model_path,
        adapter_model_path,
        preprocessor,
    ):
        args_tuple = (
            mode,
            base_model_path,
            transformer_block_path,
            styleshot_model_path,
            controlnet_model_path,
            adapter_model_path,
            preprocessor,
        )
        if self._pipeline_cache is not None and self._pipeline_args == args_tuple:
            self.styleshot = self._pipeline_cache
            return

        if preprocessor == "Lineart":
            styleshot_model_path = "Gaojunyao/StyleShot_lineart"
        elif preprocessor == "Contour":
            styleshot_model_path = "Gaojunyao/StyleShot"
        else:
            raise ValueError("Invalid preprocessor")

        if not os.path.isdir(base_model_path):
            base_model_path = snapshot_download(
                base_model_path,
                allow_patterns=["*fp16.safetensors", "*.json", "*yaml", "*.txt"],
                local_dir=os.path.join(model_dir, base_model_path.split("/")[-1]),
            )
            print(f"Downloaded model to {base_model_path}")
        if not os.path.isdir(transformer_block_path):
            transformer_block_path = snapshot_download(
                transformer_block_path,
                # allow_patterns=["*.safetensors","*.json"],
                ignore_patterns=["open_clip*", "*.bin"],
                local_dir=os.path.join(
                    model_dir, transformer_block_path.split("/")[-1]
                ),
            )
            print(f"Downloaded model to {transformer_block_path}")
        if not os.path.isdir(styleshot_model_path):
            styleshot_model_path = snapshot_download(
                styleshot_model_path,
                local_dir=os.path.join(model_dir, styleshot_model_path.split("/")[-1]),
            )
            print(f"Downloaded model to {styleshot_model_path}")

        ip_ckpt = os.path.join(styleshot_model_path, "pretrained_weight/ip.bin")
        style_aware_encoder_path = os.path.join(
            styleshot_model_path, "pretrained_weight/style_aware_encoder.bin"
        )

        if mode == "text_driven":
            pipe = StableDiffusionPipeline.from_pretrained(
                base_model_path, variant="fp16"
            )
            self.styleshot = StyleShot(
                device, pipe, ip_ckpt, style_aware_encoder_path, transformer_block_path
            )

        if mode == "image_driven":
            unet = UNet2DConditionModel.from_pretrained(
                base_model_path, subfolder="unet", variant="fp16"
            )
            content_fusion_encoder = ControlNetModel.from_unet(unet)

            pipe = StyleContentStableDiffusionControlNetPipeline.from_pretrained(
                base_model_path,
                variant="fp16",
                controlnet=content_fusion_encoder,
            )
            self.styleshot = StyleShot(
                device, pipe, ip_ckpt, style_aware_encoder_path, transformer_block_path
            )

        if mode == "t2i-adapter":
            if not os.path.isdir(adapter_model_path):
                adapter_model_path = snapshot_download(
                    adapter_model_path,
                    ignore_patterns=["*.png"],
                    local_dir=os.path.join(
                        model_dir, adapter_model_path.split("/")[-1]
                    ),
                )
                print(f"Downloaded model to {adapter_model_path}")
            adapter = T2IAdapter.from_pretrained(
                adapter_model_path, torch_dtype=torch.float16
            )
            pipe = StableDiffusionAdapterPipeline.from_pretrained(
                base_model_path, adapter=adapter, variant="fp16"
            )

            self.styleshot = StyleShot(
                device, pipe, ip_ckpt, style_aware_encoder_path, transformer_block_path
            )

        if mode == "controlnet":
            if not os.path.isdir(controlnet_model_path):
                controlnet_model_path = snapshot_download(
                    controlnet_model_path,
                    allow_patterns=["*.json", "*.fp16.safetensors"],
                    local_dir=os.path.join(
                        model_dir, controlnet_model_path.split("/")[-1]
                    ),
                )
                print(f"Downloaded model to {controlnet_model_path}")
            controlnet = ControlNetModel.from_pretrained(
                controlnet_model_path, torch_dtype=torch.float16
            )
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                base_model_path, controlnet=controlnet, variant="fp16"
            )

            self.styleshot = StyleShot(
                device, pipe, ip_ckpt, style_aware_encoder_path, transformer_block_path
            )
        self._pipeline_cache = self.styleshot
        self._pipeline_args = args_tuple

    def generate(
        self,
        mode,
        style_image,
        condition_image,
        prompt,
        preprocessor,
    ):
        print("Loading pipeline...")
        base_model_path = "ckpt/sd15"
        transformer_block_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        styleshot_model_path = "Gaojunyao/StyleShot"
        controlnet_model_path = "lllyasviel/control_v11f1p_sd15_depth"
        adapter_model_path = "TencentARC/t2iadapter_depth_sd15v2"
        self.load_pipeline(
            mode,
            base_model_path,
            transformer_block_path,
            styleshot_model_path,
            controlnet_model_path,
            adapter_model_path,
            preprocessor,
        )
        pipeline = self.styleshot
        if preprocessor == "Lineart":
            detector = LineartDetector()
        elif preprocessor == "Contour":
            detector = SOFT_HEDdetector()
        else:
            raise ValueError("Invalid preprocessor")

        style_pil = (
            comfyui_tensor_to_pil(style_image)
            if isinstance(style_image, torch.Tensor)
            else style_image
        )
        cond_pil = (
            comfyui_tensor_to_pil(condition_image)
            if isinstance(condition_image, torch.Tensor)
            else condition_image
        )

        if mode == "text_driven":
            generation = pipeline.generate(style_image=style_pil, prompt=[[prompt]])
        elif mode == "image_driven":
            content_image = np.array(cond_pil)
            content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
            content_image = detector(content_image)
            content_image = Image.fromarray(content_image)
            generation = pipeline.generate(
                style_image=style_pil, prompt=[[prompt]], content_image=content_image
            )
        elif mode == "controlnet":
            generation = pipeline.generate(
                style_image=style_pil, prompt=[[prompt]], image=cond_pil
            )
        elif mode == "t2i-adapter":
            generation = pipeline.generate(
                style_image=style_pil, prompt=[[prompt]], image=[cond_pil]
            )
        else:
            raise ValueError("Invalid mode")

        gen_pil = (
            generation[0][0]
            if isinstance(generation[0][0], Image.Image)
            else Image.fromarray(generation[0][0])
        )
        result_tensor = pil_to_tensor(gen_pil)

        return result_tensor
