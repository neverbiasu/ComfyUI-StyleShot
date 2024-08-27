import os
import cv2
import torch
import argparse
import numpy as np

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

model_dir = os.path.join(os.path.dirname(__file__), "prtrained_models")
device = "cuda"


class PipelineLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    "COMBO",
                    ["text_driven", "image_driven", "controlnet", "t2i-adapter"],
                ),
                "base_model_path": (
                    "STRING",
                    {"default": "runwayml/stable-diffusion-v1-5"},
                ),
                "transformer_block_path": (
                    "STRING",
                    {"default": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"},
                ),
                "styleshot_model_path": ("STRING", {"default": "Gaojunyao/StyleShot"}),
                "controlnet_model_path": (
                    "STRING",
                    {"default": "lllyasviel/control_v11f1p_sd15_depth"},
                ),
                "adapter_model_path": (
                    "STRING",
                    {"default": "TencentARC/t2iadapter_depth_sd15v2"},
                ),
                "preprocessor": ("COMBO", ["Lineart", "Contour"]),
            }
        }

    RETURN_TYPES = ("PIPELINE",)
    FUNCTION = "load_pipeline"
    OUTPUT_NODE = True

    CATEGORY = "PIPELINE"

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
        if preprocessor == "Lineart":
            # detector = LineartDetector()
            styleshot_model_path = "Gaojunyao/StyleShot_lineart"
        elif preprocessor == "Contour":
            # detector = SOFT_HEDdetector()
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
            styleshot = StyleShot(
                device, pipe, ip_ckpt, style_aware_encoder_path, transformer_block_path
            )
        if mode == "image_driven":
            unet = UNet2DConditionModel.from_pretrained(
                base_model_path, subfolder="unet", variant="fp16"
            )
            content_fusion_encoder = ControlNetModel.from_unet(unet)

            pipe = StyleContentStableDiffusionControlNetPipeline.from_pretrained(
                base_model_path, variant="fp16", controlnet=content_fusion_encoder
            )
            styleshot = StyleShot(
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

            styleshot = StyleShot(
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

            styleshot = StyleShot(
                device, pipe, ip_ckpt, style_aware_encoder_path, transformer_block_path
            )

        return styleshot


class StyleShot:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("PIPELINE",),
                "mode": (
                    "COMBO",
                    ["text_driven", "image_driven", "controlnet", "t2i-adapter"],
                ),
                "style_image": ("IMAGE",),
                "condition_image": ("IMAGE",),
                "prompt": ("STRING",),
            },
            "optional": {
                "preprocessor": (["Contour", "Lineart"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    OUTPUT_NODE = True

    CATEGORY = "StyleShot"

    def generate(
        self, pipeline, mode, style_image, condition_image, prompt, preprocessor
    ):
        if preprocessor == "Lineart":
            detector = LineartDetector()
        elif preprocessor == "Contour":
            detector = SOFT_HEDdetector()
        else:
            raise ValueError("Invalid preprocessor")
        if mode == "text_driven":
            generation = pipeline.generate(style_image=style_image, prompt=[[prompt]])
        elif mode == "image_driven":
            content_image = cv2.imread(condition_image)
            content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
            content_image = detector(content_image)
            content_image = Image.fromarray(content_image)
            generation = pipeline.generate(
                style_image=style_image, prompt=[[prompt]], content_image=content_image
            )
        elif mode == "controlnet":
            generation = pipeline.generate(
                style_image=style_image, prompt=[[prompt]], image=condition_image
            )
        elif mode == "t2i-adapter":
            generation = pipeline.generate(
                style_image=style_image, prompt=[[prompt]], image=[condition_image]
            )
        return generation[0][0]