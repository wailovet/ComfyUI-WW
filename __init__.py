

import hashlib
import json
import os
import shutil
import time
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import folder_paths
import server
from math import sqrt, ceil
from typing import cast
import torchvision.transforms.functional as TF
from .utils import Utils
from PIL import Image, ImageSequence
import sys
import subprocess
from PIL.PngImagePlugin import PngInfo
from typing import List, Dict, Tuple
from nodes import MAX_RESOLUTION, SaveImage
import comfy.utils
import torch.nn.functional as F
import shutil

translation_dir = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), "AIGODLIKE-COMFYUI-TRANSLATION")

translation_config = os.path.join(os.path.dirname(__file__), "ComfyUI-WW.json")
if os.path.exists(translation_dir):
    shutil.copy(translation_config, translation_dir)


def p(image):
    return image.permute([0, 3, 1, 2])


def pb(image):
    return image.permute([0, 2, 3, 1])


class ImageResize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "interpolation": (["nearest", "bilinear", "bicubic", "area", "nearest-exact", "lanczos"],),
                "keep_proportion": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "width", "height",)
    FUNCTION = "execute"
    CATEGORY = "image"

    def execute(self, image, width, height, keep_proportion, interpolation="nearest"):
        if keep_proportion is True:
            _, oh, ow, _ = image.shape
            width = ow if width == 0 else width
            height = oh if height == 0 else height
            ratio = min(width / ow, height / oh)
            width = round(ow * ratio)
            height = round(oh * ratio)

        width = width // 64 * 64
        height = height // 64 * 64

        outputs = p(image)
        if interpolation == "lanczos":
            outputs = comfy.utils.lanczos(outputs, width, height)
        else:
            outputs = F.interpolate(outputs, size=(
                height, width), mode=interpolation)
        outputs = pb(outputs)

        return (outputs, outputs.shape[2], outputs.shape[1],)


class RandString:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": False, "multiline": True, "default": ""}),
                "seed": ("INT", {"default": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"
    CATEGORY = "utils"

    def execute(self, text):
        # 换行分割
        lines = text.split("\n")
        # 随机选择一行
        line = lines[np.random.randint(0, len(lines))]

        return (line,)


NODE_CLASS_MAPPINGS = {
    "WW_ImageResize": ImageResize,
    "WW_RandString": RandString,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "WW_ImageResize": "WW_ImageResize",
    "WW_RandString": "WW_RandString",
}
