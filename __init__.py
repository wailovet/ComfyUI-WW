

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

WEB_DIRECTORY = "./web"

translation_dir = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), "AIGODLIKE-COMFYUI-TRANSLATION", "zh-CN", "Nodes")

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


class AppendString:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start": ("STRING", {"forceInput": False, "multiline": True, "default": ""}),
                "text": ("STRING", {"forceInput": True, "multiline": True, "default": ""}),
                "end": ("STRING", {"forceInput": False, "multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"
    CATEGORY = "utils"

    def execute(self, start, text, end):
        return (start + text + end,)


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

    def execute(self, text, seed):
        np.random.seed(seed)
        # 换行分割
        lines = text.split("\n")
        # 随机选择一行
        line = lines[np.random.randint(0, len(lines))]

        return (line,)


PRE_PREVIEW_IMAGE = []
ALL_PREVIEW_IMAGE = []
WW_PROGRESS_BAR_HOOK = None



prompt_queue = server.PromptServer.instance.prompt_queue
CURRENT_PREVIEW_IMAGE = {

}

def get_current_queue_id():
    current_queue = prompt_queue.get_current_queue()
    queue_running = current_queue[0]
    if queue_running is not None:
        queue_running_id = None
        try:
            queue_running_id = queue_running[0][1]
        except:
            pass
        return queue_running_id
    return None

def ww_hook(value, total, preview_image):
    if WW_PROGRESS_BAR_HOOK is not None:
        WW_PROGRESS_BAR_HOOK(value, total, preview_image)
    try:
        global ALL_PREVIEW_IMAGE
        global PRE_PREVIEW_IMAGE
        global CURRENT_PREVIEW_IMAGE
        print("value:", value, "total:", total, "preview_image:", preview_image)
        if preview_image is None:
            return
        imgformat, pliimg, size = preview_image
        if value <= 1:
            PRE_PREVIEW_IMAGE = []
        PRE_PREVIEW_IMAGE.append(pliimg)


        if value == total:
            ALL_PREVIEW_IMAGE.extend(PRE_PREVIEW_IMAGE)
            PRE_PREVIEW_IMAGE = []


        
        
        queue_running_id = get_current_queue_id()
        if queue_running_id is not None:
            if CURRENT_PREVIEW_IMAGE.get(queue_running_id, None) is None:
                CURRENT_PREVIEW_IMAGE[queue_running_id] = []
            CURRENT_PREVIEW_IMAGE[queue_running_id].append(pliimg)
    except Exception as e:
        print("ww_hook:",e)
 
# comfy.utils.set_progress_bar_global_hook(ww_hook)


# 延时启动
import threading
import time


def start_server():
    while True:
        time.sleep(1)
        if comfy.utils.PROGRESS_BAR_HOOK is not None:
            global WW_PROGRESS_BAR_HOOK
            WW_PROGRESS_BAR_HOOK = comfy.utils.PROGRESS_BAR_HOOK
            comfy.utils.set_progress_bar_global_hook(ww_hook)
            break


threading.Thread(target=start_server).start()


class AccumulationPreviewImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "utils"

    def execute(self, image):
        global ALL_PREVIEW_IMAGE
        result = ALL_PREVIEW_IMAGE.copy()

        max_w = 0
        max_h = 0
        for i in range(len(result)):
            size = result[i].size
            max_w = max(max_w, size[0])
            max_h = max(max_h, size[1])

        for i in range(len(result)):
            result[i] = utils.Utils.pil2tensor(result[i].resize((max_w, max_h)))[0]

        result.append(image[0])


        result = utils.Utils.list_tensor2tensor(result)
        # print("result:", result)
        return (result,)

class CurrentPreviewImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "utils"

    def execute(self, image):
        global CURRENT_PREVIEW_IMAGE

        queue_running_id = get_current_queue_id()



        result = CURRENT_PREVIEW_IMAGE.get(queue_running_id, [])

        max_w = 0
        max_h = 0
        for i in range(len(result)):
            size = result[i].size
            max_w = max(max_w, size[0])
            max_h = max(max_h, size[1])

        for i in range(len(result)):
            result[i] = utils.Utils.pil2tensor(result[i].resize((max_w, max_h)))[0]

        result.append(image[0])


        result = utils.Utils.list_tensor2tensor(result)


        
        if queue_running_id is not None:
            del CURRENT_PREVIEW_IMAGE[queue_running_id]

        # print("result:", result)
        return (result,)



class WW_PreviewTextNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {        
                "text": ("STRING", {"forceInput": True}),     
                },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
            }

    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True
    FUNCTION = "preview_text"

    CATEGORY = "utils"

    def preview_text(self, text, prompt=None, extra_pnginfo=None):
        return {"ui": {"string": [text,]}, "result": (text,)}

NODE_CLASS_MAPPINGS = {
    "WW_ImageResize": ImageResize,
    "WW_AppendString": AppendString,
    "WW_RandString": RandString,
    "WW_AccumulationPreviewImages": AccumulationPreviewImages,
    "WW_CurrentPreviewImages": CurrentPreviewImages,
    "WW_PreviewTextNode": WW_PreviewTextNode,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "WW_ImageResize": "WW_ImageResize",
    "WW_RandString": "WW_RandString",
    "WW_AppendString": "WW_AppendString",
    "WW_AccumulationPreviewImages": "WW_AccumulationPreviewImages",
    "WW_CurrentPreviewImages": "WW_CurrentPreviewImages",
    "WW_PreviewTextNode": "WW_PreviewTextNode",
}


routes = server.PromptServer.instance.routes


from aiohttp import web


@routes.post('/extention/clean_all_preview')
def clean_all_preview(request):
    print('clean_all_preview')
    global ALL_PREVIEW_IMAGE
    global PRE_PREVIEW_IMAGE
    PRE_PREVIEW_IMAGE = []
    ALL_PREVIEW_IMAGE = []
    return web.json_response({'data': 'ok'})





class ColorMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE",),
                "image_target": ("IMAGE",),
                "method": (
            [   
                'mkl',
                'hm', 
                'reinhard', 
                'mvgd', 
                'hm-mvgd-hm', 
                'hm-mkl-hm',
            ], {
               "default": 'mkl'
            }),
                
            },
        }
    
    CATEGORY = "utils"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "colormatch"
    DESCRIPTION = """
color-matcher enables color transfer across images which comes in handy for automatic  
color-grading of photographs, paintings and film sequences as well as light-field  
and stopmotion corrections.  

The methods behind the mappings are based on the approach from Reinhard et al.,  
the Monge-Kantorovich Linearization (MKL) as proposed by Pitie et al. and our analytical solution  
to a Multi-Variate Gaussian Distribution (MVGD) transfer in conjunction with classical histogram   
matching. As shown below our HM-MVGD-HM compound outperforms existing methods.   
https://github.com/hahnec/color-matcher/

"""
    
    def colormatch(self, image_ref, image_target, method):
        try:
            from color_matcher import ColorMatcher
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "color-matcher"])
            from color_matcher import ColorMatcher
            # raise Exception("Can't import color-matcher, did you install requirements.txt? Manual install: pip install color-matcher")
        cm = ColorMatcher()
        image_ref = image_ref.cpu()
        image_target = image_target.cpu()
        batch_size = image_target.size(0)
        out = []
        images_target = image_target.squeeze()
        images_ref = image_ref.squeeze()

        image_ref_np = images_ref.numpy()
        images_target_np = images_target.numpy()

        if image_ref.size(0) > 1 and image_ref.size(0) != batch_size:
            raise ValueError("ColorMatch: Use either single reference image or a matching batch of reference images.")

        for i in range(batch_size):
            image_target_np = images_target_np if batch_size == 1 else images_target[i].numpy()
            image_ref_np_i = image_ref_np if image_ref.size(0) == 1 else images_ref[i].numpy()
            try:
                image_result = cm.transfer(src=image_target_np, ref=image_ref_np_i, method=method)
            except BaseException as e:
                print(f"Error occurred during transfer: {e}")
                break
            out.append(torch.from_numpy(image_result))
        return (torch.stack(out, dim=0).to(torch.float32), )
    

NODE_CLASS_MAPPINGS["ColorMatch"] = ColorMatch
NODE_DISPLAY_NAME_MAPPINGS["ColorMatch"] = "ColorMatch"

import importlib
class seamlessClone:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "src": ("IMAGE",),
                "dst": ("IMAGE",),
                "mask": ("MASK",),
                "method": (['normal_clone','mixed_clone',], { "default": 'normal_clone' }),  
            },
        }
    
    CATEGORY = "utils"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "seamlessclone" 

    def seamlessclone(self, src, dst, mask, method):
        importlib.reload(utils)
        return utils.Utils.seamlessclone(src, dst, mask, method)
    

NODE_CLASS_MAPPINGS["seamlessClone"] = seamlessClone
NODE_DISPLAY_NAME_MAPPINGS["seamlessClone"] = "seamlessClone"


class mask_edge_blur:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "radius": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1, }),
            },
        }
    
    CATEGORY = "utils"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "mask_edge_blur" 

    def mask_edge_blur(self, mask, radius):
        importlib.reload(utils)
        return utils.Utils.mask_edge_blur(mask, radius)

NODE_CLASS_MAPPINGS["mask_edge_blur"] = mask_edge_blur
NODE_DISPLAY_NAME_MAPPINGS["mask_edge_blur"] = "mask_edge_blur"