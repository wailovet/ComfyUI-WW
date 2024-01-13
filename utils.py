
import os
import subprocess
import sys

import numpy as np
import folder_paths
import base64
from PIL import Image
import io
import torch

class Utils:

    def check_frames_path(frames_path):
        
        if frames_path == "" or frames_path.startswith(".") or frames_path.startswith("/") or frames_path.endswith("/") or frames_path.endswith("\\"):
            return "frames_path不能为空"
        
        frames_path = os.path.join(
            folder_paths.get_output_directory(), frames_path)
        
        if frames_path == folder_paths.get_output_directory():
            return "frames_path不能为output目录"
        
        return ""



    def base64_to_pil_image(base64_str):
        if base64_str is None:
            return None
        if len(base64_str) == 0:
            return None
        if type(base64_str) not in [str, bytes]:
            return None
        if base64_str.startswith("data:image/png;base64,"):
            base64_str = base64_str.split(",")[-1]
        base64_str = base64_str.encode("utf-8")
        base64_str = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(base64_str))


    def pil_image_to_base64(pil_image):
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())
        img_str = str(img_str, encoding="utf-8")
        return f"data:image/png;base64,{img_str}"


    def listdir_png(path):
        try:
            files = os.listdir(path)
            new_files = []
            for file in files:
                if file.endswith(".png"):
                    new_files.append(file)
            files = new_files
            files.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
            return files
        except Exception as e:
            return []
        


        

    def tensor2pil(image):
        return Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


    # Convert PIL to Tensor
    def pil2tensor(image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    
    def loadImage(path):
        img = Image.open(path)
        img = img.convert("RGB")
        return img