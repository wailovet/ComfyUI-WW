
import os
import subprocess
import sys

import cv2
import numpy as np
import folder_paths
import base64
from PIL import Image, ImageFilter, ImageEnhance
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

    
    def pil2cv(image):
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    def cv2pil(image):
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    


    def list_tensor2tensor(data):
        result_tensor = torch.stack(data)
        return result_tensor


    def loadImage(path):
        img = Image.open(path)
        img = img.convert("RGB")
        return img
    




    
    def seamlessclone(src:Image, dst, mask, method):
        try:
            import cv2 
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
            import cv2
            # raise Exception("Can't import cv2, did you install requirements.txt? Manual install: pip install opencv-python")
        

        src = Utils.tensor2pil(src)
        width, height = src.size 
        src = Utils.pil2cv(src)

        dst = Utils.tensor2pil(dst)
        dst = dst.resize((width, height)) 
        dst = Utils.pil2cv(dst)

 
        src_mask = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        src_mask = Utils.tensor2pil(src_mask)
        src_mask = src_mask.resize((width, height)) 
        f_src_mask = Utils.pil2cv(src_mask.copy().filter(ImageFilter.GaussianBlur(15)))
        src_mask = Utils.pil2cv(src_mask)    


        left, top , right, bottom = 0, 0, src_mask.shape[1], src_mask.shape[0]
        for i in range(src_mask.shape[0]):
            for j in range(src_mask.shape[1]):
                if src_mask[i][j][0] > 0:
                    top = i
                    break
            if top != 0:
                break
        for i in range(src_mask.shape[0]-1, -1, -1):
            for j in range(src_mask.shape[1]):
                if src_mask[i][j][0] > 0:
                    bottom = i
                    break
            if bottom != src_mask.shape[0]:
                break
        for j in range(src_mask.shape[1]):
            for i in range(src_mask.shape[0]):
                if src_mask[i][j][0] > 0:
                    left = j
                    break
            if left != 0:
                break

        for j in range(src_mask.shape[1]-1, -1, -1):
            for i in range(src_mask.shape[0]):
                if src_mask[i][j][0] > 0:
                    right = j
                    break
            if right != src_mask.shape[1]:
                break

        center = (left + right) // 2, (top + bottom) // 2

        if method == 'normal_clone':
            output = cv2.seamlessClone(dst, src, f_src_mask, center, cv2.NORMAL_CLONE)
        elif method == 'mixed_clone':
            output = cv2.seamlessClone(dst, src, f_src_mask, center, cv2.MIXED_CLONE) 
        else:
            raise ValueError("Unknown method")
        
        final = Utils.pil2tensor(Utils.cv2pil(output))
        
 
 
        return (final, )