from datetime import datetime
import torch
import numpy as np
from pathlib import Path
import os
from PIL import Image, ImageOps

from comfy.model_management import get_torch_device
DEVICE = get_torch_device()


#if img length is not a multiple of 8, then return the length divided by 8
#如果图片长度超过8的倍数,则返回加一后的8的倍数
def fitlength(x) -> int:
    if x % 8 == 0:
        return x
    return int(x // 8 + 1) * 8

# pad image
# 填充图片
def padimage(img):
    # w, h are original image size
    # w, h 是原始图片的大小
    w, h = img.size

    #x, y are padded image size
    # x, y 是填充图片的大小
    x = fitlength(w)
    y = fitlength(h)

    if x!= w or y!= h:
        bgimg = Image.new("RGB", (x, y), (0, 0, 0))
        bgimg.paste(img, (0, 0, w, h))
        return bgimg    
    return img

# pad image
# 填充遮罩
def padmask(img):
    # w, h are original image size
    # w, h 是原始图片的大小
    w, h = img.size

    #x, y are padded image size
    # x, y 是填充图片的大小
    x = fitlength(w)
    y = fitlength(h)

    if x!= w or y!= h:
        bgimg = Image.new("L", (x, y), 0)
        bgimg.paste(img, (0, 0, w, h))
        return bgimg    
    return img

# crop image
# 裁剪图片
def cropimage(img, x, y):
    return img.crop((0, 0, x, y))

# Convert PIL to Tensor
# 图片转张量
def pil2tensor(image, device=DEVICE):
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        raise Exception("Input image should be either PIL Image!")

    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))  # chw
        print(f"Prepare the imput images")
    elif img.ndim == 2:
        img = img[np.newaxis, ...]
        print(f"Prepare the imput masks")

    assert img.ndim == 3

    try:
        img = img.astype(np.float32) / 255
    except:
        img = img.astype(np.float16) / 255
    
    out_image = torch.from_numpy(img).unsqueeze(0).to(device)
    return out_image

# Tensor to PIL
# 张量转图片
def tensor2pil(image):
    i = 255. * image.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    return img

def tensor2pil2(image):
    # 获取图像的通道数
    n_channels = image.shape[1] if len(image.shape) == 4 else image.shape[-1]
    print(f"n_channels: {n_channels}")
    
    # 将浮点型Tensor数据转换为[0, 255]范围内的uint8数组
    i = (255. * image.cpu().numpy()).astype(np.uint8)

    # 根据通道数选择PIL图像模式
    mode = 'RGB' if n_channels == 3 else ('RGBA' if n_channels == 4 else "未知格式")
    
    img = Image.fromarray(i, mode=mode)
    
    return img


# pil to comfy
# 图片转comfy格式 (i, 3, w, h) -> (i, h, w, 3)
def pil2comfy(img):
    img = ImageOps.exif_transpose(img)
    image = img.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    return image

font_path='arial.ttf'

def generate(filename, length):
    return ''.join(np.random.choice(list(filename), length))

def generate_filename_with_date(file_format):
  file_id = generate('1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', 10)
  dir_path = datetime.now().strftime("%Y/%m/%d")
  return f"{dir_path}/{file_id}.{file_format}"


