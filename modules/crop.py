import cv2
import numpy as np
from PIL import Image
import math
from modules.lib.face_detect_crop import crop_image, Settings


def aspect_calc(in_image, basesize):
    image = pil2opencv(in_image)

    if type(image) == Image:
        image = np.array(image)
        image = np.array(image)

    h, w = image.shape[:2]    
    aspect = w / h
    crop_margin = 16
    block_size = 128
    if 1 <= aspect:
        np.clip(aspect, 1, 1.5)
#        logging.info(f'Aspect: {aspect:.2f}')
        new_basesize = round(basesize / aspect)
    
        scale_h = new_basesize + crop_margin
        crop_h = math.floor(new_basesize / block_size) * block_size
        scale_w = round(new_basesize * aspect) + crop_margin
        crop_w = math.floor((scale_w - crop_margin) / block_size) * block_size
    else:
#        logging.info('Aspect is less than 1')
        np.clip(aspect, 0.666, 1)
        new_basesize = round(basesize * aspect)
        scale_w = new_basesize + crop_margin
        crop_w = math.floor(new_basesize / block_size) * block_size
        scale_h = round(new_basesize / aspect) + crop_margin
        crop_h = math.floor((scale_h - crop_margin) / block_size) * block_size
#        logging.info(f'crop_w: {crop_w}')
    return crop_h, crop_w, scale_h, scale_w


def aspect_crop(in_image, base_size):
    image = pil2opencv(in_image)
    crop_h, crop_w, scale_h, scale_w = aspect_calc(image, base_size)
    scaled_image = cv2.resize(image, dsize=(scale_w, scale_h), interpolation=cv2.INTER_AREA)
#    logging.info(f'Scaled image shape: {scaled_image.shape}')

    x = scaled_image.shape[1]/2 - crop_w/2
    y = scaled_image.shape[0]/2 - crop_h/2
#    logging.info(f'x: {x}, y: {y}')

    cropped_image = scaled_image[int(y):int(y+crop_h), int(x):int(x+crop_w)]
#    logging.info(f'Cropped image shape: {croped_image.shape}')
    output_image = opencv2pil(cropped_image)
    return output_image

def center_crop(in_image, base_size):
    image = pil2opencv(in_image)
    crop_h, crop_w, scale_h, scale_w = aspect_calc(image, base_size)

    scaled_image = cv2.resize(image, dsize=(scale_w, scale_h), interpolation=cv2.INTER_AREA)
#    logging.info(f'Scaled image shape: {scaled_image.shape}')

    center = scaled_image.shape
    x = center[1]/2 - base_size/2
    y = center[0]/2 - base_size/2

    cropped_image = scaled_image[int(y):int(y+base_size), int(x):int(x+base_size)]
#    logging.info(f'Cropped image shape: {croped_image.shape}')
    output_image = opencv2pil(cropped_image)
    return output_image

def weighted_crop(in_image, height: int, width: int, corner_points_weight: float = 0.0, entropy_points_weight: float= 0.3, face_points_weight: float = 0.5):
    image = opencv2pil(in_image)
    settings = Settings(
        crop_width=width,
        crop_height=height,
        corner_points_weight=corner_points_weight,
        entropy_points_weight=entropy_points_weight,
        face_points_weight=face_points_weight,
        )
    output_image = crop_image(image, settings)[0]
    return output_image

def frame_crop(in_image, size):
    image = pil2opencv(in_image)
    resized = aspect_crop(image, size)
    resized = pil2opencv(resized)

    h, w = resized.shape[:2]
    dst = resized.copy()
    if h < size:
        top = (size - h) // 2
        bottom = size - h - top
        dst = cv2.copyMakeBorder(dst, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    if w < size:
        left = (size - w) // 2
        right = size - w - left
        dst = cv2.copyMakeBorder(dst, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    output_image = opencv2pil(dst)
    return output_image

def opencv2pil(in_image):
    if isinstance(in_image, np.ndarray):
        in_image_type = 'numpy'
    elif isinstance(in_image, Image.Image):
        in_image_type = 'pil'
    else:
        raise TypeError('Unknown image type')

    if in_image_type == 'numpy':
        new_image = in_image.copy()
        if new_image.ndim == 2:  # モノクロ
            pass
        elif new_image.shape[2] == 3:  # カラー
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        elif new_image.shape[2] == 4:  # 透過
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
        new_image = Image.fromarray(np.asarray(new_image, dtype=np.uint8))
    elif in_image_type == 'pil':
        new_image = in_image
    return new_image

def pil2opencv(in_image):
    if isinstance(in_image, np.ndarray):
        in_image_type = 'numpy'
    elif isinstance(in_image, Image.Image):
        in_image_type = 'pil'
    else:
        raise TypeError('Unknown image type')
    
    if in_image_type == 'pil':
        new_image = np.array(in_image, dtype=np.uint8)
        if new_image.ndim == 2:  # モノクロ
            pass
        elif new_image.shape[2] == 3:  # カラー
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        elif new_image.shape[2] == 4:  # 透過
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    elif in_image_type == 'numpy':
        new_image = in_image
    return new_image
