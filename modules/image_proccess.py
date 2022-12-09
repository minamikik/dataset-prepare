import cv2
import numpy as np
from PIL import Image
import math
from modules.autocrop import crop_image, Settings


def aspect_calc(img, basesize):
    h, w = img.shape[:2]    
    aspect = w / h
    crop_margin = 16
    if 1 <= aspect:
#        logging.info(f'Aspect: {aspect:.2f}')
        scale_h = basesize + crop_margin
        crop_h = basesize
        scale_w = round(basesize * aspect) + crop_margin
        crop_w = math.floor((scale_w - crop_margin) / 256) * 256
    else:
#        logging.info('Aspect is less than 1')
        scale_w = basesize + crop_margin
        crop_w = basesize
        scale_h = round(basesize / aspect) + crop_margin
        crop_h = math.floor((scale_h - crop_margin) / 256) * 256
#        logging.info(f'crop_w: {crop_w}')
    return crop_h, crop_w, scale_h, scale_w

def aspect_crop(img, base_size):
    crop_h, crop_w, scale_h, scale_w = aspect_calc(img, base_size)

    scaled_img = cv2.resize(img, dsize=(scale_w, scale_h), interpolation=cv2.INTER_AREA)
#    logging.info(f'Scaled image shape: {scaled_img.shape}')

    x = scaled_img.shape[1]/2 - crop_w/2
    y = scaled_img.shape[0]/2 - crop_h/2
#    logging.info(f'x: {x}, y: {y}')

    croped_img = scaled_img[int(y):int(y+crop_h), int(x):int(x+crop_w)]
#    logging.info(f'Cropped image shape: {croped_img.shape}')
    return croped_img

def center_crop(img, base_size):
    crop_h, crop_w, scale_h, scale_w = aspect_calc(img, base_size)

    scaled_img = cv2.resize(img, dsize=(scale_w, scale_h), interpolation=cv2.INTER_AREA)
#    logging.info(f'Scaled image shape: {scaled_img.shape}')

    x = scaled_img.shape[1]/2 - crop_w/2
    y = scaled_img.shape[0]/2 - crop_h/2
#    logging.info(f'x: {x}, y: {y}')

    croped_img = scaled_img[int(y):int(y+base_size), int(x):int(x+base_size)]
#    logging.info(f'Cropped image shape: {croped_img.shape}')
    return croped_img

def weighted_crop(img, height: int, width: int, corner_points_weight: float = 0.0, entropy_points_weight: float= 0.3, face_points_weight: float = 0.5):
    img = opencv2pil(img)
    settings = Settings(
        crop_width=width,
        crop_height=height,
        corner_points_weight=corner_points_weight,
        entropy_points_weight=entropy_points_weight,
        face_points_weight=face_points_weight,
        )
    new_img = crop_image(img, settings)[0]
    new_img = pil2opencv(new_img)
    return new_img

def frame_crop(img, size):
    resized = aspect_crop(img, size)

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

    return dst



def opencv2pil(in_image):
    new_image = in_image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(np.asarray(new_image, dtype=np.uint8))
    return new_image

def pil2opencv(in_image):
    new_image = np.array(in_image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image
