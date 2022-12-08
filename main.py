import os.path as osp
import os
import glob
import cv2
import numpy as np
from PIL import Image
from pytorch_memlab import profile
import time
import math
import threading
from modules.upscaler import ESRGANer, SwinIRer
from modules.taging import BLIPer, DeepDanbooru
from modules.age import AgePredictor
from modules.autocrop import crop_image, Settings
from modules.image_proccess import aspect_calc, aspect_crop, weighted_crop, frame_crop, opencv2pil, pil2opencv
import argparse
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
    )

parser = argparse.ArgumentParser()
parser.add_argument("--upscale", type = str, help="source directory")
parser.add_argument("--crop", type = str, help="croped image output directory")
parser.add_argument("--caption", action='store_true', help="caption only")
parser.add_argument("--basesize", type = int, default = 1024, help="base size")
parser.add_argument("--format", type = str, default = "png", help="jpg or png")
parser.add_argument("--force", action='store_true', help="force update")
parser.add_argument("--age", type = int, help="Age fix target")
parser.add_argument("--no_person", action='store_true', help="No person")

args = parser.parse_args()

target_path = args.upscale
if target_path[-1] == '/':
    target_path = target_path[:-1]

upscaled_dir = 'upscaled'
if args.crop:
    destination_dir = args.crop
else:
    destination_dir = target_path + upscaled_dir

def detect_image_files(path):
    logging.info(f'Detect image: Scanning {path}')
    target_files = []
    for current_dir, sub_dirs, file_list in os.walk(path):
        if not upscaled_dir in current_dir:
            for file in file_list:
                if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
                    base_file_name = osp.splitext(file)[0]
                    png_file_name = base_file_name + '.png'
                    target_files.append(osp.join(current_dir, file))
    logging.info(f'Detect image: Found {len(target_files)} images')
    return target_files


class PrepareProcess:
    def __init__(self):
        self.esrgan = ESRGANer()
        self.swinir = SwinIRer()
        self.blip = BLIPer()
        self.deepbooru = DeepDanbooru()
        self.ager = AgePredictor()

    def upscale(self, img, job):
        logging.info(f'{job.name}: Upscaling... {img.shape}')
        time_sta = time.time()
        esrgan_img = self.esrgan.upscale(img)
#        new_img = esrgan_img
        awinir_img = self.swinir.upscale(img)
        new_img = cv2.addWeighted(esrgan_img, 0.5, awinir_img, 0.5, 0)

        time_end = time.time()
        time_cost = time_end - time_sta
        logging.info(f'{job.name}: Upscale done. {new_img.shape} / {time_cost:.2f}s')
        return new_img

    def captionize  (self, img, job):
        time_sta = time.time()

        current_dir = osp.dirname(job.img_filepath).split('\\')
        current_dir_name = current_dir[len(current_dir)-1]

        pil_img = opencv2pil(img)
        blip_caption = self.blip.get_caption(pil_img)
        blip_caption = ', '.join(blip_caption)

        booru_caption = self.deepbooru.get_tag(img)

        age = -1
        if not args.no_person:
            age = self.ager.predict(pil_img)
        if age != -1:
            if args.age:
                age = int((age + args.age) / 2)
            new_caption = f'{current_dir_name}, {age} years old, {blip_caption}, {booru_caption}'
        else:
            new_caption = f'{current_dir_name}, {blip_caption}, {booru_caption}'

        time_end = time.time()
        time_cost = time_end - time_sta
        logging.info(f'{job.name}: Captionize done. / {time_cost:.2f}s')
        return new_caption
      
class Job:
    def __init__(self, img_filepath, size):
        self.img_filepath = img_filepath
        self.name = osp.splitext(osp.basename(img_filepath))[0]
        self.size = size
        self.output_dir = destination_dir

def main(target):
    purepare = PrepareProcess()

    for img_filepath in target:
        time_sta = time.time()
        logging.info('-----------------------------------------------------------------')
        # Create job
        job = Job(img_filepath=img_filepath, size=args.basesize)
        logging.info(f'{job.name}: Processing {job.img_filepath}')

        # Upscale
        output_dir = osp.join(osp.dirname(job.img_filepath), upscaled_dir)
        output_filepath = f'{output_dir}\\{job.name}.{args.format}'
        if not osp.exists(output_filepath) or args.force:
            img = cv2.imread(job.img_filepath)
            logging.info(f'{job.name}: Read image {img.shape}')
            if not osp.exists(output_dir):
                os.makedirs(output_dir)
            if img.shape[0] >= 2048 or img.shape[1] >= 2048:
                upscaled_img = img
                logging.info(f'{job.name}: The original file is large enough. skip upscale')
            else:
                upscaled_img = purepare.upscale(img, job)
            if not osp.exists(output_dir):
                os.makedirs(output_dir)
            upscaled_img = aspect_crop(upscaled_img, 2048)
            cv2.imwrite(f'{output_dir}\\{job.name}.{args.format}', upscaled_img)
            logging.info(f'{job.name}: {output_filepath} Saved.')
        else:
            upscaled_img = cv2.imread(output_filepath)
            logging.info(f'{job.name}: {output_filepath} already exists. Skip.')
        caption_filename = f'{output_dir}\\{job.name}.txt'
        if not osp.exists(caption_filename) or args.force:
            caption = purepare.captionize(img=upscaled_img, job=job)
            with open(caption_filename, 'w') as f:
                f.write(caption)
        else:
            with open(caption_filename, 'r') as f:
                caption = f.read()

        # Cropping
        if args.crop:
            for i in range(3):
                # aspect crop
                new_size = round(job.size * ((4 - i) / 4))
                h, w, _1, _2 = aspect_calc(upscaled_img, new_size)
                output_dir = osp.join(job.output_dir, f'{w}x{h}')
                output_filepath = f'{output_dir}\\{job.name}.{args.format}'
                if not osp.exists(output_filepath) or args.force:
    #                croped_img = aspect_crop(upscaled_img, new_size)
                    croped_img = weighted_crop(upscaled_img, h, w, 0.8, 0.0, 0.5)
                    if not osp.exists(output_dir):
                        os.makedirs(output_dir)
                    cv2.imwrite(output_filepath, croped_img)
                    logging.info(f'{job.name}: {output_filepath} Saved.')
                else:
                    logging.info(f'{job.name}: {output_filepath} already exists. Skip.')
                caption_filename = f'{output_dir}\\{job.name}.txt'
                if not osp.exists(caption_filename) or args.force:
                    with open(caption_filename, 'w') as f:
                        f.write(caption)
                    logging.info(f'{job.name}: {caption_filename} Saved.')
                else:
                    logging.info(f'{job.name}: {caption_filename} already exists. Skip.')

                # square crop
                h, w = round(new_size), round(new_size)
                output_dir = osp.join(job.output_dir, f'{w}x{h}')
                output_filepath = f'{output_dir}\\{job.name}.{args.format}'
                if not osp.exists(output_filepath) or args.force:
                    #square_image = square_crop(upscaled_img, new_size)
                    square_image = weighted_crop(upscaled_img, new_size, new_size, 1.0, 0.0, 40.0)
                    if not osp.exists(output_dir):
                        os.makedirs(output_dir)
                    cv2.imwrite(f'{output_dir}\\{job.name}.{args.format}', square_image)
                    logging.info(f'{job.name}: {output_filepath} Saved.')
                else:
                    logging.info(f'{job.name}: {output_filepath} already exists. Skip.')
                caption_filename = f'{output_dir}\\{job.name}.txt'
                if not osp.exists(caption_filename) or args.force:
                    with open(caption_filename, 'w') as f:
                        f.write(caption)
                    logging.info(f'{job.name}: {caption_filename} Saved.')
                else:
                    logging.info(f'{job.name}: {caption_filename} already exists. Skip.')

        time_end = time.time()
        time_cost = time_end - time_sta
        logging.info(f'caption: {caption}')
        logging.info(f'{job.name}: Job done. / {time_cost:.2f}s')
        del job



if __name__ == '__main__':
    time_sta = time.time()
    target = detect_image_files(target_path)
    main(target)
    time_end = time.time()
    time_cost = time_end - time_sta
    logging.info('All done. / {time_cost:.2f}s')
    pass