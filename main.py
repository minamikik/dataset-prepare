import os.path as osp
import os
import socket
import glob
import cv2
import numpy as np
from PIL import Image
import time
import math
import threading
from modules.upscaler import ESRGANer, SwinIRer
from modules.taging import BLIPer, DeepDanbooru
from modules.age import AgePredictor
from modules.autocrop import crop_image, Settings
from modules.image_proccess import aspect_calc, aspect_crop, center_crop, weighted_crop, frame_crop, opencv2pil, pil2opencv
import argparse
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
    )

parser = argparse.ArgumentParser()
parser.add_argument("--upscale", type = str, help="Upscale and caption images in the selected directory")
parser.add_argument("--crop", type = str, help="Crop images in the directory selected by upscaling and export them to the specified directory")
parser.add_argument("--basesize", type = int, default = 1024, help="Specify base size as integer")
parser.add_argument("--format", type = str, default = "png", help="jpg or png")
parser.add_argument("--force", action='store_true', help="Force all output")
parser.add_argument("--caption", action='store_true', help="Force caption output")
parser.add_argument("--age", type = int, help="Age fix target")
parser.add_argument("--no_person", action='store_true', help="No person")

args = parser.parse_args()

if not args.upscale:
    logging.info('No uppsclae target directory specified')
    exit()

target_path = args.upscale
if target_path[-1] == '/':
    target_path = target_path[:-1]

upscaled_dir = 'upscaled'
if args.crop:
    destination_dir = args.crop
else:
    destination_dir = target_path + upscaled_dir

def detect_image_files(path):
    target_files = []
    for current_dir, sub_dirs, file_list in os.walk(path):
        if not upscaled_dir in current_dir:
            for file in file_list:
                if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
                    base_file_name = osp.splitext(file)[0]
                    png_file_name = base_file_name + '.png'
                    target_files.append(osp.join(current_dir, file))
    return target_files


class PrepareProcess:
    def __init__(self):
        self.esrgan = ESRGANer()
        self.swinir = SwinIRer()
        self.blip = BLIPer()
        self.deepbooru = DeepDanbooru()
        self.ager = AgePredictor()

    def upscale(self, img, job):
        esrgan_img = self.esrgan.upscale(img)
        awinir_img = self.swinir.upscale(img)
        new_img = cv2.addWeighted(esrgan_img, 0.5, awinir_img, 0.5, 0)
        return new_img

    def generate_caption(self, img, job):
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
        return new_caption
      
class Job:
    def __init__(self, img_filepath, size):
        self.img_filepath = img_filepath
        self.name = osp.splitext(osp.basename(img_filepath))[0]
        self.size = size
        self.output_dir = destination_dir


class StopWatch:
    def __init__(self):
        self.start_time = time.time()
        self.lap_time = self.start_time
        self.total_time = self.start_time

    def lap(self):
        now = time.time()
        self.lap_time = now - self.total_time
        self.total_time = self.total_time + self.lap_time
        total_result = self.total_time - self.start_time
        return f'({round(self.lap_time, 2)}sec/{round(total_result, 2)}sec)'

    def total(self):
        now = time.time()
        self.lap_time = now - self.total_time
        self.total_time = self.total_time + self.lap_time
        total_result = self.total_time - self.start_time
        return f'({round(total_result, 2)}sec)'


def main():
    main_lap = StopWatch()
    logging.info(f'Initializing: {main_lap.lap()}')
    host = socket.gethostname()
    logging.info(f'Host: {host}')
    purepare = PrepareProcess()
    logging.info(f'Initialized: {main_lap.lap()}')
    logging.info(f'Detect image: Scanning {target_path} {main_lap.lap()}')
    target = detect_image_files(target_path)
    logging.info(f'Detect image: Found {len(target)} images {main_lap.lap()}')
    
    lock_file = None
    for index, img_filepath in enumerate(target):
        try:
            lap = StopWatch()
            print('-----------------------------------------------------------------')
            # Create job
            job = Job(img_filepath=img_filepath, size=args.basesize)
            logging.info(f'{job.name}: Proccessing {job.name} ({index + 1}/{len(target)})')

            # Specify output file
            output_dir = osp.join(osp.dirname(job.img_filepath), upscaled_dir)
            output_file = f'{output_dir}\\{job.name}.{args.format}'

            # Lock
            lock_file = f'{output_dir}\\{job.name}.lock'
            if not osp.exists(output_dir):
                os.makedirs(output_dir)
            if osp.exists(lock_file):
                with open(lock_file, 'r') as f:
                    lock_owner = f.read()
                    logging.info(f'{job.name}: Locked by {lock_owner} ({index + 1}/{len(target)} {lap.lap()}')
                continue
            with open(lock_file, 'w') as f:
                f.write(f'{host}')
                logging.info(f'{job.name}: Lock to {osp.basename(lock_file)} {lap.lap()}')

            # Upscale
            if not osp.exists(output_file) or args.force:
                img = cv2.imread(job.img_filepath)
                logging.info(f'{job.name}: Read image {job.img_filepath} {lap.lap()}')
                if img.shape[0] >= 2048 or img.shape[1] >= 2048:
                    upscaled_img = img
                    logging.info(f'{job.name}: The original file is large enough. skip upscale {upscaled_img.shape} {lap.lap()}')
                else:
                    logging.info(f'{job.name}: Upscaling from {img.shape} {lap.lap()}')
                    upscaled_img = purepare.upscale(img, job)
                    logging.info(f'{job.name}: Upscaled to {upscaled_img.shape} {lap.lap()}')
                if not osp.exists(output_dir):
                    os.makedirs(output_dir)
                upscaled_img = aspect_crop(upscaled_img, 2048)
                logging.info(f'{job.name}: Crop to {upscaled_img.shape} {lap.lap()}')
                cv2.imwrite(f'{output_dir}\\{job.name}.{args.format}', upscaled_img)
                logging.info(f'{job.name}: Save to {osp.basename(output_file)} {lap.lap()}')
            else:
                upscaled_img = cv2.imread(output_file)
                logging.info(f'{job.name}: {osp.basename(output_file)} already exists. Skip. {lap.lap()}')
            caption_file = f'{output_dir}\\{job.name}.txt'
            if not osp.exists(caption_file) or args.force or args.caption:
                caption = purepare.generate_caption(img=upscaled_img, job=job)
                logging.info(f'{job.name}: Caption generated {lap.lap()}')
                with open(caption_file, 'w') as f:
                    f.write(caption)
                    logging.info(f'{job.name}: Save to {osp.basename(caption_file)} {lap.lap()}')
            else:
                with open(caption_file, 'r') as f:
                    caption = f.read()
                    logging.info(f'{job.name}: {osp.basename(caption_file)} already exists. Reuse this. {lap.lap()}')

            # Cropping
            if args.crop:
                for i in range(3):
                    # aspect crop
                    new_size = round(job.size * ((4 - i) / 4))
                    h, w, _1, _2 = aspect_calc(upscaled_img, new_size)
                    output_dir = osp.join(job.output_dir, f'{w}x{h}')
                    output_file = f'{output_dir}\\{job.name}.{args.format}'
                    if not osp.exists(output_file) or args.force:
                        if not args.no_person:
                            cropped_img = weighted_crop(upscaled_img, h, w, 0.8, 0.0, 0.5)
                        else:
                            cropped_img = aspect_crop(upscaled_img, new_size)
                        if not osp.exists(output_dir):
                            os.makedirs(output_dir)
                        cv2.imwrite(output_file, cropped_img)
                        logging.info(f'{job.name}: Save to {osp.basename(output_file)} {lap.lap()}')
                    else:
                        logging.info(f'{job.name}: {osp.basename(output_file)} already exists. Skip. {lap.lap()}')
                    caption_file = f'{output_dir}\\{job.name}.txt'
                    if not osp.exists(caption_file) or args.force:
                        with open(caption_file, 'w') as f:
                            f.write(caption)
                        logging.info(f'{job.name}: Save to {osp.basename(caption_file)}')
                    else:
                        logging.info(f'{job.name}: {osp.basename(caption_file)} already exists. Skip. {lap.lap()}')

                    # square crop
                    h, w = round(new_size), round(new_size)
                    output_dir = osp.join(job.output_dir, f'{w}x{h}')
                    output_file = f'{output_dir}\\{job.name}.{args.format}'
                    if not osp.exists(output_file) or args.force:
                        if not args.no_person:
                            square_image = weighted_crop(upscaled_img, new_size, new_size, 1.0, 0.0, 40.0)
                        else:
                            square_image = center_crop(upscaled_img, new_size)
                        logging.info(f'{job.name}: Crop to {square_image.shape} {lap.lap()}')
                        if not osp.exists(output_dir):
                            os.makedirs(output_dir)
                        cv2.imwrite(f'{output_dir}\\{job.name}.{args.format}', square_image)
                        logging.info(f'{job.name}: Save to {osp.basename(output_file)} {lap.lap()}')
                    else:
                        logging.info(f'{job.name}: {osp.basename(output_file)} already exists. Skip. {lap.lap()}')
                    caption_file = f'{output_dir}\\{job.name}.txt'
                    if not osp.exists(caption_file) or args.force:
                        with open(caption_file, 'w') as f:
                            f.write(caption)
                        logging.info(f'{job.name}: Save to {osp.basename(caption_file)} {lap.lap()}')
                    else:
                        logging.info(f'{job.name}: {osp.basename(caption_file)} already exists. Skip. {lap.lap()}')

            logging.info(f'{job.name}: Job done. ({index + 1}/{len(target)}) {lap.total()}')
            print(f'{caption}')

            del job
            os.remove(lock_file)

        except KeyboardInterrupt:
            logging.info(f'KeyboardInterrupt: {job.name} {lap.total()}')
            os.remove(lock_file)
            logging.info(f'Lock file removed: {lock_file}')
            raise SystemExit('KeyboardInterrupt')
            
        except Exception as e:
            logging.exception(f'{job.name} {lap.total()}')
            os.remove(lock_file)
            logging.info(f'Lock file removed: {lock_file}')
            raise SystemExit(f'Exception: {e}')
        
    logging.info(f'All done. / {main_lap.total()}')



if __name__ == '__main__':
    main()
    pass
