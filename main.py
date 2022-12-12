import os.path as osp
import os
import socket
import glob
import cv2
import numpy as np
from PIL import Image
import time
import math
import random
import threading
from modules.upscaler import ESRGANer, SwinIRer
from modules.taging import BLIPer, DeepDanbooru
from modules.age import AgePredictor
from modules.autocrop import crop_image, Settings
from modules.image_proccess import aspect_calc, aspect_crop, center_crop, weighted_crop, frame_crop, opencv2pil, pil2opencv

from modules.concepts import CreateConceptsList
import argparse
import json
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
    )

parser = argparse.ArgumentParser()

parser.add_argument("--source", type = str, help="Source directory")
parser.add_argument("--export", type = str, help="Crop images and create concept list export them to the specified directory")

parser.add_argument("--upscale", action='store_true', help="Upscale images")
parser.add_argument("--smooth", action='store_true', help="Using swinir")
parser.add_argument("--basesize", type = int, default = 1024, help="Specify base size as integer")

parser.add_argument("--crop", action='store_true', help="Crop images")
parser.add_argument("--square", action='store_true', help="Crop to square")

parser.add_argument("--tag", action='store_true', help="Taging images")
parser.add_argument("--blip", action='store_true', help="BLIP")
parser.add_argument("--danbooru", action='store_true', help="Danbooru")
parser.add_argument("--no_person", action='store_true', help="No person")
parser.add_argument("--age", type = int, help="Age fix target")

parser.add_argument("--limit", type = int, help="Image choice limit number")
parser.add_argument("--random", action='store_true', help="Random choice")
parser.add_argument("--format", type = str, default = "png", help="jpg or png")
parser.add_argument("--force", action='store_true', help="Force all output")

parser.add_argument("--concepts", action='store_true', help="Create concepts list")
parser.add_argument("--instance_token", type = str, help="Instance token")
parser.add_argument("--class_token", type = str, help="Class token")
parser.add_argument("--merge_concepts", type = str, help="Current concepts dir")

args = parser.parse_args()


if args.upscale or args.tag or args.concepts:
    if not args.source:
        logging.info('No source directory')
        raise SystemExit('No source directory')

if args.crop:
    if not args.source or not args.export:
        logging.info('No source or destination directory')
        raise SystemExit('No source or destination directory')

if args.concepts:
    if not args.instance_token or not args.class_token:
        logging.info('No instance token or class token')
        raise SystemExit('No instance token or class token')

if args.merge_concepts:
    logging.info('Merge concepts')
    new_concepts_list = []
    for root, dirs, files in os.walk(args.merge_concepts):
        for file in files:
            if file == 'concepts_list.json':
                concepts_list_filepath = osp.join(root, file)
                logging.info(f'Load concepts list: {concepts_list_filepath}')
                with open(concepts_list_filepath, mode='r') as f:
                    concepts_list = json.load(f)
                    new_concepts_list.extend(concepts_list)
    new_concepts_list_filepath = osp.join(args.merge_concepts, 'merged_concepts_list.json')
    with open(new_concepts_list_filepath, mode='w') as f:
        json.dump(new_concepts_list, f, indent=4)
    logging.info(f'Export concepts list: {new_concepts_list_filepath}')
    raise SystemExit(0)
 
if not args.source:
    logging.info('No source directory')
    raise SystemExit('No source directory')
    

target_path = args.source
if target_path[-1] == '/':
    target_path = target_path[:-1]

upscaled_dir_name = 'upscaled'
upscaled_dir = osp.join(target_path, upscaled_dir_name)

def detect_image_files(path):
    target_files = []
    count = 0
    for root, dirs, files in os.walk(path):
        if not upscaled_dir_name in root:
            if args.random:
                files = random.suffule(files)
            for file in files:
                if args.limit and count > args.limit -1:
                    count = 0
                    break
                if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
                    base_file_name = osp.splitext(file)[0]
                    png_file_name = base_file_name + '.png'
                    target_files.append(osp.join(root, file))
                    count += 1
    return target_files


class PrepareProcess:
    def __init__(self):
        if args.upscale:
            self.esrgan = ESRGANer()
            if args.smooth:
                self.swinir = SwinIRer()
        if args.blip:
            self.blip = BLIPer()
        if args.danbooru:
            self.deepbooru = DeepDanbooru()
        if args.age:
            self.ager = AgePredictor()

    def upscale(self, img, job):
        new_img = img
        if args.upscale:
            new_img = self.esrgan.upscale(img)
            if args.smooth:
                alt_img = self.swinir.upscale(img)
                new_img = cv2.addWeighted(new_img, 0.5, alt_img, 0.5, 0)
        return new_img

    def generate_caption(self, img, job):
        current_dir = osp.dirname(job.img_filepath).split('\\')
        current_dir_name = current_dir[len(current_dir)-1]
        new_caption = current_dir_name

        if args.blip:
            pil_img = opencv2pil(img)
            blip_caption = self.blip.get_caption(pil_img)
            blip_caption = ', '.join(blip_caption)
            new_caption = f'{new_caption}, {blip_caption}'


        if args.age:
            age = -1
            if not args.no_person:
                age = self.ager.predict(pil_img)
            if age != -1:
                if args.age:
                    age = int((age + args.age) / 2)
                new_caption = f'{new_caption}, {age} years old'

        if args.danbooru:
            booru_caption = self.deepbooru.get_tag(img)
            new_caption = f'{new_caption}, {booru_caption}'

        return new_caption

    def create_concepts_list(self, dest, instance_token, class_token):
        self.concepts = CreateConceptsList(dest, instance_token, class_token)
        concepts_list = self.concepts.create_concept()
        return concepts_list
      
class Job:
    def __init__(self, img_filepath, size):
        self.img_filepath = img_filepath
        self.name = osp.splitext(osp.basename(img_filepath))[0]
        self.size = size
        self.output_dir = args.export


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
    prepare = PrepareProcess()
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
            if not osp.exists(job.img_filepath):
                logging.info(f'{job.name}: File not found ({index + 1}/{len(target)})')
                continue
            logging.info(f'{job.name}: Proccessing {job.name} ({index + 1}/{len(target)})')

            # Specify output file
            output_dir = osp.join(osp.dirname(job.img_filepath), upscaled_dir_name)
            output_filepath = f'{output_dir}\\{job.name}.{args.format}'

            # Lock
            lock_file = f'{output_dir}\\{job.name}.lock'
            if osp.exists(lock_file):
                with open(lock_file, 'r') as f:
                    lock_owner = f.read()
                    logging.info(f'{job.name}: Locked by {lock_owner} ({index + 1}/{len(target)} {lap.lap()}')
                continue
            if not osp.exists(output_dir):
                os.makedirs(output_dir)
            with open(lock_file, 'w') as f:
                f.write(f'{host}')
                logging.info(f'{job.name}: Lock to {osp.basename(lock_file)} {lap.lap()}')

            # Upscale
            if args.upscale:
                if not osp.exists(output_filepath) or args.force:
                    img = cv2.imread(job.img_filepath)
                    logging.info(f'{job.name}: Read image {job.img_filepath} {lap.lap()}')
                    if img.shape[0] >= 2048 or img.shape[1] >= 2048:
                        upscaled_img = img
                        logging.info(f'{job.name}: The original file is large enough. skip upscale {upscaled_img.shape} {lap.lap()}')
                    else:
                        logging.info(f'{job.name}: Upscaling from {img.shape} {lap.lap()}')
                        upscaled_img = prepare.upscale(img, job)
                        logging.info(f'{job.name}: Upscaled to {upscaled_img.shape} {lap.lap()}')
                    if not osp.exists(output_dir):
                        os.makedirs(output_dir)
                    upscaled_img = aspect_crop(upscaled_img, 2048)
                    logging.info(f'{job.name}: Crop to {upscaled_img.shape} {lap.lap()}')
                    cv2.imwrite(output_filepath, upscaled_img)
                    logging.info(f'{job.name}: Save to {osp.basename(output_filepath)} {lap.lap()}')
                else:
                    upscaled_img = cv2.imread(output_filepath)
                    logging.info(f'{job.name}: {osp.basename(output_filepath)} already exists. Skip. {lap.lap()}')
            
            caption_file = f'{output_dir}\\{job.name}.txt'
            # Caption
            if args.tag and osp.exists(output_filepath):
                if not args.upscale:
                    upscaled_img = cv2.imread(output_filepath)                
                if not osp.exists(caption_file) or (args.force and args.tag):
                    caption = prepare.generate_caption(img=upscaled_img, job=job)
                    logging.info(f'{job.name}: Caption generated {lap.lap()}')
                    with open(caption_file, 'w') as f:
                        f.write(caption)
                        logging.info(f'{job.name}: Save to {osp.basename(caption_file)} {lap.lap()}')
                else:
                    with open(caption_file, 'r') as f:
                        caption = f.read()
                        logging.info(f'{job.name}: {osp.basename(caption_file)} already exists. Reuse this. {lap.lap()}')
                print(f'{caption}')


            # Export
            if args.crop and osp.exists(output_filepath):
                if not args.upscale:
                    upscaled_img = cv2.imread(output_filepath)
                if osp.exists(caption_file):
                    with open(caption_file, 'r') as f:
                        caption = f.read()
                else:
                    continue

                for i in range(3):
                    if not args.square:
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

                    else:
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

            # Unlock
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

    if args.concepts:
        print(f'Create concepts list: {args.export} {args.class_token} {args.instance_token}')
        concepts_list = prepare.create_concepts_list(args.export, args.class_token, args.instance_token)
        conceots_list_filepath = osp.join(args.export, 'concepts_list.json')
        with open(conceots_list_filepath, mode='w') as f:
            json.dump(concepts_list, f, indent=4)
        
    logging.info(f'All done. / {main_lap.total()}')



if __name__ == '__main__':
    main()
    pass
