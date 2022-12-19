import os
from os import path as osp
import shutil
import io
import time
import base64
import random
import json
import requests
from PIL import Image, PngImagePlugin
from threading import Thread
from queue import Queue
from modules.crop import aspect_calc, aspect_crop, center_crop, weighted_crop

import argparse
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
    )

flag=True

parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", type = str, help="Source directory")
parser.add_argument("--export_dir", type = str, help="Export directory")
parser.add_argument("--basesize", type = int, default = 768, help="Base size")
parser.add_argument("--aspect", action="store_true", help="Aspect only mode (Default is all mode)")
parser.add_argument("--square", action="store_true", help="Square only mode (Default is all mode)")
parser.add_argument("--no_person", action="store_true", help="No Person mode")
parser.add_argument("--force", action="store_true", help="Force update mode")
args = parser.parse_args()

def create_job(path, jobs):
    try:
        for root, dirs, files in os.walk(path):
            if 'upscaled' in root:
                if files != []:
                    logging.info(f'{root}: {len(files)}')
                    for file in files:
                        if 'upscaled' in root:
                            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
                                image_path = osp.join(root, file)
                                text_path = osp.join(root, osp.splitext(osp.basename(file))[0] + '.txt')
                                job = {
                                    "name": osp.splitext(osp.basename(file))[0][:8] + '...',
                                    "status": f'{osp.split(osp.dirname(root))[1]}/{osp.basename(root)}/{osp.basename(file)[:64] + "..."}',
                                    "image_path": image_path,
                                    "text_path": text_path
                                }
                                jobs.put(job)
    except Exception as e:
        logging.error(f'create_job : {e}')
        return None

def worker_main(job):
    try:
        source_image_path = job['image_path']
        source_text_path = job['text_path']
        source_image = Image.open(source_image_path)
        new_size = args.basesize
        h, w, _1, _2 = aspect_calc(source_image, new_size)
        logging.info(f'Ready to crop {source_image_path}')

        if args.aspect or not args.square:
            cropped_output_image_filename = f'{osp.splitext(osp.basename(source_image_path))[0]}.png'
            cropped_output_dir = osp.abspath(osp.join(osp.dirname(source_image_path), '..', 'cropped', f'{w}x{h}'))
            cropped_output_image_path = osp.join(cropped_output_dir, cropped_output_image_filename)
            if not osp.exists(cropped_output_image_filename) or args.force:
                os.makedirs(cropped_output_dir, exist_ok=True)
                with open(source_text_path, 'r') as f:
                    prompt = f.read()
                if not args.no_person:
                    cropped_image = weighted_crop(source_image, h, w, 0.8, 0.0, 0.5)
                else:
                    cropped_image = aspect_crop(source_image, new_size)
                cropped_image.save(cropped_output_image_path)
                output_text_filename = cropped_output_image_filename.replace('.png', '.txt')
                logging.info(f'Save aspect crop: {cropped_output_image_filename}')
                with open(osp.join(cropped_output_dir, output_text_filename), 'w') as f:
                    f.write(prompt)
            else:
                logging.info(f'Skip aspect crop: {cropped_output_image_filename} already exists.')



        if args.square or not args.aspect:
            square_output_image_filename = f'{osp.splitext(osp.basename(source_image_path))[0]}.png'
            square_output_dir = osp.abspath(osp.join(osp.dirname(source_image_path), '..', 'cropped', f'{new_size}x{new_size}'))
            square_output_image_path = osp.join(square_output_dir, square_output_image_filename)
            if not osp.exists(square_output_image_path) or args.force:
                os.makedirs(square_output_dir, exist_ok=True)
                with open(source_text_path, 'r') as f:
                    prompt = f.read()
                if not args.no_person:
                    square_image = weighted_crop(source_image, new_size, new_size, 0.8, 0.0, 0.5)
                else:
                    square_image = center_crop(source_image, h, w)
                output_text_filename = square_output_image_filename.replace('.png', '.txt')
                square_image.save(square_output_image_path)
                logging.info(f'Save square crop: {square_output_image_path}')
                with open(osp.join(square_output_dir, output_text_filename), 'w') as f:
                    f.write(prompt)
            else:
                logging.info(f'Skip square crop:  {cropped_output_image_filename} already exists.')

        return 0
    except Exception as e:
        logging.error(f'worker_main : {e}')
        raise e

def worker(jobs):
    try:
        while not jobs.empty():
            job = jobs.get()
            worker_main(job)
        logging.info('All finished')
    except Exception as e:
        raise e

def main():
    jobs = Queue()
    create_job(args.source_dir, jobs)
    worker(jobs)
    return 0

if __name__ == '__main__':
    main()
    pass

