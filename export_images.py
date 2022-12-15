from modules.stop_watch import StopWatch
import argparse
import logging
import random
import os
from os import path as osp
import json
import cv2
import queue

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
    )

parser = argparse.ArgumentParser()

parser.add_argument("--source_dir", type = str, help="Source directory")
parser.add_argument("--export_dir", type = str, help="Export directory")
parser.add_argument("--basesize", type = int, default = 768, help="Base size")
parser.add_argument("--aspect", action="store true", help="Aspect only mode (Default is all mode)")
parser.add_argument("--square", action="store true", help="Square only mode (Default is all mode)")
args = parser.parse_args()


class Job:
    def __init__(self, img_filepath, size):
        self.img_filepath = img_filepath
        self.name = osp.splitext(osp.basename(img_filepath))[0]
        self.size = size
        self.output_dir = args.export

def create_job(source_dir, jobs):
    for root, dirs, files in os.walk(source_dir):
        logging.info(f'files: {files}')
        for file in files:
            if 'upscaled' in root:
                if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
                    job = Job(img_filepath=osp.join(root, file), size=args.basesize)
                    jobs.put(job)

def worker_main(host, job):
    if not osp.exists(job.img_filepath):
        logging.info(f'{job.name}: File not found')
        return
    



def worker(host, jobs):
    try:
        global flag
        logging.info(f'{host["name"]} worker started')
        while flag:
            if not jobs.empty():
                job = jobs.get()
                response = worker_main(host, job)
            else:
                flag = False
        return f'{host["name"]} worker finished'
    except Exception as e:
        logging.error(f'{host["name"]} worker : {e}')
        return None



def main():
    main_lap = StopWatch()
    hosts_open = open('hosts.json', 'r')
    hosts = json.load(hosts_open)
    jobs = queue.Queue()
    lock = queue.Queue()



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
            output_dir = osp.join(osp.dirname(job.img_filepath), 'cropped')
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


            # Export
            if args.crop and osp.exists(output_filepath):
                if not args.upscale:
                    upscaled_img = cv2.imread(output_filepath)
                if osp.exists(caption_file):
                    with open(caption_file, 'r') as f:
                        caption = f.read()
                else:
                    continue

                if not args.square:
                    # aspect crop
                    new_size = args.crop_size
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

        except Exception as e:
            logging.error(f'{job.name}: {e} {lap.total()}')
            continue