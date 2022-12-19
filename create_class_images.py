import os
from os import path as osp
import io
import time
import base64
import random
import json
import requests
from PIL import Image, PngImagePlugin
from threading import Thread
from queue import Queue
from modules.stable_diffusion.txt2img import txt2img
from modules.stable_diffusion.hosts import check_available_hosts

import argparse
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
    )

flag=True
debug=False

parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", type = str, help="Source directory")
parser.add_argument("--sample_count", type = int, default=10, help="n_sample")
args = parser.parse_args()

def create_job(path, jobs):
    try:
        for root, dirs, files in os.walk(path):
            if not 'class' in root:
                if files != []:
                    logging.info(f'{root}: {len(files)}')
                    for file in files:
                        if not 'class' in root:
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


def worker_main(host, job):
    try:
        logging.info(f'{host}: {job["status"]}')
        instance_image_path = job['image_path']
        instance_text_path = job['text_path']
        instance_image = Image.open(instance_image_path)
        for i in range(args.sample_count):
            output_image_filename = f'class_{osp.basename(instance_image_path)}_{str(i).zfill(4)}.jpg'
            class_dir = osp.abspath(osp.join(osp.dirname(instance_image_path), '..', 'class'))
            class_image_dir = osp.join(class_dir, osp.basename(osp.dirname(instance_image_path)))
            class_image_path = osp.join(class_image_dir, output_image_filename)
            if osp.exists(class_image_path):
                continue
            os.makedirs(class_image_dir, exist_ok=True)
            width, height = instance_image.size
            with open(instance_text_path, 'r') as f:
                prompt = f.read()
            class_image = txt2img(host=host, prompt=prompt, width=width, height=height)
            if class_image is None:
                return f'{output_image_filename} failed'
            text_filename = output_image_filename.replace('.jpg', '.txt')
            class_image.save(class_image_path)
            with open(osp.join(class_image_dir, text_filename), 'w') as f:
                f.write(prompt)
        return f'{output_image_filename} done'
    except Exception as e:
        return None


def worker(host, jobs):
    try:
        global flag
        while flag:
            if not jobs.empty():
                job = jobs.get()
                worker_main(host, job)
            else:
                return f'{host} worker finished'
    except Exception as e:
        return None

def interrupt(jobs):
    try:
        global flag
        while True:
            if not jobs.empty():
                time.sleep(1)
            else:
                return
    except KeyboardInterrupt:
        flag=False
        return 0

def main(hosts):
    jobs = Queue()
    create_job(args.source_dir, jobs)

    available_hosts = check_available_hosts(hosts)
    if len(available_hosts) == 0:
        logging.error('No available hosts')
        raise SystemExit('No available hosts')

    threads = []

    t_interrupt = Thread(target=interrupt, args=(jobs,))
    t_interrupt.start()
    threads.append(t_interrupt)

    for host in available_hosts:
        t = Thread(target=worker, args=(host, jobs), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.5)

    for t in threads:
        t.join()
        logging.info(f'{t} finished')

    logging.info('All finished')


if __name__ == '__main__':
    logging.info('Start')
    hosts = json.load(open('hosts.json'))
    main(hosts)
    pass
