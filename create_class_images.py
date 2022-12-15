import requests
import json
from PIL import Image, PngImagePlugin
import io
import os
from os import path as osp
import time
import base64
import random
from threading import Thread
from queue import Queue
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
parser.add_argument("--source", type = str, help="Source directory")
parser.add_argument("--batch_size", type = int, default=10, help="n_sample")

args = parser.parse_args()


hosts = [
    {
        'name': 'moog:7860   ',
        'url': 'moog.sync.local:7860',
    },
    {
        'name': 'wezen:7860  ',
        'url': 'wezen.sync.local:7860',
    },
    {
        'name': 'peacock:7860',
        'url': 'peacock.sync.local:7860',
    },
    {
        'name': 'peacock:7861',
        'url': 'peacock.sync.local:7861',
    },
    {
        'name': 'peacock:7862',
        'url': 'peacock.sync.local:7862',
    }
]


def service_check(hosts):
    available_hosts = []
    for host in hosts:
        try:
            payload = {
                "prompt": 'test',
                "steps": 5,
                "width": 64,
                "height": 64,
                "batch_size": 1
            }
            response = requests.post(
                url=f'http://{host["url"]}/sdapi/v1/txt2img',
                data = json.dumps(payload),
                timeout = (3.0, 30.0)
            )
            if response.status_code == 200:
                available_hosts.append(host)
                logging.info(f'{host["name"]} is available: {response.status_code}')
            else:
                logging.info(f'{host["name"]} is not available: {response.status_code}')
            continue

        except Exception as e:
            logging.error(f'{host["name"]} is not available: {e}')
            continue
    return available_hosts



def create_job(path, jobs):
    try:
        image_files = []
        text_files = []
        for root, dirs, files in os.walk(path):
            if not 'class' in root:
                if files != []:
                    logging.info(f'{root}: {len(files)}')
                    random.shuffle(files)
                    for file in files:
                        if not 'class' in root:
                            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
                                image_path = osp.join(root, file)
                                text_path = osp.join(root, osp.splitext(osp.basename(file))[0] + '.txt')
                                job = {
                                    "name": osp.splitext(osp.basename(file))[0][:8] + '...',
                                    "status": f'{osp.split(osp.dirname(root))[1]}/{osp.basename(root)}/{osp.splitext(osp.basename(file))[0][:8] + "..."}',
                                    "image_path": image_path,
                                    "text_path": text_path
                                }
                                jobs.put(job)
    except Exception as e:
        logging.error(f'create_job : {e}')
        return None


def txt2img(host, prompt="", negative_prompt="", width=512, height=512):
    try:
        payload = {
            "prompt": prompt,
            "steps": 40,
            "width": width,
            "height": height,
            "batch_size": 1
        }
        response = requests.post(
            url=f'http://{host["url"]}/sdapi/v1/txt2img',
            data = json.dumps(payload),
            timeout = (5.0, 300.0)
        )
        r = response.json()
        for i in r['images']:
            image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
            png_payload = {"image": "data:image/png;base64," + i}
            response2 = requests.post(
                url=f'http://{host["url"]}/sdapi/v1/png-info',
                json=png_payload
            )
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("parameters", response2.json().get("info"))

        return image
    except Exception as e:
        logging.error(f'{host["name"]}: txt2img : {e}')
        return None

def worker_main(host, job):
    try:
        instance_image_path = job['image_path']
        instance_text_path = job['text_path']
        instance_image = Image.open(instance_image_path)
        logging.info(f'{host["name"]} {job["status"]} / Take 0: File opened')
        for i in range(args.batch_size):
            output_image_filename = f'class_{osp.basename(instance_image_path)}_{str(i).zfill(4)}.jpg'
            class_dir = osp.abspath(osp.join(osp.dirname(instance_image_path), '..', 'class'))
            class_image_dir = osp.join(class_dir, osp.basename(osp.dirname(instance_image_path)))
            class_image_path = osp.join(class_image_dir, output_image_filename)
            if osp.exists(class_image_path):
                logging.info(f'{host["name"]} {job["status"]} / Take {i+1}: {osp.basename(class_image_path)} already exists')
                continue
            if not osp.exists(class_dir):
                os.mkdir(class_dir)
            if not osp.exists(class_image_dir):
                os.mkdir(class_image_dir)
            width, height = instance_image.size
            with open(instance_text_path, 'r') as f:
                prompt = f.read()
            class_image = txt2img(host=host, prompt=prompt, width=width, height=height)
            if class_image is None:
                return f'{output_image_filename} failed'
            text_filename = output_image_filename.replace('.jpg', '.txt')
            class_image.save(class_image_path)
            logging.info(f'{host["name"]} {job["status"]} / Take {i+1}: {osp.basename(class_image_path)} saved')
            with open(osp.join(class_image_dir, text_filename), 'w') as f:
                f.write(prompt)
        return f'{output_image_filename} done'
    except Exception as e:
        logging.error(f'{host["name"]} {job["status"]}: worker_main : {e}')
        return None


def worker(host, jobs):
    try:
        global flag
        logging.info(f'{host["name"]} worker started')
        while flag:
            if not jobs.empty():
                job = jobs.get()
                response = worker_main(host, job)
            else:
                return f'{host["name"]} worker finished'
    except Exception as e:
        logging.error(f'{host["name"]} worker : {e}')
        return None


def close(jobs):
    try:
        while True:
            if not jobs.empty():
                time.sleep(1)
            else:
                return
    except KeyboardInterrupt:
        flag=False
        raise SystemExit('KeyboardInterrupt')

def main(hosts):

    available_hosts = service_check(hosts)
    if len(available_hosts) == 0:
        logging.error('No available hosts')
        return

    jobs = Queue()
    create_job(args.source, jobs)

    threads = []

    if not debug:
        for host in available_hosts:
            t = Thread(target=worker, args=(host, jobs))
            t.setDaemon(True)
            t.start()
            threads.append(t)
            time.sleep(0.5)
    else:
        t = Thread(target=worker, args=(available_hosts[0], jobs))
        t.setDaemon(True)
        t.start()
        threads.append(t)
        time.sleep(0.5)

    close(jobs)

    for t in threads:
        t.join()
        logging.info(f'{t} finished')

    logging.info('All finished')


if __name__ == '__main__':
    main(hosts)
    pass
