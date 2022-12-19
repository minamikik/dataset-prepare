import os
from os import path as osp
from PIL import Image
from queue import Queue
from modules.crop import frame_crop
from modules.tagger import Tagger
from modules.stop_watch import StopWatch

import argparse

flag=True
st = StopWatch()

models = [
    "ViT-L-14/openai",
    "ViT-H-14/laion2b_s32b_b79k",
    "xlm-roberta-large-ViT-H-14/frozen_laion5b_s13b_b90k",
    "deepdanbooru"
]

parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", type = str, help="Source directory")
parser.add_argument("--model", type = str, default = "ViT-L-14/openai", help="Clip model name. ViT-L-14/openai or ViT-H-14/laion2b_s32b_b79k")
parser.add_argument("--basesize", type = int, default = 768, help="Base size")
parser.add_argument("--age", type = int, help="Age estimation")
parser.add_argument("--force", action="store_true", help="Force to overwrite existing tags")
parser.add_argument("--alt_flavors", type = str, help="Alternative flavors text path")
args = parser.parse_args()

def create_job(path, jobs):
    try:
        for root, dirs, files in os.walk(path):
            if 'upscaled' in root:
                if files != []:
                    print(f'{root}: {len(files)}')
                    for file in files:
                        if 'upscaled' in root:
                            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
                                image_path = osp.join(root, file)
                                text_path = osp.join(root, osp.splitext(osp.basename(file))[0] + '.txt')
                                alt_text_path = osp.join(root, osp.splitext(osp.basename(file))[0] + "_" + args.model.split('/')[0] + '.txt')
                                job = {
                                    "name": osp.splitext(osp.basename(file))[0][:8] + '...',
                                    "status": f'{osp.split(osp.dirname(root))[1]}/{osp.basename(root)}/{osp.basename(file)[:64] + "..."}',
                                    "image_path": image_path,
                                    "text_path": text_path,
                                    "alt_text_path": alt_text_path
                                }
                                jobs.put(job)
    except Exception as e:
        print(f'create_job : {e}')
        return None


def worker_main(jobs, tagger):
    try:
        st.lap('worker_main start')
        while not jobs.empty():
            job = jobs.get()
            model = args.model
            print(f'worker_main: {job["status"]}')
            instance_image_path = job['image_path']
            instance_text_path = job['text_path']
            alt_text_path = job['alt_text_path']
            if args.force or not osp.exists(alt_text_path):
                instance_image = Image.open(instance_image_path)
                new_image = frame_crop(instance_image, args.basesize)
                if model == 'deepdanbooru':
                    prompt = tagger.deep_danbooru.get_tag(new_image, 0.8)
                else:
                    prompt = tagger.clip_interrogator.interrogate(new_image)

                if args.age:
                    age = tagger.age_predictor.predict(new_image)
                    prompt = f'{prompt}, {age} years old'
                with open(alt_text_path, 'w', encoding='utf-8') as f:
                    f.write(prompt)
            else:
                print(f'Already exists: {instance_text_path}')

            with open(alt_text_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
                with open(instance_text_path, 'w', encoding='utf-8') as f:
                    f.write(prompt)
            print(f'prompt: {prompt}')
            st.lap('worker_main loop end')
        return
    except Exception as e:
        raise e

def main():
    try:
        st.lap('main start')
        tagger =Tagger(args)
        st.lap('tagger init end')
        jobs = Queue()
        create_job(args.source_dir, jobs)
        st.lap('create_job end')
        worker_main(jobs, tagger)
    except Exception as e:
        print(f'main : {e}')
        raise e


if __name__ == '__main__':
    print('Start')
    main()
    print('All finished')
    pass
