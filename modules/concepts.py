import os
from os import path as osp
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
    )



class CreateConceptsList:
    def __init__(self, target_dir, instance_token, class_token):
        self.target_dir = osp.abspath(target_dir)
        self.instance_token = instance_token
        self.class_token = class_token

        self.concepts_list = []
    
    def create_concept(self):
        logging.info('CreateConceptsList: Creating concepts list')
        for root, dirs, files in os.walk(self.target_dir):
            if not 'class' in root:
                for dir in dirs:
                    instance_dir = osp.join(self.target_dir, dir)
                    class_dir = osp.join(self.target_dir, 'class')
                    sub_class_dir = osp.join(class_dir, dir)
                
                    if not osp.exists(class_dir):
                        os.mkdir(class_dir)
                    if not osp.exists(sub_class_dir):
                        os.mkdir(sub_class_dir)

                    instance_dir_files_count = len(os.listdir(instance_dir))
                    logging.info(f'{instance_dir}: {instance_dir_files_count}')

                    concept = {
                        "max_steps": -1,
                        "instance_data_dir": instance_dir,
                        "class_data_dir": sub_class_dir,
                        "instance_prompt": f"{self.class_token}, [filewords]",
                        "class_prompt": "[filewords]",
                        "save_sample_prompt": "[filewords]",
                        "save_sample_template": "",
                        "instance_token": self.instance_token,
                        "class_token": self.class_token,
                        "num_class_images": int(instance_dir_files_count * 5),
                        "class_negative_prompt": "",
                        "class_guidance_scale": 7.5,
                        "class_infer_steps": 40,
                        "save_sample_negative_prompt": "",
                        "n_save_sample": 1,
                        "sample_seed": -1,
                        "save_guidance_scale": 7.5,
                        "save_infer_steps": 40
                    }

                    self.concepts_list.append(concept)

        return self.concepts_list


