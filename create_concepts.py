import os.path as osp
import argparse
import json
import logging
from modules.concepts import CreateConceptsList

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
    )

parser = argparse.ArgumentParser()

parser.add_argument("--source", type = str, help="Source directory")
parser.add_argument("--export", type = str, help="Crop images and create concept list export them to the specified directory")
parser.add_argument("--instance_token", type = str, help="Instance token")
parser.add_argument("--class_token", type = str, help="Class token")
parser.add_argument("--sub_token", type = str, help="Sub token")


args = parser.parse_args()

def create_concepts_list(dest, instance_token, class_token, sub_token):
        concepts = CreateConceptsList(dest, instance_token, class_token, sub_token)
        concepts_list = concepts.create_concept()
        return concepts_list


def main():
    logging.info(f'Create concepts list')
    logging.info(f'dest:{args.export} instance token:{args.instance_token} class_token:{args.class_token} sub_token:{args.sub_token}')
    concepts_list = create_concepts_list(args.export, args.instance_token, args.class_token, args.sub_token)
    conceots_list_filepath = osp.join(args.export, 'concepts_list.json')
    with open(conceots_list_filepath, mode='w') as f:
        json.dump(concepts_list, f, indent=4)
    

if __name__ == '__main__':
    main()
    pass