import os
import re
from PIL import Image
import requests
import numpy as np
import cv2
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision

from facenet_pytorch import MTCNN

from age_estimation import AgePredictor
from clip_interrogator import Interrogator, Config
from torch_deep_danbooru import DeepDanbooru

class Tagger:
    def __init__(self, args):
        try:
            print('Tagger: Initializing')

            if args.model == 'deepdanbooru':
                self.deep_danbooru = DeepDanbooru('cache\\model-resnet_custom_v3.pt')
                print('DeepDanbooru: Initializing')
            else:
                self.clip_interrogator = ClipInterrogator(args.model, args)
                print('clip_interrogator: Initialized')

            self.age_predictor = AgePredictor('cache\\megaage_fusion.pth')
            print('AgePredictor: Initialized')

        except Exception as e:
            torch.cuda.empty_cache()
            print(f'Tagger: __init__ : {e}')
            raise e

class ClipInterrogator:
    def __init__(self, model, args):
        try:
            device = 'cuda'
            self.model = model
            self.ci = Interrogator(Config(
                clip_model_name=self.model
                ))
        except Exception as e:
            torch.cuda.empty_cache()
            print(f'ClipInterrogator: __init__ : {e}')
            raise e

    def interrogate(self, image):
        print('ClipInterrogator: start interrogate')
        try:
            w, h = image.size
            aspect = w / h
            basesize = 512
            if aspect > 1:
                w = basesize
                h = int(basesize / aspect)
            else:
                h = basesize
                w = int(basesize * aspect)
            image = image.resize((w, h), Image.LANCZOS)

            prompt = self.ci.interrogate(image)

            print(f'ClipInterrogator: interrogate : {prompt}')
            torch.cuda.empty_cache()
            return prompt
        except Exception as e:
            torch.cuda.empty_cache()
            print(f'ClipInterrogator: interrogate : {e}')
            return None
