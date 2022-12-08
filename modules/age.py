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

from src.age.model import get_model
from src.age.defaults import _C as cfg
from src.age.dataset import expand_bbox

from PIL import Image
import math

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
    )

class AgePredictor:
    def __init__(self, model_path='models\\age\\megaage_fusion.pth'):
        logging.info('AgePredictor: Initializing')
        self.model_path = model_path

        if os.path.exists(model_path):
            logging.info(f'AgePredictor: Loading model from {model_path}')
        else:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            url = 'https://github.com/yjl450/age-estimation-ldl-pytorch/releases/download/v1.0/megaage_fusion.pth'
            logging.info(f'AgePredictor: Downloading model {model_path}')
            r = requests.get(url, allow_redirects=True)
            open(model_path, 'wb').write(r.content)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'AgePredictor: Using device {self.device}')

        self.model = get_model(model_name=cfg.MODEL.ARCH, pretrained=None)

        self.model.eval()
        self.model = self.model.to(self.device)
        logging.info(f'AgePredictor: Model loaded')

        checkpoint = torch.load(self.model_path, map_location="cpu")
        self.model.load_state_dict(checkpoint['state_dict'])

        self.mtcnn = MTCNN(device=self.device, post_process=False, keep_all=False)
        self.img_size = cfg.MODEL.IMG_SIZE


    def predict(self, pil_img):
        try:
            predicted_ages=[0]
            detected=[[0,0]]

            with torch.no_grad():
                image = pil_img
                detected, _, landmarks = self.mtcnn.detect(image, landmarks=True)

                if detected is not None and len(detected) > 0:
                    detected = detected.astype(int)
                    box = detected[0]
                    image = image.crop(box)
                    # image.show()
                    image.resize((self.img_size,self.img_size))
                    image = torchvision.transforms.ToTensor()(image)
                    image = image.unsqueeze(0).to(self.device)

                    # predict ages
                    outputs = self.model(image)
                    outputs = F.softmax(outputs, dim=1)
                    _, predicted_ages = outputs.max(1)

                    age = predicted_ages[0]

                else:
                    age = -1

            torch.cuda.empty_cache()
            return age
        except Exception as e:
            logging.error(f'AgePredictor: Error predicting age: {e}')
            torch.cuda.empty_cache()
            return -1


