import os
import re
from PIL import Image
import requests
import numpy as np
import torch
from torch import autocast
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from modules.blip_model import blip_decoder
from modules.deepbooru_model import DeepDanbooruModel
from modules.image_proccess import aspect_crop, weighted_crop, frame_crop, opencv2pil, pil2opencv
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
    )

re_special = re.compile(r'([\\()])')

 
class BLIPer:
    def __init__(self, model_path='models\\blip\\model_large_caption.pth', half=True, gpu_id=0, image_size=768):
        self.model_path = model_path
        self.half = half
        self.gpu_id = gpu_id
        self.image_size = image_size

        if os.path.exists(model_path):
            logging.info(f'BLIPer: Loading model from {model_path}')
        else:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth'
            logging.info(f'BLIPer: Downloading model {model_path}')
            r = requests.get(url, allow_redirects=True)
            open(model_path, 'wb').write(r.content)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'BLIPer: Using device {self.device}')
        self.model = blip_decoder(pretrained=self.model_path, image_size=image_size, vit='large')
        self.model.eval()
        if self.half:
            self.model.half()
        self.model = self.model.to(self.device)
        logging.info(f'BLIPer: Model loaded')

    def get_caption(self, pil_img):
  
  
        pic = resize_image(pil_img.convert("RGB"), 512, 512)

        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])

        with torch.no_grad(), autocast("cuda"):
            img = transform(pic).unsqueeze(0).to(self.device)
            # beam search
            caption = self.model.generate(img, sample=False, num_beams=3, max_length=20, min_length=5)
            # nucleus sampling
            # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
        torch.cuda.empty_cache()
        return caption



class DeepDanbooru:
    def __init__(self, model_path='models\\deepbooru\\model-resnet_custom_v3.pt', half=True, gpu_id=0, image_size=768):
        self.model_path = model_path
        self.half = half
        self.gpu_id = gpu_id
        self.image_size = image_size

        if os.path.exists(model_path):
            logging.info(f'DeepDanbooru: Loading model from {model_path}')
        else:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            url = 'https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/model-resnet_custom_v3.pt'
            logging.info(f'DeepDanbooru: Downloading model {model_path}')
            r = requests.get(url, allow_redirects=True)
            open(model_path, 'wb').write(r.content)


        self.device = torch.device('cuda')
        logging.info(f'DeepDanbooru: Using device {self.device}')
        self.model = DeepDanbooruModel()
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))

        self.model.eval()
        if self.half:
            self.model.half()
        self.model = self.model.to(self.device)
        logging.info(f'DeepDanbooru: Model loaded')

    def get_tag(self, img):
        threshold = 0.5
        use_spaces = True
        use_escape = True
        alpha_sort = True
        include_ranks = False

        if type(img) == np.ndarray:
            pil_image = opencv2pil(img)

        pic = resize_image(pil_image.convert("RGB"), 512, 512)
        a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255

        with torch.no_grad(), autocast("cuda"):
            x = torch.from_numpy(a).to(self.device)
            y = self.model(x).cpu().numpy()[0]
        torch.cuda.empty_cache()

        probability_dict = {}

        for tag, probability in zip(self.model.tags, y):
            if probability < threshold:
                continue

            if tag.startswith("rating:"):
                continue

            probability_dict[tag] = probability

        if alpha_sort:
            tags = sorted(probability_dict)
        else:
            tags = [tag for tag, _ in sorted(probability_dict.items(), key=lambda x: -x[1])]

        res = []

        for tag in tags:
            probability = probability_dict[tag]
            tag_outformat = tag
            if use_spaces:
                tag_outformat = tag_outformat.replace('_', ' ')
            if use_escape:
                tag_outformat = re.sub(re_special, r'\\\1', tag_outformat)
            if include_ranks:
                tag_outformat = f"({tag_outformat}:{probability:.3f})"

            res.append(tag_outformat)

        return ", ".join(res)
        


def resize_image(im, width, height):    
    ratio = width / height
    src_ratio = im.width / im.height

    src_w = width if ratio < src_ratio else im.width * height // im.height
    src_h = height if ratio >= src_ratio else im.height * width // im.width

    resized = im.resize((src_w, src_h), Image.LANCZOS)
    res = Image.new("RGB", (width, height))
    res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

    if ratio < src_ratio:
        fill_height = height // 2 - src_h // 2
        res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
        res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
    elif ratio > src_ratio:
        fill_width = width // 2 - src_w // 2
        res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
        res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))

    return res