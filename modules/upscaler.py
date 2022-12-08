import os.path as osp
import os
import glob
import cv2
import requests
import numpy as np
import torch
from torch import autocast
import time
import math

import modules.esrgan_model as arch
from modules.swinir_model import SwinIR as net
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
    )

class ESRGANer:
    def __init__(self, model_path='models/ESRGAN.pth', half=True, scale=4, gpu_id=0):
        self.model_path = model_path
        self.half = half
        self.gpu_id = gpu_id
        self.scale = scale
        self.tile = 512
        self.tile_overlap = 32
        self.window_size = 8

        if os.path.exists(model_path):
            logging.info(f'ESRGANer: Loading model from {model_path}')
        else:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            url = 'https://github.com/cszn/KAIR/releases/download/v1.0/ESRGAN.pth'
            logging.info(f'ESRGANer: Downloading model {model_path}')
            r = requests.get(url, allow_redirects=True)
            open(model_path, 'wb').write(r.content)


        # load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'ESRGANer: Using device {self.device}')
        self.model = arch.RRDBNet(3, 3, 64, 23, gc=32)
        self.model.load_state_dict(torch.load(self.model_path), strict=True)
        self.model.eval()
        if self.half:
            self.model.half()
        if self.gpu_id >= 0:
            self.model = self.model.cuda(self.gpu_id)
        logging.info('ESRGANer: Model loaded.')

    def upscale(self, img):
        time_sta = time.time()
        img_lq = img.astype(np.float32) / 255.
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(self.device)  # CHW-RGB to NCHW-RGB

        with torch.no_grad(), autocast("cuda"):
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
            w_pad = (w_old // self.window_size + 1) * self.window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = test(img_lq, self.model, self.tile, self.tile_overlap, self.scale, self.window_size)
            output = output[..., :h_old * self.scale, :w_old * self.scale]

        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round()
        torch.cuda.empty_cache()
        return output
        

class SwinIRer:
    def __init__(self, model_path='models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth', half=True, scale=4, gpu_id=0):
        self.model_path = model_path
        self.half = half
        self.gpu_id = gpu_id
        self.scale = scale
        self.tile = 512
        self.tile_overlap = 32
        self.window_size = 8

        # set up model
        if os.path.exists(self.model_path):
            logging.info(f'SwinIRer: Loading model from {self.model_path}')
        else:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth'
            logging.info(f'SwinIRer: Downloading model {self.model_path}')
            r = requests.get(url, allow_redirects=True)
            open(self.model_path, 'wb').write(r.content)


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'SwinIRer: Using device {self.device}')
        self.model = net(upscale=self.scale, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
        param_key_g = 'params_ema'
        pretrained_model = torch.load(model_path)
        self.model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

        self.model.eval()
        if self.half:
            self.model.half()
        if self.gpu_id >= 0:
            self.model = self.model.cuda(self.gpu_id)
        logging.info('SwinIRer: Model loaded.')

    def upscale(self, img):
        time_sta = time.time()
        img_lq = img.astype(np.float32) / 255.
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(self.device)  # CHW-RGB to NCHW-RGB

        with torch.no_grad(), autocast("cuda"):
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
            w_pad = (w_old // self.window_size + 1) * self.window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = test(img_lq, self.model, self.tile, self.tile_overlap, self.scale, self.window_size)
            output = output[..., :h_old * self.scale, :w_old * self.scale]

        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round()
        torch.cuda.empty_cache()
        time_end = time.time()
        time_cost = time_end - time_sta
        return output



def test(img_lq, model, tile, tile_overlap, scale, window_size):
    if tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = tile_overlap
        sf = scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output