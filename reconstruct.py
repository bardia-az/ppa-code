# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import os
import sys
from pathlib import Path
from PIL import Image

import numpy as np
import torch
from tqdm import tqdm

from compress import compress_input, compress_tensors, tensors_to_tiled, tiled_to_tensor

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from models.supplemental import AutoEncoder, Decoder_Rec
from utils.datasets import LoadImages
from utils.general import check_requirements, check_suffix, increment_path, print_args
from utils.torch_utils import select_device
from utils.utils import add_noise


@torch.no_grad()
def run(weights=None,  # model.pt path(s)
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        augment=False,  # augmented inference
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        noise_type=None,
        noise_spot='latent',
        noise_param=1,
        cut_layer=4,
        model=None,
        save_dir=Path(''),
        autoencoder=None,
        rec_model = None,
        sample_img = None,
        store_img = None,
        compression = None,
        qp = None,
        chs_in_w = 8,
        chs_in_h = 8,
        tensors_min = -1,
        tensors_max = 1
        ):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        device = select_device(device, batch_size=1)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        check_suffix(weights, '.pt')
        model = attempt_load(weights, map_location=device)  # load FP32 model
        model.cutting_layer = getattr(model, 'cutting_layer', cut_layer)
        print('YOLO loaded successfully')

        # Loading the supplemental models
        ckpt = torch.load(weights[0], map_location=device)  # load checkpoint
        # ---- Autoencoder -----#
        if 'autoencoder' in ckpt:
            autoenc_chs = ckpt['autoencoder'].chs
            autoencoder = AutoEncoder(autoenc_chs).to(device)
            autoencoder.load_state_dict(ckpt['autoencoder'].state_dict())
            print('Autoencoder loaded successfully')
        else:
            raise Exception('autoencoder is not available in the checkpoint')
        # Loading Reconstruction model
        if 'rec_model' in ckpt:
            rec_model = Decoder_Rec(cin=ckpt['rec_model'].cin, cout=ckpt['rec_model'].cout, first_chs=getattr(ckpt['rec_model'], 'first_chs', None) or getattr(ckpt['rec_model'], 'autoenc_chs', None)).to(device)
            rec_model.load_state_dict(ckpt['rec_model'].float().state_dict())
            print('RecNet loaded successfully')
        else:
            raise Exception('RecNet model is not available in the checkpoint')
        
    # Half precision
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    model.eval()
    model.half() if half else model.float()
    if rec_model is not None:
        rec_model.eval()
        rec_model.half() if half else rec_model.float()
    if autoencoder is not None:
        autoencoder.eval()
        autoencoder.half() if half else autoencoder.float()

    if not training and device.type != 'cpu':
        model(torch.zeros(1, 3, 1024, 1024).to(device).type_as(next(model.parameters())))  # run once

    gs = 8  # grid size (max stride for reconstruction only)

    if sample_img is not None:
        imgs = LoadImages(sample_img, img_size=None, stride=gs, auto=True, scaleup=False, store_img=(compression=='input'))
        data_iter = tqdm(imgs, desc='storing the reconstructed images') if not training else imgs
        for path, img, _, _, _ in data_iter:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()
            img /= 255
            ch_h, ch_w = img.shape[-2:]
            ch_w //= 8     # stride is 8 at layers 3 and 4
            ch_h //= 8     # stride is 8 at layers 3 and 4
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            if noise_spot=='input':
                img = add_noise(img, noise_type=noise_type, noise_param=noise_param)
            if compression=='input':
                rec_img, _, _ = compress_input(img, qp, half)
            else:
                T = model(img, augment=augment, cut_model=1)  # first half of the model
                if noise_spot=='latent':
                    T = add_noise(T, noise_type=noise_type, noise_param=noise_param)
                T_bottleneck = autoencoder(T, task='enc') if autoencoder is not None else T
                if noise_spot=='bottleneck':
                    T_bottleneck = add_noise(T_bottleneck, noise_type=noise_type, noise_param=noise_param)
                if compression=='bottleneck':
                    tiled_tensors = tensors_to_tiled(T_bottleneck, chs_in_w, chs_in_h, tensors_min, tensors_max)
                    rec_tensors, _, _ = compress_tensors(tiled_tensors, ch_w*chs_in_w, ch_h*chs_in_h, qp)
                    rec_tensors = rec_tensors.half() if half else rec_tensors.float()
                    rec_tensors = tiled_to_tensor(rec_tensors, ch_w, ch_h, tensors_min, tensors_max)
                    T_bottleneck = rec_tensors[None, :]
                rec_img = rec_model(T_bottleneck)
            pic = (rec_img.squeeze().detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            im = Image.fromarray(np.moveaxis(pic,0,-1), 'RGB')
            store_path = str(save_dir / Path(path).with_suffix('.png').name) if store_img is None else store_img
            im.save(store_path, format='PNG')
    
    # Return results
    model.float()  # for training


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--noise-type', default=None, choices=['uniform', 'gaussian', 'laplacian', 'dropout'], help='type of the added noise')
    parser.add_argument('--noise-spot', default='latent', choices=['latent', 'bottleneck', 'input'], help='where noise should be applied')
    parser.add_argument('--noise-param', type=float, default=1, help='noise parameter (length for uniform, std for gaussian, lambda for laplacia, prob for dropout)')
    parser.add_argument('--cut-layer', type=int, default=4, help='the index of the cutting layer (AFTER this layer, the model will be split)')
    parser.add_argument('--sample-img', type=str, default=ROOT / 'data/images', help='A sample image or a directory of images that wouold be reconstructed and stored')
    parser.add_argument('--compression', type=str, default=None, choices=['input', 'bottleneck'], help='compress input or latent space or do not compress at all')
    parser.add_argument('--qp', type=int, default=24, help='QP for the vvc encoder')
    parser.add_argument('--chs-in-w', type=int, default=8, help='number of channels in width in the tiled tensor')
    parser.add_argument('--chs-in-h', type=int, default=8, help='number of channels in height in the tiled tensor')
    parser.add_argument('--tensors-min', type=float, default=-0.2786, help='the clipping lower bound for the intermediate tensors')
    parser.add_argument('--tensors-max', type=float, default=1.4, help='the clipping upper bound for the intermediate tensors')

    opt = parser.parse_args()
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
