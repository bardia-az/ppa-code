# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
# import torchjpeg.dct as dct

from val_image_compression import compress_input, compress_tensors, tensors_to_tiled, tiled_to_tensor

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from models.supplemental import AutoEncoder, Decoder_Rec
from utils.datasets import create_rec_dataloader, LoadImages
from utils.general import LOGGER, StatCalculator, check_img_size, check_requirements, check_suffix, increment_path, print_args
from utils.torch_utils import select_device
from utils.loss import ComputeRecLoss, CompressibilityLoss

def add_noise(Tin, noise_type=None, noise_param=1):
    if noise_type is not None:
        if(noise_type.lower()=='gaussian'):
            N = (noise_param) * torch.randn_like(Tin)    # gauss_var is actually sigma, not variance
            Tout = Tin + N
        elif(noise_type.lower()=='uniform'):
            N = (noise_param) * torch.rand_like(Tin) - noise_param/2
            Tout = Tin + N
        elif(noise_type.lower()=='dropout'):
            Tout = F.dropout(Tin, p=noise_param)
        elif(noise_type.lower()=='laplacian'):
            N = torch.from_numpy(np.random.laplace(0, noise_param, Tin.shape)).to(Tin.dtype).to(Tin.device)
            Tout = Tin + N
    else:
        Tout = Tin
    return Tout

@torch.no_grad()
def run(rec_list,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        augment=False,  # augmented inference
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        noise_type=None,
        noise_spot='latent',
        noise_param=1,
        track_stats=False,
        dist_range=[-10,14],
        bins=10000,
        cut_layer=-1,
        model=None,
        dataloader=None,
        save_dir=Path(''),
        autoencoder=None,
        rec_model = None,
        sample_img = None,
        store_img = None,
        recnet_chs = [],
        compression = None,
        qp = None,
        chs_in_w = 8,
        chs_in_h = 8,
        tensors_min = -1,
        tensors_max = 1
        ):
    # Initialize/load model and set device
    training = model is not None
    gs = 8
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        check_suffix(weights, '.pt')
        model = attempt_load(weights, map_location=device)  # load FP32 model
        model.cutting_layer = model.cutting_layer if hasattr(model, 'cutting_layer') else cut_layer
        imgsz = check_img_size(imgsz, s=gs)  # check image size

        # Loading the models
        ckpt = torch.load(weights[0], map_location=device)  # load checkpoint
        print('YOLO loaded successfully')
        # ---- Autoencoder -----#
        if 'autoencoder' in ckpt:
            autoenc_chs = ckpt['autoencoder'].chs
            autoencoder = AutoEncoder(autoenc_chs).to(device)
            autoencoder.load_state_dict(ckpt['autoencoder'].state_dict())
            autoencoder.half() if half else autoencoder.float()
            print('Autoencoder loaded successfully')
        else:
            raise Exception('autoencoder is not available in the checkpoint')
        # Loading Reconstruction model
        if 'rec_model' in ckpt:
            rec_model = Decoder_Rec(cin=ckpt['rec_model'].cin, cout=ckpt['rec_model'].cout, first_chs=recnet_chs).to(device)
            rec_model.load_state_dict(ckpt['rec_model'].float().state_dict())
            rec_model.half() if half else rec_model.float()
            print('RecNet loaded successfully')
        else:
            raise Exception('autoencoder is not available in the checkpoint')

    # Half
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    model.half() if half else model.float()
    if autoencoder is not None:
        autoencoder.half() if half else autoencoder.float()
    if rec_model is not None:
        rec_model.half() if half else rec_model.float()

    rec_model.eval()
    autoencoder.eval()
    model.eval()

    compute_rec_loss = ComputeRecLoss(MAX=1.0)  # init loss class
    compressibility_loss = CompressibilityLoss(device)
    stats_bottleneck = StatCalculator(dist_range, bins, per_chs=False)

    # Dataloader
    if len(imgsz)==1:
        imgsz = imgsz * 2
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz[0], imgsz[1]).to(device).type_as(next(model.parameters())))  # run once
        dataloader = create_rec_dataloader(rec_list, imgsz, batch_size, augment=False, infinite=False, shuffle=False)

    s = ('%11s' * 6) % ('cmprs', 'L1', 'L2', 'PSNR', 'grad_loss', 'MS-SSIM')
    loss_rec = torch.zeros(5, device=device)
    loss_r = torch.zeros(1, device=device)
    
    for img in tqdm(dataloader, desc=s):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.
        if noise_spot=='input':
            img = add_noise(img, noise_type=noise_type, noise_param=noise_param)
        # Run model
        T = model(img, augment=augment, cut_model=1)  # first half of the model
        if noise_spot=='latent':
            T = add_noise(T, noise_type=noise_type, noise_param=noise_param)
        T_bottleneck = autoencoder(T, task='enc')
        if noise_spot=='bottleneck':
            T_bottleneck = add_noise(T_bottleneck, noise_type=noise_type, noise_param=noise_param)
        rec_img = rec_model(T_bottleneck).clamp_(0,1)
        loss_r += compressibility_loss(T_bottleneck.float())[1]
        loss_rec += compute_rec_loss(img, rec_img)[1]
        if track_stats:
            stats_bottleneck.update_stats(T_bottleneck.detach().clone().cpu().numpy())
            
    # Print results
    pf = ('%11.3g' * 6)
    loss_tot = (torch.cat((loss_r, loss_rec)) / len(dataloader)).cpu()
    LOGGER.info(pf % (*loss_tot,))
    if not training:
        csv_file = save_dir.parent / 'results.csv'
        s = '' if csv_file.exists() else (('%20s,' + '%11s,' * 6) % ('ID', 'cmprs', 'L1', 'L2', 'PSNR', 'grad_loss', 'MS-SSIM')) + '\n'
        with open( csv_file, 'a') as f: 
            f.write(s + (('%20s,' + '%11.3g,'*6) % (name, *loss_tot)) + '\n') 
    
    if sample_img is not None:
        imgs = LoadImages(sample_img, img_size=None, stride=gs, auto=True, scaleup=False, store_img=(compression=='input'))
        data_iter = tqdm(imgs, desc='storing the reconstructed images') if not training else imgs
        for path, img, _, _, _ in data_iter:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()
            img /= 255
            ch_h, ch_w = img.shape[-2:]
            ch_w //= 8     # beware that the stride is 8 at layers 3 and 4
            ch_h //= 8     # beware that the stride is 8 at layers 3 and 4
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
    
    if track_stats:
        stats_bottleneck.output_stats(save_dir)

    # Return results
    model.float()  # for training

    return loss_tot


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rec-list', type=str, default=None, help='txt file containing the training list for the reconstruction task')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, nargs='*', default=[256], help='inference size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--noise-type', default=None, choices=['uniform', 'gaussian', 'laplacian', 'dropout'], help='type of the added noise')
    parser.add_argument('--noise-spot', default='latent', choices=['latent', 'bottleneck', 'input'], help='where noise should be applied')
    parser.add_argument('--noise-param', type=float, default=1, help='noise parameter (length for uniform, std for gaussian, lambda for laplacia, prob for dropout)')
    parser.add_argument('--track-stats', action='store_true', help='track the statistical properties of the residuals')
    parser.add_argument('--dist-range',  type=float, nargs='*', default=[-3,3], help='the range of the distribution')
    parser.add_argument('--bins', type=int, default=1000, help='number of bins in histogram')
    parser.add_argument('--cut-layer', type=int, default=-1, help='the index of the cutting layer (AFTER this layer, the model will be split)')
    parser.add_argument('--sample-img', type=str, default=None, help='A sample image or a directory of images that wouold be reconstructed and stored')
    parser.add_argument('--recnet-chs',  type=int, nargs='*', default=[], help='number of channels in the frist non-yolo layers of RecNet')
    parser.add_argument('--compression', type=str, default=None, choices=['input', 'bottleneck'], help='compress input or latent space or do not compress at all')
    parser.add_argument('--qp', type=int, default=24, help='QP for the vvc encoder')
    parser.add_argument('--chs-in-w', type=int, default=8, help='number of channels is width in the tiled tensor')
    parser.add_argument('--chs-in-h', type=int, default=8, help='number of channels is height in the tiled tensor')
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
