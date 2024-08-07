# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import subprocess
import numpy as np
import cv2

from val_image_compression import tensors_to_tiled, tiled_to_tensor, read_y_channel

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from models.supplemental import AutoEncoder, Decoder_Rec
from utils.datasets import LoadImages
from utils.general import (LOGGER, check_requirements, increment_path, print_args)
from utils.torch_utils import select_device, time_sync


def compress_input(img, qp, half=False):
    dt = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    t0 = time_sync()
    h, w = img.shape[-2:]
    jpg2yuv_report = Path('../vvc_new/jpg2yuv_report.txt').open('w')
    vvc_report = Path('../vvc_new/vvc_report.txt').open('w')
    vvc_report_dec = Path('../vvc_new/vvc_report_dec.txt').open('w')
    yuv2png_report = Path('../vvc_new/yuv2png_report.txt').open('w')
    t1 = time_sync()
    dt[0] = t1 - t0
    
    jpg2yuv_command = ['ffmpeg', '-i', '../vvc_new/image.png', '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-dst_range', '1', '../vvc_new/yuv_img.yuv', '-y']
    subprocess.call(jpg2yuv_command, stdout=jpg2yuv_report, stderr=subprocess.STDOUT)
    t2 = time_sync()
    dt[1] = t2 - t1

    VVC_command = ['../vvc_new/vvencFFapp', '-c', '../vvc_new/lowdelay_faster.cfg', '-i', '../vvc_new/yuv_img.yuv', '-b', '../vvc_new/bitstream.bin', 
                   '-o', '../vvc_new/reconst.yuv', '--SourceWidth', str(w), '--SourceHeight', str(h), '-f', '1', '-fr', '1', '-q', str(qp)]
    subprocess.call(VVC_command, stdout=vvc_report)
    t3 = time_sync()
    dt[2] = t3 - t2

    VVC_dec_command = ['../vvc_new/DecoderAppStatic', '-b', '../vvc_new/bitstream.bin', '-o', '../vvc_new/reconst_dec.yuv']
    subprocess.call(VVC_dec_command, stdout=vvc_report_dec)
    t4 = time_sync()
    dt[3] = t4 - t3
    
    yuv2png_command = ['ffmpeg', '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-s', f"{w}x{h}", '-src_range', '1', '-i', '../vvc_new/reconst.yuv',
                       '-frames', '1', '-pix_fmt', 'rgb24', '../vvc_new/output.png', '-y']
    subprocess.call(yuv2png_command, stdout=yuv2png_report, stderr=subprocess.STDOUT)
    t5 = time_sync()
    dt[4] = t5 - t4

    rec = cv2.imread('../vvc_new/output.png')  # BGR
    rec = rec.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    rec = np.ascontiguousarray(rec)
    rec = torch.from_numpy(rec).to(img.device, non_blocking=True)
    rec = rec.half() if half else rec.float()  # uint8 to fp16/32
    rec /= 255  # 0 - 255 to 0.0 - 1.0
    t6 = time_sync()
    dt[5] = t6 - t5

    jpg2yuv_report.close()
    vvc_report.close()
    yuv2png_report.close()
    return rec.unsqueeze(0), dt


def compress_tensors(tensors, tensors_w, tensors_h, qp):
    dt = [0.0, 0.0, 0.0, 0.0]
    t0 = time_sync()
    data = tensors.cpu().numpy().flatten().astype(np.uint8)
    vvc_report = Path('../vvc_new/vvc_report2.txt').open('w')
    vvc_report_dec = Path('../vvc_new/vvc_report_dec2.txt').open('w')
    VVC_command = ['../vvc_new/vvencFFapp', '-c', '../vvc_new/lowdelay_faster_latent.cfg', '-i', '../vvc_new/tiled_img.yuv', '-b', '../vvc_new/bitstream2.bin', 
                   '-o', '../vvc_new/reconst2.yuv', '--SourceWidth', str(tensors_w), '--SourceHeight', str(tensors_h), '-f', '1', '-fr', '1', '-q', str(qp)]
    VVC_dec_command = ['../vvc_new/DecoderAppStatic', '-b', '../vvc_new/bitstream2.bin', '-o', '../vvc_new/reconst_dec2.yuv']
    to_be_coded_file = '../vvc_new/tiled_img.yuv'
    with open(to_be_coded_file, 'wb') as f:
        f.write(data)
    t1 = time_sync()
    dt[0] = t1 - t0

    subprocess.call(VVC_command, stdout=vvc_report)
    t2 = time_sync()
    dt[1] = t2 - t1

    subprocess.call(VVC_dec_command, stdout=vvc_report_dec)
    t3 = time_sync()
    dt[2] = t3 - t2

    with open('../vvc_new/reconst2.yuv', 'rb') as f:
        tmp_reconst = read_y_channel(f, tensors_w, tensors_h)
    t4 = time_sync()
    dt[3] = t4 - t3

    vvc_report.close()
    return torch.from_numpy(tmp_reconst.copy()).to(tensors.device, non_blocking=True), dt


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=None,  # inference size (pixels)
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_csv=False,  # save results to *.csv
        no_vvc=False,  # do not perfrom vvc compression
        augment=False,  # augmented inference
        project=ROOT / 'runs/timing',  # save results to project/name
        half=False,  # use FP16 half-precision inference
        autoenc_chs=None,   # number of channels in the auto encoder
        qp = None,
        chs_in_w = 8,
        chs_in_h = 8,
        tensors_min = -1,
        tensors_max = 1
        ):
    source = str(source)

    # Directories
    save_dir = Path(project)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = str(weights[0] if isinstance(weights, list) else weights)
    stride = 64

    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())  # model stride
    if half:
        model.half()  # to FP16
    
    # imgsz = check_img_size(imgsz, s=stride)  # check image size

   # Loading Autoencoder
    ckpt = torch.load(w, map_location=device)  # load checkpoint
    autoencoder = None
    if 'autoencoder' in ckpt:
        autoenc_chs = ckpt['autoencoder'].chs
        autoencoder = AutoEncoder(autoenc_chs).to(device)
        autoencoder.load_state_dict(ckpt['autoencoder'].state_dict())
        autoencoder.half() if half else autoencoder.float()
        autoencoder.eval()
        print('pretrained autoencoder')
    # Loading Reconstruction model
    rec_model = None
    if 'rec_model' in ckpt:
        rec_model = Decoder_Rec(cin=ckpt['rec_model'].cin, cout=ckpt['rec_model'].cout, first_chs=autoenc_chs).to(device)
        rec_model.load_state_dict(ckpt['rec_model'].float().state_dict())
        rec_model.half() if half else rec_model.float()
        rec_model.eval()
        print('pretrained reconstruction model')

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=True, store_img=True)

    dt, seen = [[], [], [], [], [], [], [], [], []], 0
    for path, img, im0s, vid_cap, s in tqdm(dataset):
        seen += 1
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()
        dt[0].append(t2 - t1)

        # Inference
        if autoencoder:
            T = model(img, augment=augment, cut_model=1, visualize=False)  # first half of the model
            t3 = time_sync()
            dt[1].append(t3 - t2)

            T_bottleneck = autoencoder(T, task='enc')
            tiled_tensors = tensors_to_tiled(T_bottleneck, chs_in_w, chs_in_h, tensors_min, tensors_max)
            t4 = time_sync()
            dt[2].append(t4 - t3)

            ch_h, ch_w = img.shape[-2:]
            ch_w //= 8     # beware that the stride is 8 at layers 3 and 4
            ch_h //= 8     # beware that the stride is 8 at layers 3 and 4
            if not no_vvc:
                rec_tensors, dt_tmp = compress_tensors(tiled_tensors, ch_w*chs_in_w, ch_h*chs_in_h, qp)
            for i in range(4):
                if no_vvc:
                    dt[i+3].append(0)
                else:
                    dt[i+3].append(dt_tmp[i])

            t5 = time_sync()
            if not no_vvc:
                rec_tensors = rec_tensors.half() if half else rec_tensors.float()
                rec_tensors = tiled_to_tensor(rec_tensors, ch_w, ch_h, tensors_min, tensors_max)
                T_bottleneck = rec_tensors[None, :]
            T_hat = autoencoder(T, task='dec', bottleneck=T_bottleneck)
            t6 = time_sync()
            dt[7].append(t6 - t5)

            pred = model(None, cut_model=2, T=T_hat, visualize=False)[0]  # second half of the model  
            t7 = time_sync()
            dt[8].append(t7 - t6)
        else:
            if not no_vvc:
                rec_img, dt_tmp = compress_input(img, qp, half)
            else:
                rec_img = img
            for i in range(6):
                if no_vvc:
                    dt[i+1].append(0)
                else:
                    dt[i+1].append(dt_tmp[i])
            t3 = time_sync()
            pred = model(rec_img, augment=augment, visualize=False)[0]
            t4 = time_sync()
            dt[7].append(t4 - t3)

    # Print results
    if autoencoder:
        csv_file = save_dir / f'bottleneck_{device}.csv'
        dt_arr = np.array(dt) * 1E3
        mean = dt_arr.mean(axis=1)
        std = dt_arr.std(axis=1)
        LOGGER.info(f'pre-process: %.1fms , edge model: %.1fms , edge encoder: %.1fms , VVC pre-process: %.1fms , VVC encoding: %.1fms , VVC decoding: %.1fms , read data: %.1fms , cloud decoder: %.1fms , cloud model: %.1fms' % tuple(mean))
        s = '' if csv_file.exists() else (('%20s,' * 11) % ('STAT', 'QP', 'pre-process', 'edge model', 'edge encoder', 'VVC pre-process', 'VVC encoding', 'VVC decoding', 'read data', 'cloud decoder', 'cloud model')) + '\n'
        if save_csv:
            with open( csv_file, 'a') as f: 
                f.write(s + (('%20s,' + '%20.1f,'*10) % ('mean', qp, *mean)) + '\n') 
                f.write((('%20s,' + '%20.1f,'*10) % ('std', qp, *std)) + '\n') 
    else:
        csv_file = save_dir / f'{device}.csv'
        dt_arr = np.array(dt[:8]) * 1E3
        mean = dt_arr.mean(axis=1)
        std = dt_arr.std(axis=1)
        LOGGER.info(f'pre-process: %.1fms , VVC pre-process: %.1fms , JPEG conversion: %.1fms , VVC encoding: %.1fms , VVC decoding: %.1fms , JPEG conversion: %.1fms , read data: %.1fms , cloud model: %.1fms' % tuple(mean))
        s = '' if csv_file.exists() else (('%20s,' * 10) % ('STAT', 'QP', 'pre-process', 'VVC pre-process', 'JPEG conversion', 'VVC encoding', 'VVC decoding', 'JPEG conversion', 'read data', 'cloud model')) + '\n'   
        if save_csv:
            with open( csv_file, 'a') as f: 
                f.write(s + (('%20s,' + '%20.1f,'*9) % ('mean', qp, *mean)) + '\n') 
                f.write((('%20s,' + '%20.1f,'*9) % ('std', qp, *std)) + '\n') 

    return pred


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / '../datasets/kodak/test', help='directory containing the test images')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=None, help='inference size h,w')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-csv', action='store_true', help='save results to *.txt')
    parser.add_argument('--no-vvc', action='store_true', help='do not perform vvc compression')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default=ROOT / 'runs/timing', help='save results to project/name')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--qp', type=int, default=24, help='QP for the vvc encoder')
    parser.add_argument('--chs-in-w', type=int, default=8, help='number of channels is width in the tiled tensor')
    parser.add_argument('--chs-in-h', type=int, default=8, help='number of channels is height in the tiled tensor')
    parser.add_argument('--tensors-min', type=float, default=-1.4, help='the clipping lower bound for the intermediate tensors')
    parser.add_argument('--tensors-max', type=float, default=1.4, help='the clipping upper bound for the intermediate tensors')


    opt = parser.parse_args()
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
