# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
from enum import auto
import json
import os
import sys
from pathlib import Path
from threading import Thread
import yaml

import numpy as np
import torch
from tqdm import tqdm
import subprocess
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 200

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from models.supplemental import AutoEncoder, Decoder_Rec
from utils.datasets import create_dataloader
from utils.general import (LOGGER, StatCalculator, check_file, box_iou, check_dataset, check_img_size, check_requirements, check_suffix, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh, print_args_to_file)
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync
from utils.callbacks import Callbacks
from utils.loss import ComputeRecLoss


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


def read_y_channel(file, w, h):
    raw = file.read(w*h)
    raw = np.frombuffer(raw, dtype=np.uint8)
    raw = raw.reshape((h,w))
    return raw

def get_tensors(img, model, autoencoder):
    T = model(img, cut_model=1)  # first half of the model
    if autoencoder is not None:
        T_bottleneck = autoencoder(T, task='enc')
    else:
        T_bottleneck = T
    return T_bottleneck

def tensors_to_tiled(tensor, chs_in_w, chs_in_h, tensors_min, tensors_max):
    shape = tensor.shape
    assert shape[1] == chs_in_w*chs_in_h, 'chs_in_w & chs_in_h are not valid values based on the latent tensors'
    tensor = torch.clamp(tensor, tensors_min, tensors_max)
    tensor = torch.reshape(tensor, (chs_in_h, chs_in_w, shape[2], shape[3]))
    tensor = torch.round((tensor - tensors_min) * 255 / (tensors_max - tensors_min))
    return tensor.swapaxes(1,2).reshape((chs_in_h*shape[2], chs_in_w*shape[3]))

def tiled_to_tensor(tiled, ch_w, ch_h, tensors_min, tensors_max):
    shape = tiled.shape
    ch_num_w, ch_num_h = shape[1] // ch_w, shape[0] // ch_h
    tiled = torch.reshape(tiled, (ch_num_h, ch_h, ch_num_w, ch_w))
    tiled = tiled / 255 * (tensors_max - tensors_min) + tensors_min
    tensor = tiled.swapaxes(1,2).reshape((-1, ch_h, ch_w))
    return tensor

def compress_tensors(tensors, tensors_w, tensors_h, qp):
    data = tensors.cpu().numpy().flatten().astype(np.uint8)
    vvc_report = Path('../vvc/vvc_report2.txt').open('w')
    VVC_command = ['../vvc/vvencFFapp', '-c', '../vvc/lowdelay_faster_latent.cfg', '-i', '../vvc/tiled_img.yuv', '-b', '../vvc/bitstream2.bin', 
                   '-o', '../vvc/reconst2.yuv', '--SourceWidth', str(tensors_w), '--SourceHeight', str(tensors_h), '-f', '1', '-fr', '1', '-q', str(qp)]
    to_be_coded_file = '../vvc/tiled_img.yuv'
    with open(to_be_coded_file, 'wb') as f:
        f.write(data)
    subprocess.call(VVC_command, stdout=vvc_report)
    vvc_report.close()
    bpp = os.path.getsize('../vvc/bitstream2.bin') * 8 / (tensors_w*tensors_h)
    KB_num = os.path.getsize('../vvc/bitstream2.bin') / 1024.0
    with open('../vvc/reconst2.yuv', 'rb') as f:
        tmp_reconst = read_y_channel(f, tensors_w, tensors_h)
    return torch.from_numpy(tmp_reconst.copy()).to(tensors.device, non_blocking=True), bpp, KB_num

def compress_input(img, qp, half=False):
    import cv2
    h, w = img.shape[-2:]
    jpg2yuv_report = Path('../vvc/jpg2yuv_report.txt').open('w')
    vvc_report = Path('../vvc/vvc_report.txt').open('w')
    yuv2png_report = Path('../vvc/yuv2png_report.txt').open('w')
    jpg2yuv_command = ['ffmpeg', '-i', '../vvc/image.png', '-f', 'rawvideo', '-pix_fmt', 'yuv444p', '-dst_range', '1', '../vvc/yuv_img.yuv', '-y']
    subprocess.call(jpg2yuv_command, stdout=jpg2yuv_report, stderr=subprocess.STDOUT)
    VVC_command = ['../vvc/vvencFFapp', '-c', '../vvc/lowdelay_faster_444.cfg', '-i', '../vvc/yuv_img.yuv', '-b', '../vvc/bitstream.bin', 
                   '-o', '../vvc/reconst.yuv', '--SourceWidth', str(w), '--SourceHeight', str(h), '-f', '1', '-fr', '1', '-q', str(qp)]
    subprocess.call(VVC_command, stdout=vvc_report)
    bpp = os.path.getsize('../vvc/bitstream.bin') * 8 / (w*h)
    KB_num = os.path.getsize('../vvc/bitstream.bin') / 1024.0
    yuv2png_command = ['ffmpeg', '-f', 'rawvideo', '-pix_fmt', 'yuv444p', '-s', f"{w}x{h}", '-src_range', '1', '-i', '../vvc/reconst.yuv',
                       '-frames', '1', '-pix_fmt', 'rgb24', '../vvc/output.png', '-y']
    subprocess.call(yuv2png_command, stdout=yuv2png_report, stderr=subprocess.STDOUT)
    jpg2yuv_report.close()
    vvc_report.close()
    yuv2png_report.close()
    rec = cv2.imread('../vvc/output.png')  # BGR
    rec = rec.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    rec = np.ascontiguousarray(rec)
    rec = torch.from_numpy(rec).to(img.device, non_blocking=True)
    rec = rec.half() if half else rec.float()  # uint8 to fp16/32
    rec /= 255  # 0 - 255 to 0.0 - 1.0
    return rec.unsqueeze(0), bpp, KB_num

def get_yolo_prediction(T, autoencoder, model):
    if autoencoder is not None:
        T_hat = autoencoder(T, task='dec', bottleneck=T)
        out, _ = model(None, cut_model=2, T=T_hat)  # second half of the model
    else:
        out, _ = model(None, cut_model=2, T=T)  # second half of the model
    return out



@torch.no_grad()
def val_closed_loop(opt,
                    callbacks=Callbacks()):

    save_dir, batch_size, weights, single_cls, data, half, plots, tensors_min, \
        tensors_max, track_stats, chs_in_w, chs_in_h, compression, qp = \
        Path(opt.save_dir), opt.batch_size, opt.weights, opt.single_cls, opt.data, opt.half, False, \
        opt.tensors_min, opt.tensors_max, opt.track_stats, opt.chs_in_w, opt.chs_in_h, opt.compression, opt.qp

    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    device = select_device(opt.device, batch_size=batch_size)

    # Load model
    check_suffix(weights, '.pt')
    model = attempt_load(weights, map_location=device)  # load FP32 model
    model.cutting_layer = model.cutting_layer if hasattr(model, 'cutting_layer') else opt.cut_layer
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, s=gs)  # check image size

    # Loading Autoencoder
    ckpt = torch.load(weights, map_location=device)  # load checkpoint
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
    
    # Data
    data = check_dataset(data)  # check

    # Half
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    model.half() if half else model.float()

    # Configure
    model.eval()
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    pad = 0.0
    task = 'val'
    dataloader = create_dataloader(data[task], imgsz, batch_size, gs, single_cls, pad=pad, rect=True, rect_img=False, store_img=(compression=='input'),
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []

    stats_error = StatCalculator(opt.dist_range, opt.bins) if track_stats else None
    compute_rec_loss = ComputeRecLoss(MAX=1.0, w_grad=0, compute_grad=True)  # init loss class

    ch_num = chs_in_w * chs_in_h
    bpp_seq = []
    KB_seq = []
    psnr_seq = []
    psnr_mag_seq = []

    if compression=='bottleneck' and (tensors_min is None or tensors_max is None):
        stats_bottleneck = StatCalculator((-3,3), 1000, per_chs=False)
        for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
            t1 = time_sync()
            img = img.to(device, non_blocking=True)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255  # 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)
            nb, _, height, width = img.shape  # batch size, channels, height, width
            t2 = time_sync()
            dt[0] += t2 - t1
            # Run model
            T = model(img, cut_model=1)  # first half of the model
            T_bottleneck = autoencoder(T, task='enc') if autoencoder is not None else T
            stats_bottleneck.update_stats(T_bottleneck.detach().clone().cpu().numpy())
        tensors_min, tensors_max = -6*stats_bottleneck.std, 6*stats_bottleneck.std
        print(f'clipping range: [{tensors_min:.4f}, {tensors_max:.4f}]')

    for fr, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        t1 = time_sync()
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1
        ch_h, ch_w = img.shape[-2:]
        ch_w //= 8     # beware that the stride is 8 at layers 3 and 4
        ch_h //= 8     # beware that the stride is 8 at layers 3 and 4
        
        if compression == 'input':
            rec_img, bpp, KB_num = compress_input(img, qp, half)
            bpp_seq.append(bpp)
            KB_seq.append(KB_num)
            out, _ = model(rec_img)  # inference and training outputs
            loss_rec, psnr_mag = compute_rec_loss(img, rec_img)[1:]
            psnr = loss_rec[2]
            psnr_seq.append(psnr)
            psnr_mag_seq.append(psnr_mag)
        elif compression == 'bottleneck':
            tensors = get_tensors(img, model, autoencoder)
            tiled_tensors = tensors_to_tiled(tensors, chs_in_w, chs_in_h, tensors_min, tensors_max)
            rec_tensors, bpp, KB_num = compress_tensors(tiled_tensors, ch_w*chs_in_w, ch_h*chs_in_h, qp)
            bpp_seq.append(bpp)
            KB_seq.append(KB_num)
            rec_tensors = rec_tensors.half() if half else rec_tensors.float()
            rec_tensors = tiled_to_tensor(rec_tensors, ch_w, ch_h, tensors_min, tensors_max)
            if rec_model is not None:
                rec_img = rec_model(rec_tensors[None, :])
                loss_rec, psnr_mag = compute_rec_loss(img, rec_img)[1:]
                psnr = loss_rec[2]
                psnr_seq.append(psnr)
                psnr_mag_seq.append(psnr_mag)
            out = get_yolo_prediction(rec_tensors[None, :], autoencoder, model)
            error = rec_tensors - tensors[0]
            if track_stats:
                stats_error.update_stats(error.detach().clone().cpu().numpy())

        dt[1] += time_sync() - t2

        # Run NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if opt.save_hybrid else []  # for autolabelling
        t3 = time_sync()
        out = non_max_suppression(out, opt.conf_thres, opt.iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        dt[2] += time_sync() - t3

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

            # Save/log
            if opt.save_txt:
                save_one_txt(predn, opt.save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            if opt.save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, img[si])

        # Plot images
        if plots and fr < 3:
            f = save_dir / f'val_batch{fr}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'val_batch{fr}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()
    
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    bpp_tot = sum(bpp_seq) / len(dataloader)
    MB_tot = sum(KB_seq) / 1024.0
    psnr_tot = sum(psnr_seq) / len(dataloader)
    psnr_mag_tot = sum(psnr_mag_seq) / len(dataloader)

    # Print results   
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    # Save results
    file = (save_dir / 'result').with_suffix('.csv')
    s = '' if file.exists() else (('%20s,' * 9) % ('QP', 'bpp', 'MB_tot', 'psnr', 'psnr_grad', 'mAP@.5 (%)', 'mAP@.5:.95 (%)', 'P', 'R')) + '\n'  # add header
    with open(file, 'a') as f:
        f.write(s + (('%20.5g,' * 9) % (qp, bpp_tot, MB_tot, psnr_tot, psnr_mag_tot, map50*100, map*100, mp, mr)) + '\n')
    
    if track_stats:
        stats_error.output_stats(save_dir, f'qp{qp}')
        
    # Print results per class
    if (opt.verbose or nc<50) and nc > 1 and len(stats):
        file = (save_dir / f'res_per_class_qp{qp}').with_suffix('.csv')
        s = '' if file.exists() else (('%20s,' * 8) % ('QP', 'Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')) + '\n'  # add header
        with open(file, 'a') as f:
            f.write(s + (('%20.5g,' + '%20s,' + '%20.5g,' * 6) % (qp, 'all', seen, nt.sum(), mp, mr, map50, map)) + '\n')
            for i, c in enumerate(ap_class):
                LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
                f.write(('%20.5g,' + '%20s,' + '%20.5g,' * 6) % (qp, names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]) + '\n')

    print(f'\nbit per pixel <qp{qp}> = {bpp_tot:.2f}')
    print(f'total size of the images (MB) <qp{qp}> = {MB_tot:.2f}')
    print(f'mean PSNR (dB) <qp{qp}> = {psnr_tot:.2f}')
    print(f'mean PSNR_Grad (dB) <qp{qp}> = {psnr_mag_tot:.2f}')

    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    shape = (batch_size, 3, imgsz, imgsz)
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end')

    # Save JSON
    if opt.save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
            # Save results
            file = (save_dir / 'result').with_suffix('.csv')
            s = '' if file.exists() else (('%20s,' * 7) % ('QP', 'bpp', 'MB_tot', 'psnr', 'psnr_grad', 'mAP@.5 (%)', 'mAP@.5:.95 (%)')) + '\n'  # add header
            with open(file, 'a') as f:
                f.write(s + (('%20.5g,' * 7) % (qp, bpp_tot, MB_tot, psnr_tot, psnr_mag_tot, map50*100, map*100)) + '\n')
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if opt.save_txt else ''
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}\n")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/img_compression', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')

    # Supplemental arguments
    parser.add_argument('--track-stats', action='store_true', help='track the statistical properties of the error')
    parser.add_argument('--dist-range',  type=float, nargs='*', default=[-4,4], help='the range of the distribution')
    parser.add_argument('--bins', type=int, default=10000, help='number of bins in histogram')
    parser.add_argument('--qp', type=int, default=24, help='QP for the vvc encoder')
    parser.add_argument('--chs-in-w', type=int, default=8, help='number of channels is width in the tiled tensor')
    parser.add_argument('--chs-in-h', type=int, default=8, help='number of channels is height in the tiled tensor')
    # parser.add_argument('--tensors-min', type=float, default=-0.2786, help='the clipping lower bound for the intermediate tensors')
    # parser.add_argument('--tensors-max', type=float, default=1.4, help='the clipping upper bound for the intermediate tensors')
    parser.add_argument('--tensors-min', type=float, default=None, help='the clipping lower bound for the intermediate tensors')
    parser.add_argument('--tensors-max', type=float, default=None, help='the clipping upper bound for the intermediate tensors')
    parser.add_argument('--compression', type=str, default='input', help='compress input or latent space ("input", "bottleneck")')
    parser.add_argument('--cut-layer', type=int, default=-1, help='the index of the cutting layer (AFTER this layer, the model will be split)')

    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    # Directories
    opt.data, opt.weights, opt.project = check_file(opt.data), str(opt.weights), str(opt.project)  # checks
    save_dir = (Path(opt.project) / opt.name)  # increment run
    (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    opt.save_dir = str(save_dir)

    val_closed_loop(opt)


def run(**kwargs):
    # Usage: import val_closed_loop; val_closed_loop.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
