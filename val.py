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

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from models.supplemental import AutoEncoder, Decoder_Rec
from utils.datasets import create_dataloader
from utils.general import (LOGGER, StatCalculator, check_dataset, check_img_size, check_requirements, check_suffix, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, print_args_to_file)
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync
from utils.callbacks import Callbacks
from utils.loss import ComputeRecLoss
from utils.utils import add_noise
from utils.utils import save_one_json, save_one_txt, process_batch



@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=512,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        noise_type=None,
        noise_spot='latent',
        noise_param=1,
        autoenc_chs=None,
        track_stats=False,
        dist_range=[-10,14],
        bins=10000,
        data_suffix='',
        cut_layer=4,
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        compressibility_loss=None,
        compute_rec_loss = None,
        autoencoder=None,
        rec_model = None,
        ):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        print_args_to_file(opt, save_dir / 'result.txt')

        # Load model
        check_suffix(weights, '.pt')
        model = attempt_load(weights, map_location=device)  # load FP32 model
        model.cutting_layer = getattr(model, 'cutting_layer', cut_layer)

        # Loading Autoencoder
        ckpt = torch.load(weights[0], map_location=device)  # load checkpoint
        # ---- Autoencoder -----#
        if 'autoencoder' in ckpt:
            autoenc_chs = ckpt['autoencoder'].chs
            autoencoder = AutoEncoder(autoenc_chs).to(device)
            autoencoder.load_state_dict(ckpt['autoencoder'].state_dict())
            print('pretrained autoencoder')
        # Loading Reconstruction model
        if 'rec_model' in ckpt:
            rec_model = Decoder_Rec(cin=ckpt['rec_model'].cin, cout=ckpt['rec_model'].cout, first_chs=getattr(ckpt['rec_model'], 'first_chs', None) or getattr(ckpt['rec_model'], 'autoenc_chs', None)).to(device)
            rec_model.load_state_dict(ckpt['rec_model'].float().state_dict())
            compute_rec_loss = ComputeRecLoss(MAX=1.0, w_grad=0, compute_grad=False)  # init loss class
            print('pretrained reconstruction model')

        # Data
        data = check_dataset(data, suffix=data_suffix)  # check
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check image size

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

    # Configure
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        pad = 0.0 if task == 'speed' else 0.5
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, single_cls, pad=pad, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    loss_rec = torch.zeros(5, device=device)
    loss_r = torch.zeros(1, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    psnr_mag = 0
    stats_bottleneck = StatCalculator(dist_range, bins, per_chs=True) if track_stats else None

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        t1 = time_sync()
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1
        if noise_spot=='input':
            img = add_noise(img, noise_type=noise_type, noise_param=noise_param)
        # Run model
        T = model(img, augment=augment, cut_model=1)  # first half of the model
        if noise_spot=='latent':
            T = add_noise(T, noise_type=noise_type, noise_param=noise_param)
        T_bottleneck = autoencoder(T, task='enc') if autoencoder is not None else T
        if noise_spot=='bottleneck':
            T_bottleneck = add_noise(T_bottleneck, noise_type=noise_type, noise_param=noise_param)
        T_hat = autoencoder(T, task='dec', bottleneck=T_bottleneck) if autoencoder is not None else T_bottleneck
        out, train_out = model(None, cut_model=2, T=T_hat)  # second half of the model
        if track_stats:
            stats_bottleneck.update_stats(T_bottleneck.detach().clone().cpu().numpy())
        if compressibility_loss:
            loss_r += compressibility_loss(T_bottleneck.float())[1]
        if compute_rec_loss:
            rec_img = rec_model(T_bottleneck).clamp_(0,1)
            loss_rec_tmp, psnr_mag_tmp = compute_rec_loss(img, rec_img)[1:]
            loss_rec += loss_rec_tmp
            psnr_mag += psnr_mag_tmp

        dt[1] += time_sync() - t2

        # Compute loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

        # Run NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t3 = time_sync()
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
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
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, img[si])

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
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

    # Print results
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    with open(save_dir / 'result.txt', 'a') as f:
        f.write('\n\n' + s + '\n')
        f.write(pf % ('all', seen, nt.sum(), mp, mr, map50*100, map*100) + '\n')
    
    if track_stats:
        stats_bottleneck.output_stats(save_dir)
        
    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
            with open(save_dir / 'result.txt', 'a') as f:
                f.write(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]) + '\n')

    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end')

    # Save JSON
    if save_json and len(jdict):
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
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    loss_tot = torch.cat((loss, loss_r, loss_rec))

    csv_file = save_dir.parent / 'results.csv'
    s = '' if csv_file.exists() else (('%20s,' * 5) % ('ID', 'mAP@.5:.95', 'mAP@.5', 'psnr', 'ms-ssim')) + '\n'
    with open( csv_file, 'a') as f: 
        f.write(s + (('%20s,' + '%20.5g,'*4) % (name, map, map50, loss_rec[2]/len(dataloader), loss_rec[4]/len(dataloader))) + '\n') 

    if not training:
        s = ('%11s' * 12) % ('lbox', 'lobj', 'lcls', 'lcmprss', 'rec_l1', 'rec_l2', 'psnr', 'grad_loss', 'ms-ssim', 'gmag_psnr', 'mAP@.5', 'mAP@.5:.95')
        pf = '%11.3g' * 12 + '\n' # print format
        LOGGER.info(s)
        LOGGER.info(pf % (*loss_tot.cpu() / len(dataloader), psnr_mag / len(dataloader), map50, map))

    return (mp, mr, map50, map, *(loss_tot.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    # Object detection arguments
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    # Noise addition arguments
    parser.add_argument('--noise-type', default=None, choices=['uniform', 'gaussian', 'laplacian', 'dropout'], help='type of the added noise. If None, no noise will be added')
    parser.add_argument('--noise-spot', default='latent', choices=['latent', 'bottleneck', 'input'], help='where noise should be applied')
    parser.add_argument('--noise-param', type=float, default=1, help='noise parameter (length for uniform, std for gaussian, lambda for laplacian, prob for dropout)')
    # Supplemental arguments
    parser.add_argument('--track-stats', action='store_true', help='track the statistical properties of the bottleneck')
    parser.add_argument('--dist-range',  type=float, nargs='*', default=[-3,3], help='the range of the distribution')
    parser.add_argument('--bins', type=int, default=1000, help='number of bins in histogram')
    parser.add_argument('--data-suffix', type=str, default='', help='data path suffix')
    parser.add_argument('--cut-layer', type=int, default=4, help='the index of the cutting layer (AFTER this layer, the model will be split)')

    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        run(**vars(opt))

    elif opt.task == 'speed':  # speed benchmarks
        # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=opt.imgsz, conf_thres=.25, iou_thres=.45,
                device=opt.device, save_json=False, plots=False)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                LOGGER.info(f'\nRunning {f} point {i}...')
                r, _, t = run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=i, conf_thres=opt.conf_thres,
                              iou_thres=opt.iou_thres, device=opt.device, save_json=opt.save_json, plots=False)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
