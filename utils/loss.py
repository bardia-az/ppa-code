# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
from torch.cuda import device
import torch.nn as nn
import numpy as np
import math
from scipy.linalg import toeplitz
from collections import defaultdict
import warnings

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel
from torch.nn.functional import mse_loss, l1_loss
from pytorch_msssim import ms_ssim


def gradient_img(img):
    device = img.device
    dtype = img.dtype
    # Y = 0.2126 R + 0.7152 G + 0.0722 B
    Y = (0.2126*img[:,0,:,:] + 0.7152*img[:,1,:,:] + 0.0722*img[:,2,:,:]).unsqueeze(1)

    Sx=np.array([[1, 0, -1],[2,0,-2],[1,0,-1]])
    kx = torch.from_numpy(Sx).to(device, dtype=dtype, non_blocking=True).unsqueeze(0).unsqueeze(0)
    G_x = nn.functional.conv2d(Y, kx, padding=1)

    Sy=np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])
    ky = torch.from_numpy(Sy).to(device, dtype=dtype, non_blocking=True).unsqueeze(0).unsqueeze(0)
    G_y = nn.functional.conv2d(Y, ky, padding=1)

    G=torch.cat((G_x, G_y), dim=1)

    G_mag = (G_x.square() + G_y.square()).sqrt()

    return G , G_mag

def img_gradient_loss(pred, gt, MAX):
    G_pred, G_pred_mag = gradient_img(pred)
    G_gt, G_gt_mag = gradient_img(gt)
    loss = l1_loss(G_pred, G_gt)
    loss_mag = mse_loss(G_pred_mag, G_gt_mag, reduction='none').mean((2,3))
    psnr_per_feature = 10 * torch.log10(MAX**2 / loss_mag)
    psnr_mag = torch.mean(psnr_per_feature)
    return loss, psnr_mag


class CompressibilityLoss:
    def __init__(self, device, weight=0.0):
        self.device = device
        self.weight = weight

    def __call__(self, features):
        Z = self.entropy_friendly_loss_made_differentiable(features)
        loss_items = torch.zeros(1, device=self.device)
        loss_items[0] = self.DCT_rate(Z)
        loss = self.weight * loss_items
        return loss, loss_items.detach()


    def DCT_rate(self, feature):    
        dct_feature = self.batch_DCT_2D(feature) 
        energy_dct_feature = torch.mean(torch.abs(dct_feature))
        return energy_dct_feature

    def batch_DCT_2D(self, feature):
        shape_of_feature = feature.shape
        row = shape_of_feature[2] 
        col = shape_of_feature[3]
        feature_permute = feature.permute(0,1,3,2)

        dctm_c = torch.Tensor(self.dctmat(col)).to(self.device)
        dctm_r = torch.Tensor(self.dctmat(row)).to(self.device)
        
        mult1_v1 = torch.bmm(dctm_c.expand(shape_of_feature[0]*shape_of_feature[1],dctm_c.shape[0],dctm_c.shape[1]),
                        feature_permute.contiguous().view(shape_of_feature[0]*shape_of_feature[1],shape_of_feature[3],shape_of_feature[2]))
        mult1_v2 = mult1_v1.permute(0,2,1)
        mult2 = torch.bmm(dctm_r.expand(shape_of_feature[0]*shape_of_feature[1],dctm_r.shape[0],dctm_r.shape[1]),
                        mult1_v2)
        return mult2

    def dctmat(self, n):
        x = np.arange(n)
        y = np.arange(n)
        cc, rr = np.meshgrid(x, y)
        c = math.sqrt(2 / n) * np.cos(np.multiply(math.pi * (2*cc + 1), rr) / (2 * n))
        c[0,:] = c[0,:] / math.sqrt(2)
        return c

    def DCT_tiled(self, feature):
        tiled_image = self.change_to_filed(feature)
        shape_tiled_image = tiled_image.shape
        row = shape_tiled_image[1] 
        col = shape_tiled_image[2]
        tiled_image_permute = tiled_image.permute(0,2,1)

        dctm_c = torch.Tensor(self.dctmat(col)).to(self.device)
        dctm_r = torch.Tensor(self.dctmat(row)).to(self.device)
        
        mult1_v1 = torch.bmm(dctm_c.expand(shape_tiled_image[0],dctm_c.shape[0],dctm_c.shape[1]),tiled_image_permute)
        mult1_v2 = mult1_v1.permute(0,2,1)
        mult2 = torch.bmm(dctm_r.expand(shape_tiled_image[0],dctm_r.shape[0],dctm_r.shape[1]),mult1_v2)
        return mult2

    def change_to_filed(self, feature):
        shape_feature = feature.shape
        fch = shape_feature[1]
        frow = shape_feature[2]
        fcol = shape_feature[3]
        num_ch_in_row = int(2** np.floor(1/2* (np.log2(fch))))
        num_ch_in_col  = int(2** np.ceil (1/2* (np.log2(fch))))
        feature_unfold_permuted = feature.unfold(1,num_ch_in_col,num_ch_in_col)
        feature_unfold = feature_unfold_permuted.permute(0,1,2,4,3)
        tiled_image = feature_unfold.contiguous().view(-1,num_ch_in_row*frow,num_ch_in_col*fcol)
        return tiled_image

    def entropy_friendly_loss_made_differentiable(self, feature):
        bit_number = 8
        shape_of_feature = feature.shape
        column_toeplitz = np.zeros((1,shape_of_feature[3]))
        column_toeplitz[0][0]=1
        row_toeplitz = np.zeros((1,shape_of_feature[3]))
        row_toeplitz[0][0]=1
        row_toeplitz[0][1]=-1
        toeplitz_matrix = toeplitz(column_toeplitz,row_toeplitz)

        toeplitz_matrix = torch.from_numpy(toeplitz_matrix).to(self.device)
        
        toeplitz_matrix = toeplitz_matrix.float()

        #transposed toeplitz
        column_toeplitz_t = np.zeros((1,shape_of_feature[2]))
        column_toeplitz_t[0][0]=1
        row_toeplitz_t = np.zeros((1,shape_of_feature[2]))
        row_toeplitz_t[0][0]=1
        row_toeplitz_t[0][1]=-1
        toeplitz_matrix_t = toeplitz(column_toeplitz_t,row_toeplitz_t)
        
        toeplitz_matrix_t = torch.from_numpy(toeplitz_matrix_t).to(self.device)

        toeplitz_matrix_t = toeplitz_matrix_t.float()
        
        G_x_1 = torch.bmm(feature.view((shape_of_feature[0]*shape_of_feature[1],shape_of_feature[2],shape_of_feature[3])),
                        toeplitz_matrix.unsqueeze(0).expand(feature.size(0)*feature.size(1), *toeplitz_matrix.size()))

        G_x_2 = G_x_1.view((shape_of_feature[0],shape_of_feature[1],shape_of_feature[2],shape_of_feature[3]))
        temp_feature = feature.permute(0,1,3,2)

        G_y_1 = torch.bmm(temp_feature.contiguous().view((temp_feature.size(0) * temp_feature.size(1), temp_feature.size(2), temp_feature.size(3))),
                        toeplitz_matrix_t.unsqueeze(0).expand(feature.size(0)*feature.size(1), *toeplitz_matrix_t.size()))

        G_y_2 = G_y_1.view((feature.size(0), feature.size(1), feature.size(3), feature.size(2)))
        G_y_3 = G_y_2.permute(0,1,3,2)

        scale = torch.Tensor([0.5]).to(self.device)
        G_t  = (G_x_2+G_y_3) * scale.expand_as(G_x_2+G_y_3)
        return G_t

class ComputeRecLoss:
    def __init__(self, MAX=1.0, w_grad=0.0, compute_grad=True, type='L1', K_ssim=(0.01, 0.4), win_ssim=11):
        self.MAX = MAX
        self.w_grad = w_grad
        self.compute_grad = compute_grad
        self.type = type
        self.K_ssim = K_ssim
        self.win_ssim = win_ssim
        self.size_warn = False
    def __call__(self, T1, T2):
        device = T1.device
        loss_l1, loss_l2, psnr, grad_loss, psnr_mag, loss_msssim = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        loss_l1[0] = l1_loss(T1, T2)
        mse = mse_loss(T1, T2, reduction='none').mean((2,3))
        loss_l2[0] = torch.mean(mse)
        psnr_per_feature = 10 * torch.log10(self.MAX**2 / mse)
        psnr[0] = torch.mean(psnr_per_feature)
        if T1.shape[-2]>160 and T1.shape[-1]>160:
            loss_msssim[0] = ms_ssim(T1, T2, data_range=self.MAX, K=self.K_ssim, win_size=self.win_ssim)
        elif not self.size_warn:
            self.size_warn = True 
            warnings.warn("The image_size of the current batch results in wrong ms-ssim calculations (dimensions should be greater than 160). The ms-ssim values are ignored for the current and possibly next batches", RuntimeWarning)
        if self.compute_grad:
            grad_loss[0], psnr_mag = img_gradient_loss(T1, T2, self.MAX*4)
        
        if self.type=='L1':
            loss = loss_l1 + self.w_grad * grad_loss
        elif self.type=='L2':
            loss = loss_l2 + self.w_grad * grad_loss
        elif self.type=='MS-SSIM':
            loss = (1 - loss_msssim) + self.w_grad * grad_loss
        else:
            raise Exception('Supported reconstruction losses are "L1", "L2", "MS-SSIM", but got {}'.format(self.type))

        return loss, torch.cat((loss_l1, loss_l2, psnr, grad_loss, loss_msssim)).detach(), psnr_mag.detach()


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
