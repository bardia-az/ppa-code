# supplementary DNN modules

import sys
from pathlib import Path
from turtle import forward

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.common import *
from utils.plots import feature_visualization, visualize_one_channel, motion_field_visualization
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import itertools

################ Input Reconstruction Decoder #######################
class Bottleneck_Inv(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=1.0):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c1 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 3, 1, g=g)
        self.cv2 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3_Inv(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c1 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * c_, 1)  # act=FReLU(c2)
        self.cv2 = Conv(c_, c2, 1, 1)
        self.cv3 = Conv(c_, c2, 1, 1)  
        self.m = nn.Sequential(*(Bottleneck_Inv(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        # return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        y = self.cv1(x)
        y1, y2 = torch.chunk(y, 2, dim=1)
        return self.cv2(y2) + self.cv3(self.m(y1))

class Conv_Inv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True) -> None:
        super().__init__()
        op = s-1 if (k % 2) == 1 else s-2
        self.deconv = nn.ConvTranspose2d(c1, c2, k, s, padding=autopad(k, p), output_padding=op, groups=g)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.deconv(x)))

class Decoder_Rec(nn.Module):
    def __init__(self, cin, cout, first_chs, e=0.5):
        super().__init__()
        self.first_chs = first_chs
        self.cin = cin
        self.cout = cout
        self.e = e
        c1 = int(cin * e)
        c2 = int(c1 * e)
        layers = [
            Decoder(first_chs, k=3, s=1),
            C3_Inv(cin, cin, n=4),
            Conv_Inv(cin, c1, k=3, s=2),
            C3_Inv(c1, c1, n=2),
            Conv_Inv(c1, c2, k=3, s=2),
            Conv_Inv(c2, cout, k=6, s=2, p=2),
        ]
        LOGGER.info(f"\n{'':>3}{'params':>10}  {'module':<40}")
        for i, layer in enumerate(layers):
            np = sum(x.numel() for x in layer.parameters())  # number params
            LOGGER.info(f'{i:>3}{np:10.0f}  {str(type(layer))[8:-2]:<40}')  # print
        self.net = nn.Sequential(*(layers))

    def forward(self,x):
        return self.net(x)

        # Conv([3, 48, 6, 2, 2])
        # Conv([48, 96, 3, 2])
        # C3([96, 96, 2])
        # Conv([96, 192, 3, 2])
        # C3([192, 192, 4])

################## Auto Encoder ######################
# class ResBlock(nn.Module):
#     def __init__(self, ch, k=3, p=None):
#         super().__init__()
#         self.Conv1 = nn.Conv2d(ch, ch, k, s=1, padding=autopad(k, p))
#         self.Conv2 = nn.Conv2d(ch, ch, k, s=1, padding=autopad(k, p))
#         self.act = nn.SiLU()

#     def forward(self, x):
#         y = x
#         return (x + self.Conv2(self.act(self.Conv1(y))))

class Encoder(nn.Module):
    def __init__(self, chs, k=1, s=1, p=None):
        super().__init__()
        layers = [nn.Identity()]
        for i in range(len(chs)-1):
            if i == len(chs) - 2:
                layers.append(ResBlock(chs[i], k, act=nn.SiLU()))
                nn.BatchNorm2d(chs[i])
            layers.append(nn.Conv2d(chs[i], chs[i+1], k, s, autopad(k, p), bias=False))
            if i < len(chs) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*(layers))

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, chs, k=1, s=1, p=None):
        super().__init__()
        layers = [nn.Identity()]
        for i in range(len(chs)-1, 0, -1):
            if i == 1:
                layers.append(ResBlock(chs[i], k, act=nn.SiLU()))
                nn.BatchNorm2d(chs[i])
            layers.append(nn.Conv2d(chs[i], chs[i-1], k, s, autopad(k, p), bias=False))
            layers.append(nn.SiLU())
        self.net = nn.Sequential(*(layers))

    def forward(self, x):
        return self.net(x)


class AutoEncoder(nn.Module):
    # def __init__(self, cin, cmid):
    def __init__(self, chs):
        super().__init__()
        # print(chs)
        self.chs = chs
        self.enc = Encoder(chs, k=3)
        self.dec = Decoder(chs, k=3)

    def forward(self, x, visualize=False, task='enc_dec', bottleneck=None, s=''):
        if task=='dec':
            x = bottleneck
        else:
            x = self.enc(x)

        if visualize:
            visualize_one_channel(x, n=8, save_dir=visualize, s=s)
            feature_visualization(x, 'encoder', '', save_dir=visualize, cut_model=s)
            
        if task=='enc':
            return x
        else:
            return self.dec(x)


################## Model-1 ######################
class InterPrediction_1(nn.Module):
    def __init__(self, in_channels = 2):
        super().__init__()

        self.c = in_channels

        def EstimateOffsets(numInputCh, numMidCh, numOutCh):
            return nn.Sequential(
                nn.Conv2d(in_channels=numInputCh,  out_channels=numMidCh, kernel_size=5, stride=1, padding=2),
                nn.Conv2d(in_channels=numMidCh,  out_channels=numOutCh , kernel_size=5, stride=1, padding=2),
            )

        def ConvBasic(numInputCh, numMidCh, numOutCh):
            return nn.Sequential(
                nn.Conv2d(in_channels=numInputCh,  out_channels=numMidCh, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(in_channels=numMidCh,  out_channels=numOutCh , kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
            )

        def Upsampling(intInput, kSize = 3):
            padding = (kSize - 1)//2
            return nn.Sequential(
                nn.Conv2d(in_channels=intInput, out_channels=intInput, kernel_size=kSize, stride=1, padding=padding),
                nn.SiLU(),
            )
        # end

        # downward path
        self.EstOff1_d    = EstimateOffsets(numInputCh=in_channels*2, numMidCh=in_channels*9, numOutCh=in_channels*9*2)
        for v in self.EstOff1_d.modules():
            if isinstance(v, nn.Conv2d) :
                init.constant_(v.weight, 0.)
        self.MoCmpns1_d   = torchvision.ops.DeformConv2d(in_channels=in_channels*2, out_channels=in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.Conv1_d      = ConvBasic(numInputCh=in_channels, numMidCh=in_channels*2, numOutCh=in_channels*4)

        self.Pool1        = nn.AvgPool2d(kernel_size=2, stride=2)

        self.EstOff2_d    = EstimateOffsets(numInputCh=in_channels*4, numMidCh=in_channels*9, numOutCh=in_channels*9*2)
        for v in self.EstOff2_d.modules():
            if isinstance(v, nn.Conv2d) :
                init.constant_(v.weight, 0.)
        self.MoCmpns2_d   = torchvision.ops.DeformConv2d(in_channels=in_channels*4, out_channels=in_channels*4, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.Conv2_d      = ConvBasic(numInputCh=in_channels*4, numMidCh=in_channels*6, numOutCh=in_channels*8)

        self.Pool2        = nn.AvgPool2d(kernel_size=2, stride=2)

        # bottom
        self.EstOff3_d    = EstimateOffsets(numInputCh=in_channels*8, numMidCh=in_channels*9, numOutCh=in_channels*9*2)
        for v in self.EstOff3_d.modules():
            if isinstance(v, nn.Conv2d) :
                init.constant_(v.weight, 0.)
        self.MoCmpns3_d   = torchvision.ops.DeformConv2d(in_channels=in_channels*8, out_channels=in_channels*8, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.Conv3_d      = ConvBasic(numInputCh=in_channels*8, numMidCh=in_channels*7, numOutCh=in_channels*6)

        # upward path
        self.Upsample2    = Upsampling(intInput=in_channels*6, kSize=3)

        self.Conv2_u      = ConvBasic(numInputCh=in_channels*14, numMidCh=in_channels*8, numOutCh=in_channels*4)

        self.Upsample1    = Upsampling(intInput=in_channels*4, kSize=3)

        self.Conv1_u      = ConvBasic(numInputCh=in_channels*8, numMidCh=in_channels*4, numOutCh=in_channels)


    def forward(self, x1, x2, visualize=False):
        x = torch.cat((x1, x2), 1)
        off1_d      = self.EstOff1_d(x)
        compns1_d   = self.MoCmpns1_d(input=x, offset=off1_d)
        conv1_d     = self.Conv1_d(compns1_d)

        pool1       = self.Pool1(conv1_d)

        off2_d      = self.EstOff2_d(pool1)
        compns2_d   = self.MoCmpns2_d(input=pool1, offset=off2_d)
        conv2_d     = self.Conv2_d(compns2_d)

        pool2       = self.Pool2(conv2_d)


        off3_d      = self.EstOff3_d(pool2)
        compns3_d   = self.MoCmpns3_d(input=pool2, offset=off3_d)
        conv3_d     = self.Conv3_d(compns3_d)


        usmpl2      = self.Upsample2(F.interpolate(conv3_d, scale_factor=2.0, mode='bilinear', align_corners=True))

        u2_in       = torch.cat((usmpl2, conv2_d), 1)
        conv2_u     = self.Conv2_u(u2_in)
    

        usmpl1      = self.Upsample1(F.interpolate(conv2_u, scale_factor=2.0, mode='bilinear', align_corners=True))

        u1_in       = torch.cat((usmpl1, conv1_d), 1)
        conv1_u     = self.Conv1_u(u1_in)

        if visualize:
            motion_field_visualization(off1_d, g=self.c, g_h=8, save_dir=visualize, l=1)
            motion_field_visualization(off2_d, g=self.c, g_h=8, save_dir=visualize, l=2)
            motion_field_visualization(off3_d, g=self.c, g_h=8, save_dir=visualize, l=3)

        return conv1_u

################## Model-2 ######################
class InterPrediction_2(nn.Module):
    def __init__(self, in_channels=2, G=1):     # number of input channels, number of Groups in doform_conv
        super().__init__()

        def Conv2x(numInputCh, numMidCh, numOutCh, k=3):
            return nn.Sequential(
                nn.Conv2d(in_channels=numInputCh,  out_channels=numMidCh, kernel_size=k, stride=1, padding=k//2),
                nn.Conv2d(in_channels=numMidCh,  out_channels=numOutCh , kernel_size=k, stride=1, padding=k//2),
            )
        def Conv1x(numInputCh, numOutCh, k=3):
            return nn.Conv2d(in_channels=numInputCh,  out_channels=numOutCh, kernel_size=k, stride=1, padding=k//2)

        def ConvStandard(numInputCh, numOutCh, k=3):
            return nn.Sequential(
                nn.Conv2d(in_channels=numInputCh,  out_channels=numOutCh, kernel_size=k, stride=1, padding=k//2),
                nn.BatchNorm2d(numOutCh),
                nn.SiLU(),
            )

        def ResBlock(ch, k=3):
            return nn.Sequential(
                nn.Conv2d(in_channels=ch,  out_channels=ch, kernel_size=k, stride=1, padding=k//2),
                nn.SiLU(),
                nn.Conv2d(in_channels=ch,  out_channels=ch, kernel_size=k, stride=1, padding=k//2),
            )

        def UpSampling(numInputCh, k=3):
            return nn.Sequential(
                nn.Conv2d(in_channels=numInputCh, out_channels=numInputCh, kernel_size=k, stride=1, padding=k//2),
                nn.SiLU(),
            )

        self.g = G
        c = in_channels
        #--- Motion Estimation ---#
        # Layer1
        self.GetMasterMotion1 = ConvStandard(numInputCh=2*c, numOutCh=2*c)
        self.GetMasterMotion_Layer1 = Conv2x(numInputCh=2*c, numMidCh=2*c, numOutCh=2*c)
        self.GetMotion_Layer1_1 = Conv1x(numInputCh=2*c, numOutCh=2*9*G)
        self.GetMotion_Layer1_2 = Conv1x(numInputCh=2*c, numOutCh=2*9*G)
        # Layer2
        self.Motion_DownSample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.GetMasterMotion2 = ConvStandard(numInputCh=2*c, numOutCh=4*c)
        self.GetMasterMotion_Layer2 = Conv2x(numInputCh=4*c, numMidCh=4*c, numOutCh=4*c)
        self.GetMotion_Layer2_1 = Conv1x(numInputCh=4*c, numOutCh=2*9*(2*G))
        self.GetMotion_Layer2_2 = Conv1x(numInputCh=4*c, numOutCh=2*9*(2*G))
        me_modules = itertools.chain(self.GetMasterMotion1.modules(), self.GetMasterMotion_Layer1.modules(), self.GetMotion_Layer1_1.modules(), self.GetMotion_Layer1_2.modules(),\
                                     self.GetMasterMotion2.modules(), self.GetMasterMotion_Layer2.modules(), self.GetMotion_Layer2_1.modules(), self.GetMotion_Layer2_2.modules())
        for v in me_modules:
            if isinstance(v, nn.Conv2d) :
                init.zeros_(v.weight)

        #--- Motion Compensation ---#
        # Layer1 (down-scale)
        self.DeformConv_Layer1_1 = torchvision.ops.DeformConv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, groups=G, bias=False)
        self.DeformConv_Layer1_2 = torchvision.ops.DeformConv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, groups=G, bias=False)
        self.Conv_Layer1_1 = ConvStandard(numInputCh=c, numOutCh=2*c)
        self.Conv_Layer1_2 = ConvStandard(numInputCh=c, numOutCh=2*c)
        self.DownSample_Layer1_1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.DownSample_Layer1_2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # Layer2 (down-scale)
        self.DeformConv_Layer2_1 = torchvision.ops.DeformConv2d(in_channels=2*c, out_channels=2*c, kernel_size=3, padding=1, groups=G, bias=False)
        self.DeformConv_Layer2_2 = torchvision.ops.DeformConv2d(in_channels=2*c, out_channels=2*c, kernel_size=3, padding=1, groups=G, bias=False)
        self.Conv_Layer2_1 = ConvStandard(numInputCh=2*c, numOutCh=4*c)
        self.Conv_Layer2_2 = ConvStandard(numInputCh=2*c, numOutCh=4*c)
        self.DownSample_Layer2_1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.DownSample_Layer2_2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # Layer3 (up-scale)
        self.Conv_Layer3_first_1 = ConvStandard(numInputCh=4*c, numOutCh=4*c)
        self.Conv_Layer3_first_2 = ConvStandard(numInputCh=4*c, numOutCh=4*c)
        self.Conv_Layer3_second_1 = ResBlock(4*c)
        self.Conv_Layer3_second_2 = ResBlock(4*c)
        self.Conv_Layer3_third_1 = ConvStandard(numInputCh=4*c, numOutCh=2*c)
        self.Conv_Layer3_third_2 = ConvStandard(numInputCh=4*c, numOutCh=2*c)
        self.UpSample_Layer3_1 = UpSampling(numInputCh=2*c)
        self.UpSample_Layer3_2 = UpSampling(numInputCh=2*c)
        # Layer4 (up-scale)
        self.Conv_Layer4_first_1 = ConvStandard(numInputCh=4*c, numOutCh=2*c)
        self.Conv_Layer4_first_2 = ConvStandard(numInputCh=4*c, numOutCh=2*c)
        self.Conv_Layer4_second_1 = ResBlock(2*c)
        self.Conv_Layer4_second_2 = ResBlock(2*c)
        self.Conv_Layer4_third_1 = ConvStandard(numInputCh=2*c, numOutCh=c)
        self.Conv_Layer4_third_2 = ConvStandard(numInputCh=2*c, numOutCh=c)
        self.UpSample_Layer4_1 = UpSampling(numInputCh=c)
        self.UpSample_Layer4_2 = UpSampling(numInputCh=c)

        #--- Fusion and Refinement ---#
        self.Fusion = Conv2x(numInputCh=4*c, numMidCh=3*c, numOutCh=2*c)
        self.Refine = ConvStandard(numInputCh=2*c, numOutCh=c, k=1)


    def forward(self, x1, x2, visualize=False):
        # motion estimation
        master_motion1          = self.GetMasterMotion1(torch.cat((x1, x2), 1))
        master_motion_layer1    = self.GetMasterMotion_Layer1(master_motion1)
        motion_layer1_1         = self.GetMotion_Layer1_1(master_motion_layer1 + master_motion1)
        motion_layer1_2         = self.GetMotion_Layer1_2(master_motion_layer1 + master_motion1)

        master_motion2          = self.GetMasterMotion2(self.Motion_DownSample(master_motion_layer1 + master_motion1))
        master_motion_layer2    = self.GetMasterMotion_Layer2(master_motion2)
        motion_layer2_1         = self.GetMotion_Layer2_1(master_motion_layer2 + master_motion2)
        motion_layer2_2         = self.GetMotion_Layer2_2(master_motion_layer2 + master_motion2)

        # motion compensation
        deform_layer1_1         = self.DeformConv_Layer1_1(x1, offset=motion_layer1_1)
        deform_layer1_2         = self.DeformConv_Layer1_2(x2, offset=motion_layer1_2)
        conv_layer1_1           = self.Conv_Layer1_1(deform_layer1_1)
        conv_layer1_2           = self.Conv_Layer1_2(deform_layer1_2)
        conv_layer1_1           = self.DownSample_Layer1_1(conv_layer1_1)
        conv_layer1_2           = self.DownSample_Layer1_2(conv_layer1_2)

        deform_layer2_1         = self.DeformConv_Layer2_1(conv_layer1_1, offset=motion_layer2_1)
        deform_layer2_2         = self.DeformConv_Layer2_2(conv_layer1_2, offset=motion_layer2_2)
        conv_layer2_1           = self.Conv_Layer2_1(deform_layer2_1)
        conv_layer2_2           = self.Conv_Layer2_2(deform_layer2_2)
        conv_layer2_1           = self.DownSample_Layer2_1(conv_layer2_1)
        conv_layer2_2           = self.DownSample_Layer2_2(conv_layer2_2)

        conv_layer3_first_1     = self.Conv_Layer3_first_1(conv_layer2_1)
        conv_layer3_first_2     = self.Conv_Layer3_first_2(conv_layer2_2)
        conv_layer3_second_1    = self.Conv_Layer3_second_1(conv_layer3_first_1)
        conv_layer3_second_2    = self.Conv_Layer3_second_2(conv_layer3_first_2)
        conv_layer3_third_1     = self.Conv_Layer3_third_1(conv_layer3_first_1 + conv_layer3_second_1)
        conv_layer3_third_2     = self.Conv_Layer3_third_2(conv_layer3_first_2 + conv_layer3_second_2)
        upsample_layer3_1       = self.UpSample_Layer3_1(F.interpolate(conv_layer3_third_1, scale_factor=2.0, mode='bilinear', align_corners=True))
        upsample_layer3_2       = self.UpSample_Layer3_2(F.interpolate(conv_layer3_third_2, scale_factor=2.0, mode='bilinear', align_corners=True))

        conv_layer4_first_1     = self.Conv_Layer4_first_1(torch.cat((upsample_layer3_1, deform_layer2_1), 1))
        conv_layer4_first_2     = self.Conv_Layer4_first_2(torch.cat((upsample_layer3_2, deform_layer2_2), 1))
        conv_layer4_second_1    = self.Conv_Layer4_second_1(conv_layer4_first_1)
        conv_layer4_second_2    = self.Conv_Layer4_second_2(conv_layer4_first_2)
        conv_layer4_third_1     = self.Conv_Layer4_third_1(conv_layer4_first_1 + conv_layer4_second_1)
        conv_layer4_third_2     = self.Conv_Layer4_third_2(conv_layer4_first_2 + conv_layer4_second_2)
        upsample_layer4_1       = self.UpSample_Layer4_1(F.interpolate(conv_layer4_third_1, scale_factor=2.0, mode='bilinear', align_corners=True))
        upsample_layer4_2       = self.UpSample_Layer4_2(F.interpolate(conv_layer4_third_2, scale_factor=2.0, mode='bilinear', align_corners=True))

        compensated_1           = torch.cat((upsample_layer4_1, deform_layer1_1), 1)
        compensated_2           = torch.cat((upsample_layer4_2, deform_layer1_2), 1)

        # fusion and refinement
        fused                   = self.Fusion(torch.cat((compensated_1, compensated_2), 1))
        refined                 = self.Refine(fused)

        if visualize:
            motion_field_visualization(motion_layer1_1, g=self.g, g_h=4, save_dir=visualize, l='1_1')
            motion_field_visualization(motion_layer1_2, g=self.g, g_h=4, save_dir=visualize, l='1_2')
            motion_field_visualization(motion_layer2_1, g=2*self.g, g_h=4, save_dir=visualize, l='2_1')
            motion_field_visualization(motion_layer2_2, g=2*self.g, g_h=4, save_dir=visualize, l='2_2')

        return refined



################## Model-3 ######################
def Conv2x(numInputCh, numMidCh, numOutCh, k=3):
    return nn.Sequential(
        nn.Conv2d(in_channels=numInputCh,  out_channels=numMidCh, kernel_size=k, stride=1, padding=k//2),
        nn.Conv2d(in_channels=numMidCh,  out_channels=numOutCh , kernel_size=k, stride=1, padding=k//2),
    )
def Conv1x(numInputCh, numOutCh, k=3):
    return nn.Conv2d(in_channels=numInputCh,  out_channels=numOutCh, kernel_size=k, stride=1, padding=k//2)

def ConvStandard(numInputCh, numOutCh, k=3, act=nn.LeakyReLU()):
    return nn.Sequential(
        nn.Conv2d(in_channels=numInputCh,  out_channels=numOutCh, kernel_size=k, stride=1, padding=k//2),
        nn.BatchNorm2d(numOutCh),
        act,
    )

def UpSampling(numInputCh, numOutCh, k=3, act=nn.LeakyReLU()):
    return nn.Sequential(
        nn.Conv2d(in_channels=numInputCh, out_channels=numOutCh, kernel_size=k, stride=1, padding=k//2),
        act,
    )

class ResBlock(nn.Module):
    def __init__(self, channels, k=3, act=nn.LeakyReLU()):
        super().__init__()
        self.Conv1 = nn.Conv2d(in_channels=channels,  out_channels=channels, kernel_size=k, stride=1, padding=k//2)
        self.Conv2 = nn.Conv2d(in_channels=channels,  out_channels=channels, kernel_size=k, stride=1, padding=k//2)
        self.act = act

    def forward(self, x):
        y = x
        return (x + self.Conv2(self.act(self.Conv1(y))))

class PreProcessing(nn.Module):
    def __init__(self, c, k=3):
        super().__init__()
        self.blk1 = ResBlock(c, k=k)
        self.blk2 = ResBlock(c, k=k)
        self.blk3 = ConvStandard(c, c, k=k)

    def forward(self, x):
        return self.blk3(self.blk2(self.blk1(x)))

class MotionEstimation(nn.Module):
    def __init__(self, c):
        super().__init__()

        self.layer1_blk1 = ResBlock(c)
        self.layer1_blk2 = ResBlock(c)
        self.layer1_blk3 = ConvStandard(c, 2*c)
        self.layer1_blk4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.layer2_blk1 = ResBlock(2*c)
        self.layer2_blk2 = ResBlock(2*c)
        self.layer2_blk3 = ConvStandard(2*c, 4*c)
        self.layer2_blk4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.layer3_blk1 = ResBlock(4*c)
        self.layer3_blk2 = ConvStandard(4*c, 4*c)
        self.layer3_blk3 = Conv2x(4*c, 4*c, 4*c)

        self.layer4_blk1 = UpSampling(4*c, 3*c)
        self.layer4_blk2 = ResBlock(5*c)
        self.layer4_blk3 = ConvStandard(5*c, 3*c)
        self.layer4_blk4 = Conv2x(3*c, 3*c, 3*c)

        self.layer5_blk1 = UpSampling(3*c, 2*c)
        self.layer5_blk2 = ResBlock(3*c)
        self.layer5_blk3 = ConvStandard(3*c, 2*c)
        self.layer5_blk4 = Conv2x(2*c, 2*c, 2*c)

    def forward(self, x):
        x1_1 = self.layer1_blk1(x)
        branch1 = self.layer1_blk2(x1_1)
        x1_2 = self.layer1_blk3(branch1)
        x1_2 = self.layer1_blk4(x1_2)

        x2_1 = self.layer2_blk1(x1_2)
        branch2 = self.layer2_blk2(x2_1)
        x2_2 = self.layer2_blk3(branch2)
        x2_2 = self.layer2_blk4(x2_2)

        x3_1 = self.layer3_blk1(x2_2)
        x3_1 = self.layer3_blk2(x3_1)
        x3_2 = self.layer3_blk3(x3_1)
        output1 = x3_1 + x3_2

        x4_1 = self.layer4_blk1(F.interpolate(output1, scale_factor=2.0, mode='bilinear', align_corners=True))
        x4_2 = self.layer4_blk2(torch.cat((x4_1, branch2),1))
        x4_2 = self.layer4_blk3(x4_2)
        x4_3 = self.layer4_blk4(x4_2)
        output2 = x4_2 + x4_3

        x5_1 = self.layer5_blk1(F.interpolate(output2, scale_factor=2.0, mode='bilinear', align_corners=True))
        x5_2 = self.layer5_blk2(torch.cat((x5_1, branch1),1))
        x5_2 = self.layer5_blk3(x5_2)
        x5_3 = self.layer5_blk4(x5_2)
        output3 = x5_2 + x5_3

        return output1, output2, output3

class MotionCompensation(nn.Module):
    def __init__(self, c, G=1):
        super().__init__()

        self.layer1_blk1 = ResBlock(c)
        self.layer1_blk2 = ResBlock(c)
        self.layer1_blk3 = ConvStandard(c, 2*c)
        self.layer1_blk4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.layer2_blk1 = ResBlock(2*c)
        self.layer2_blk2 = ResBlock(2*c)
        self.layer2_blk3 = ConvStandard(2*c, 4*c)
        self.layer2_blk4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.layer3_blk1 = ResBlock(4*c)
        self.layer3_blk2 = torchvision.ops.DeformConv2d(in_channels=4*c, out_channels=4*c, kernel_size=3, padding=1, groups=G, bias=False)
        self.layer3_blk3 = ConvStandard(4*c, 2*c)
        self.layer3_blk4 = UpSampling(2*c, c)

        self.layer4_blk1 = ResBlock(3*c)
        self.layer4_blk2 = torchvision.ops.DeformConv2d(in_channels=3*c, out_channels=3*c, kernel_size=3, padding=1, groups=G, bias=False)
        self.layer4_blk3 = ConvStandard(3*c, 2*c)
        self.layer4_blk4 = UpSampling(2*c, c)

        self.layer5_blk1 = ResBlock(2*c)
        self.layer5_blk2 = torchvision.ops.DeformConv2d(in_channels=2*c, out_channels=2*c, kernel_size=3, padding=1, groups=G, bias=False)
        self.layer5_blk3 = ConvStandard(2*c, c)

    def forward(self, x, off1, off2, off3):
        x1 = self.layer1_blk1(x)
        x1 = self.layer1_blk2(x1)
        x1 = self.layer1_blk3(x1)
        x1 = self.layer1_blk4(x1)

        x2 = self.layer2_blk1(x1)
        x2 = self.layer2_blk2(x2)
        x2 = self.layer2_blk3(x2)
        x2 = self.layer2_blk4(x2)

        x3 = self.layer3_blk1(x2)
        x3 = self.layer3_blk2(x3, offset=off1)
        x3 = self.layer3_blk3(x3)
        x3 = self.layer3_blk4(F.interpolate(x3, scale_factor=2.0, mode='bilinear', align_corners=True))

        x4 = self.layer4_blk1(torch.cat((x1, x3),1))
        x4 = self.layer4_blk2(x4, offset=off2)
        x4 = self.layer4_blk3(x4)
        x4 = self.layer4_blk4(F.interpolate(x4, scale_factor=2.0, mode='bilinear', align_corners=True))

        x5 = self.layer5_blk1(torch.cat((x, x4),1))
        x5 = self.layer5_blk2(x5, offset=off3)
        x5 = self.layer5_blk3(x5)

        return x5

class Refinement(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.blk1 = ResBlock(c)
        self.blk2 = Conv1x(c, c, k=3)
        self.blk3 = ConvStandard(c, c//2, k=1)

    def forward(self, x):
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        return x

class InterPrediction_3(nn.Module):
    def __init__(self, c, G) -> None:
        super().__init__()
        self.g = G
        self.pre_processing = PreProcessing(c)
        self.me = MotionEstimation(2*c)
        self.conv1_1 = Conv1x(8*c, 4*G*2*9)
        self.conv1_2 = Conv1x(8*c, 4*G*2*9)
        self.conv2_1 = Conv1x(6*c, 3*G*2*9)
        self.conv2_2 = Conv1x(6*c, 3*G*2*9)
        self.conv3_1 = Conv1x(4*c, 2*G*2*9)
        self.conv3_2 = Conv1x(4*c, 2*G*2*9)
        self.mc1 = MotionCompensation(c, G=G)
        self.mc2 = MotionCompensation(c, G=G)
        self.refinement = Refinement(2*c)

    def forward(self, x1, x2, visualize=False):
        x1_refined = self.pre_processing(x1)
        x2_refined = self.pre_processing(x2)
        x_me = torch.cat((x1_refined, x2_refined), 1)
        master_off1, master_off2, master_off3 = self.me(x_me)
        off1_1 = self.conv1_1(master_off1)
        off1_2 = self.conv1_2(master_off1)
        off2_1 = self.conv2_1(master_off2)
        off2_2 = self.conv2_2(master_off2)
        off3_1 = self.conv3_1(master_off3)
        off3_2 = self.conv3_2(master_off3)
        x_cmp1 = self.mc1(x1, off1=off1_1, off2=off2_1, off3=off3_1)
        x_cmp2 = self.mc2(x2, off1=off1_2, off2=off2_2, off3=off3_2)
        x_out = self.refinement(torch.cat((x_cmp1, x_cmp2), 1))
        if visualize:
            motion_field_visualization(off1_1, g=(4*self.g), g_h=4, save_dir=visualize, l='1_1')
            motion_field_visualization(off1_2, g=(4*self.g), g_h=4, save_dir=visualize, l='1_2')
            motion_field_visualization(off2_1, g=(3*self.g), g_h=4, save_dir=visualize, l='2_1')
            motion_field_visualization(off2_2, g=(3*self.g), g_h=4, save_dir=visualize, l='2_2')
            motion_field_visualization(off3_1, g=(2*self.g), g_h=4, save_dir=visualize, l='3_1')
            motion_field_visualization(off3_2, g=(2*self.g), g_h=4, save_dir=visualize, l='3_2')
        return x_out

