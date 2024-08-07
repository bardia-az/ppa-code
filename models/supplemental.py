# supplementary DNN modules

import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from models.common import *
from utils.plots import feature_visualization, visualize_one_channel



################ Input Reconstruction Decoder #######################
class Bottleneck_Inv(nn.Module):
    # Standard inverse bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=1.0):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c1 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 3, 1, g=g)
        self.cv2 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    

class C3_Inv(nn.Module):
    # Inverse CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c1 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * c_, 1)  # act=FReLU(c2)
        self.cv2 = Conv(c_, c2, 1, 1)
        self.cv3 = Conv(c_, c2, 1, 1)  
        self.m = nn.Sequential(*(Bottleneck_Inv(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
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


################## Auto Encoder ######################
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
    def __init__(self, chs):
        super().__init__()
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


class ResBlock(nn.Module):
    def __init__(self, channels, k=3, act=nn.LeakyReLU()):
        super().__init__()
        self.Conv1 = nn.Conv2d(in_channels=channels,  out_channels=channels, kernel_size=k, stride=1, padding=k//2)
        self.Conv2 = nn.Conv2d(in_channels=channels,  out_channels=channels, kernel_size=k, stride=1, padding=k//2)
        self.act = act

    def forward(self, x):
        y = x
        return (x + self.Conv2(self.act(self.Conv1(y))))

