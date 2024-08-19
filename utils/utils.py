import numpy as np

import torch
import torch.nn.functional as F


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