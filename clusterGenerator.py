import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

import math, base64, io, os, time, cv2
import numpy as np
from typing import List, Optional

import re
import click
import imageio
import dnnlib
import numpy as np
import pandas as pd
import PIL.Image
import torch
import copy

from skimage import io
from sklearn.cluster import KMeans

import legacy

class HookedGenerator(nn.Module):
    def __init__(self, network_pkl):
        super(HookedGenerator, self).__init__()
        print('Loading networks from "%s"...' % network_pkl)
        self.device = torch.device('cuda')
        with dnnlib.util.open_url(network_pkl) as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(self.device) # type: ignore

        self.activation = {}
        h = self.G.synthesis.b64.conv1.register_forward_hook(self.getActivation('comp'))

    def getActivation(self, name):
        def hook(model, input, output):
            self.activation[name] = output
        return hook

    def forward(self, seed, trunc):
        label = torch.zeros([1, self.G.c_dim], device = self.device)
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, self.G.z_dim)).to(self.device)
        img = self.G(z, label, truncation_psi=trunc)
        output = self.activation['comp']
        return img, output


def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


def generate_clusters(
    network_pkl,
    seeds,
    truncation_psi,
    n_colors
):
    print("hello world")

    HG = HookedGenerator(network_pkl).eval().requires_grad_(False)

    imgs = []
    features = []
    seeds = num_range(seeds)
    for seed_idx, seed in enumerate(seeds):
        with torch.no_grad():
            img, feature = HG(seed, truncation_psi)
        img128 = nn.functional.interpolate(img,
                                        size=(128, 128),
                                        mode='bilinear',
                                        align_corners=True).clamp(min=-1.0, max=1.0).detach()
        feature128 = nn.functional.interpolate(feature,
                                            size=(128, 128),
                                            mode='bilinear',
                                            align_corners=True).detach()

        imgs.append(img128)
        features.append(feature128)


    features_all = torch.cat(features, axis=0)
    features_flat = features_all.permute(0, 2, 3, 1).reshape(-1, 256)


    arr = features_flat.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(arr)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    labels_spatial = labels.reshape(features_all.shape[0], features_all.shape[2], features_all.shape[3])
    
    return centers, labels_spatial, imgs