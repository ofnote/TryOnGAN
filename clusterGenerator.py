import torch.nn as nn

import numpy as np
from typing import List

import re
import dnnlib
import numpy as np
import torch

from sklearn.cluster import KMeans
import pickle
from projector import getProjection

import legacy

class HookedGenerator(nn.Module):
    def __init__(self, network_pkl):
        super(HookedGenerator, self).__init__()
        print('Loading networks from "%s"...' % network_pkl)
        self.device = torch.device('cuda')
        with dnnlib.util.open_url(network_pkl) as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(self.device) # type: ignore

        self.activation = {}

        featureLayer = None
        for i, layer in enumerate(self.G.synthesis.children()):
            featureLayer = layer
            h = featureLayer.conv1.register_forward_hook(self.getActivation('comp'+str(i)))

    def getActivation(self, name):
        def hook(model, input, output):
            self.activation[name] = output
        return hook

    def forward(self, w, fLayerIdx):
        img = self.G.synthesis(w)
        outputs=[]
        for i,_ in enumerate(self.G.synthesis.children()):
            if i == fLayerIdx:
                output = self.activation['comp' + str(i)]
                outputs.append(output)
        return img, outputs


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
    n_colors,
    featureLayer,
    kmeans_path = None
):

    '''
    kmeans, centres, labels, imgs = generate_clusters( network_pkl="/gdrive/MyDrive/ada-tryongan/best_checkpoints/UC-scratch-network-snapshot-002404.pkl",
                seeds="120-200",
                truncation_psi=0.75,
                n_colors=7,
                featureLayer=0)
    '''
    HG = HookedGenerator(network_pkl).eval().requires_grad_(False)

    imgs = []
    feature_all = []
    seeds = num_range(seeds)
    for seed_idx, seed in enumerate(seeds):
        with torch.no_grad():
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, HG.G.z_dim)).to(HG.device)
            label = torch.zeros([1, HG.G.c_dim], device = HG.device)
            w = HG.G.mapping(z, label, truncation_psi=truncation_psi)

            img, features = HG(w, featureLayer)

            features128 = []
            for f in features:
                f128 = nn.functional.interpolate(f,
                                                size=(128, 128),
                                                mode='bilinear',
                                                align_corners=True).clamp(min=-1.0, max=1.0).detach()

                features128.append(f128)

            features_cat=torch.cat(features128, axis = 1)

        img128 = nn.functional.interpolate(img,
                                        size=(128, 128),
                                        mode='bilinear',
                                        align_corners=True).clamp(min=-1.0, max=1.0).detach()

        imgs.append(img128)
        feature_all.append(features_cat)

    features_all = torch.cat(feature_all, axis=0)
    features_tr = features_all.permute(0, 2, 3, 1)
    features_flat = features_tr.reshape(-1, features_tr.shape[-1])
    arr = features_flat.detach().cpu().numpy()

    if kmeans_path is None:
        #train a kmeans model and get clusters and labels
        kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(arr)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        labels_spatial = labels.reshape(features_all.shape[0], features_all.shape[2], features_all.shape[3])

        imgs_all = torch.cat(imgs, axis=0)
        img_arr = imgs_all.permute(0, 2, 3, 1).detach().cpu().numpy()

        return kmeans, centers, labels_spatial, img_arr

    else:
        #use a saved kmeans pickle
        with open(kmeans_path, "rb") as f:
            kmeans = pickle.load(f)

        predictions_flat = kmeans.predict(arr)
        labels_spatial = predictions_flat.reshape(features_all.shape[0], features_all.shape[2], features_all.shape[3])

        imgs_all = torch.cat(imgs, axis=0)
        img_arr = imgs_all.permute(0, 2, 3, 1).detach().cpu().numpy()

        return labels_spatial, img_arr



def clusterRealImage(
    network_pkl,
    kmeans_path,
    featureLayer,
    imgPath
):

    '''
    invert image to latent,
    get features for latent
    generate clusterss using features and kmeans

    img_arr, predictions = clusterRealImage(network_pkl="network-snapshot.pkl",
                                        kmeans_path="k_means_cluster_7_layer_4.pkl",
                                        featureLayer=4,
                                        imgPath="img.jpeg")
    '''

    w = getProjection(network_pkl=network_pkl, target_fname=imgPath, num_steps=1000)
    HG = HookedGenerator(network_pkl, featureLayer).eval().requires_grad_(False)

    with torch.no_grad():
        img, feature = HG(w)
        print(feature[0].shape)

        features128 = []
        for f in feature:
            f128 = nn.functional.interpolate(f,
                                            size=(128, 128),
                                            mode='bilinear',
                                            align_corners=True).clamp(min=-1.0, max=1.0).detach()

            features128.append(f128)

        feature=torch.cat(features128, axis = 1)

    img128 = nn.functional.interpolate(img,
                                    size=(128, 128),
                                    mode='bilinear',
                                    align_corners=True).clamp(min=-1.0, max=1.0).detach()

    img_arr = img128.permute(0, 2, 3, 1).detach().cpu().numpy()

    features_tr = feature.permute(0, 2, 3, 1)
    features_flat = features_tr.reshape(-1, features_tr.shape[-1])

    arr = features_flat.detach().cpu().numpy()

    with open(kmeans_path, "rb") as f:
        kmeans = pickle.load(f)

    predictions_flat = kmeans.predict(arr)
    predictions = predictions_flat.reshape(feature.shape[0], feature.shape[2], feature.shape[3])
    
    return img_arr, predictions


def getMask(featureList,
            clusterLabelList,
            kmeans
):
    featureReshaped = []
    for f in featureList:
        f_new = nn.functional.interpolate(f,
                                        size=(128, 128),
                                        mode='bilinear',
                                        align_corners=True).clamp(min=-1.0, max=1.0).detach()

        featureReshaped.append(f_new)

    features_all=torch.cat(featureReshaped, axis = 1).permute(0, 2, 3, 1)
    features_flat = features_all.reshape(-1, features_all.shape[-1])

    arr = features_flat.detach().cpu().numpy()
    predictions_flat = kmeans.predict(arr)
    predictions = predictions_flat.reshape(features_all.shape[0], features_all.shape[1], features_all.shape[2])

    #convert to binary mask
    mask = None
    for clusterLabel in clusterLabelList:
        if mask is None: mask = (predictions[0]==clusterLabel)
        else : mask = mask + (predictions[0]==clusterLabel)

    #upscale predictions
    mask = torch.from_numpy(mask)
    mask = mask.unsqueeze(0).unsqueeze(0).float()
    mask = torch.broadcast_to(mask, (mask.shape[0], 3, mask.shape[2], mask.shape[3]))
    return mask
