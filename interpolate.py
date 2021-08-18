"""Interpolates between two latent vectors using pretrained network pickle."""

import os
import re
from typing import List, Optional

import click
import imageio
import dnnlib
import numpy as np
import pandas as pd
import PIL.Image
import torch
import copy

import legacy
from scipy.stats import multivariate_normal

#----------------------------------------------------------------------------
def get_pose(filename, df, image_size):
    base = os.path.basename(filename)
    keypoint = df[df['name'] == base]['keypoints'].tolist()
    if len(keypoint) > 0:
        keypoint = keypoint[0]
        ptlist = keypoint.split(':')
        ptlist = [float(x) for x in ptlist]
        maps = getHeatMap(ptlist, image_size)
    else:
        maps = torch.zeros(17, 64, 64)
    return maps.float()

def getHeatMap(pose, image_size):
    '''
    pose should be a list of length 51, every 3 number for
    x, y and confidence for each of the 17 keypoints.
    '''

    stack = []
    for i in range(17):
        x = pose[3*i]
        
        y = pose[3*i + 1]
        c = pose[3*i + 2]
        
        ratio = 64.0 / image_size
        map = getGaussianHeatMap([x*ratio, y*ratio])

        if c < 0.4:
            map = 0.0 * map
        stack.append(map)
    
    maps = np.dstack(stack)
    heatmap = torch.from_numpy(maps).transpose(0, -1)
    return heatmap

def getGaussianHeatMap(bonePos):
    width = 64
    x, y = np.mgrid[0:width:1, 0:width:1]
    pos = np.dstack((x, y))

    gau = multivariate_normal(mean = list(bonePos), cov = [[width*0.02, 0.0], [0.0, width*0.02]]).pdf(pos)
    return gau

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.75, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w1', help='Projection result file', type=str, metavar='FILE')
@click.option('--projected-w2', help='Projection result file', type=str, metavar='FILE')
@click.option('--posefile', help='csv file with pose keypoints', type=str, metavar='FILE')
@click.option('--poselabel', help='poselabel', type=str)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--imagesize', help='size', type=int)
@click.option('--mix', help='mix', type=bool, default=False)
def interpolate_latents(
    ctx: click.Context,
    network_pkl: str,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    projected_w1: Optional[str],
    projected_w2: Optional[str],
    posefile,
    poselabel,
    imagesize,
    mix
):
    """Generate images using pretrained network pickle.
    Examples:
    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl
    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Synthesize the result of a W projection.
    if projected_w1 is not None and projected_w2 is not None:
        print(f'Generating interpolation images between projected W "{projected_w1}" and "{projected_w2}"')
        
        ws1 = np.load(projected_w1)['w']
        ws2 = np.load(projected_w2)['w']
        assert ws1.shape == (1, G.num_ws, G.w_dim)
        assert ws2.shape == (1, G.num_ws, G.w_dim)
        
        ws1 = ws1[0]
        ws2 = ws2[0]
        assert ws1.shape == (G.num_ws, G.w_dim)
        assert ws1.shape == (G.num_ws, G.w_dim)
        
        #pose
        pose = None
        if posefile is not None:
            if poselabel is not None:
                df = pd.read_csv(posefile)
                pose = get_pose(poselabel, df, imagesize)
                pose = torch.tensor(pose, device=device)
            else:
                print("Provide label for the pose in csv file provided")

        num_intermediate = 100
        wlist = np.linspace(ws1, ws2, num_intermediate)
        ws = torch.tensor(wlist, device=device) # pylint: disable=not-callable
        video = imageio.get_writer(f'{outdir}/interp.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        imgs = []
        for idx, w in enumerate(ws):
            if pose is None:
                img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            else:
                img = G.synthesis(w.unsqueeze(0), pose.unsqueeze(0), ret_pose=False, noise_mode=noise_mode)

            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = img[0].cpu().numpy()
            if idx % 25 == 0:
                imgs.append(img[:, 40:216, :])

            video.append_data(img)
        imgs.append(img[:, 40:216, :])
        imgsx = np.concatenate(imgs, axis=1)
        imgsaved = PIL.Image.fromarray(imgsx, 'RGB').save(f'{outdir}/interp.png')
        video.close()
        
        imgsxy = [imgsx]
        if mix:
            wmix1 = copy.deepcopy(ws2)
            wmix1[0:6] = ws1[0:6]
            wmix1 = torch.tensor(wmix1, device=device)
            if pose is None:
                img = G.synthesis(wmix1.unsqueeze(0), noise_mode=noise_mode)
            else:
                img = G.synthesis(wmix1.unsqueeze(0), pose.unsqueeze(0), ret_pose=False, noise_mode=noise_mode)

            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = img[0].cpu().numpy()
            img = img[:, 40:216, :]
            imgsx = np.concatenate([imgs[0], img, imgs[-1]], axis=1)
            imgsaved = PIL.Image.fromarray(imgsx, 'RGB').save(f'{outdir}/src->tgt.png')
        
            wmix2 = copy.deepcopy(ws1)
            wmix2[0:6] = ws2[0:6]
            wmix2 = torch.tensor(wmix2, device=device)
            if pose is None:
                img = G.synthesis(wmix2.unsqueeze(0), noise_mode=noise_mode)
            else:
                img = G.synthesis(wmix2.unsqueeze(0), pose.unsqueeze(0), ret_pose=False, noise_mode=noise_mode)

            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = img[0].cpu().numpy()
            img = img[:, 40:216, :]
            imgsx = np.concatenate([imgs[0], img, imgs[-1]], axis=1)
            imgsaved = PIL.Image.fromarray(imgsx, 'RGB').save(f'{outdir}/tgt->src.png')

            row_seeds=[100,200,250,356]
            all_seeds = list(set(row_seeds))
            all_z = np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])
            all_w = G.mapping(torch.from_numpy(all_z).to(device), None)
            w_avg = G.mapping.w_avg
            all_w = w_avg + (all_w - w_avg) * truncation_psi
            w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))}

            print('Generating style-mixed images...')
            wmix3l = []
            wmix4l = []
            for row_seed in row_seeds:
                wmix3 = w_dict[row_seed].clone()
                wmix3[0:6] = torch.tensor(ws1[0:6], device=device)
                if pose is None:
                    image = G.synthesis(wmix3[np.newaxis], noise_mode=noise_mode)
                else:
                    image = G.synthesis(wmix3[np.newaxis], pose.unsqueeze(0), ret_pose=False, noise_mode=noise_mode)

                image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                img = image[0].cpu().numpy()
                img = img[:, 40:216, :]
                wmix3l.append(img)

                wmix4 = w_dict[row_seed].clone()
                wmix4[0:6] = torch.tensor(ws2[0:6], device=device)
                if pose is None:
                    image = G.synthesis(wmix4[np.newaxis], noise_mode=noise_mode)
                else:
                    image = G.synthesis(wmix4[np.newaxis], pose.unsqueeze(0), ret_pose=False, noise_mode=noise_mode)

                image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                img = image[0].cpu().numpy()
                img = img[:, 40:216, :]
                wmix4l.append(img)

            imgsx = np.concatenate(wmix3l, axis=1)
            imgsaved = PIL.Image.fromarray(imgsx, 'RGB').save(f'{outdir}/srcmix.png')

            imgsx = np.concatenate(wmix4l, axis=1)
            imgsaved = PIL.Image.fromarray(imgsx, 'RGB').save(f'{outdir}/tgtmix.png')
            
        return


#----------------------------------------------------------------------------

if __name__ == "__main__":
    interpolate_latents() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------