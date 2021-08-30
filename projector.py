# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import dnnlib
import legacy

import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms

class VGG16_for_Perceptual(torch.nn.Module):
    def __init__(self,requires_grad=False,n_layers=[2,4,14,21]):
        super(VGG16_for_Perceptual,self).__init__()
        vgg_pretrained_features=models.vgg16(pretrained=True).features

        self.slice0=torch.nn.Sequential()
        self.slice1=torch.nn.Sequential()
        self.slice2=torch.nn.Sequential()
        self.slice3=torch.nn.Sequential()

        for x in range(n_layers[0]):#relu1_1
            self.slice0.add_module(str(x),vgg_pretrained_features[x])
        for x in range(n_layers[0],n_layers[1]): #relu1_2
            self.slice1.add_module(str(x),vgg_pretrained_features[x])
        for x in range(n_layers[1],n_layers[2]): #relu3_2
            self.slice2.add_module(str(x),vgg_pretrained_features[x])

        for x in range(n_layers[2],n_layers[3]):#relu4_2
            self.slice3.add_module(str(x),vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad=False

    def forward(self,x):
        h0=self.slice0(x)
        h1=self.slice1(h0)
        h2=self.slice2(h1)
        h3=self.slice3(h2)
        return h0,h1,h2,h3


def caluclate_loss(synth_img,img,perceptual_net,img_p,MSE_Loss,upsample2d):
     #calculate MSE Loss
     mse_loss=MSE_Loss(synth_img,img) # (lamda_mse/N)*||G(w)-I||^2

     #calculate Perceptual Loss
     real_0,real_1,real_2,real_3=perceptual_net(img_p)
     synth_p=upsample2d(synth_img) #(1,3,256,256)
     synth_0,synth_1,synth_2,synth_3=perceptual_net(synth_p)

     perceptual_loss=0
     perceptual_loss+=MSE_Loss(synth_0,real_0)
     perceptual_loss+=MSE_Loss(synth_1,real_1)
     perceptual_loss+=MSE_Loss(synth_2,real_2)
     perceptual_loss+=MSE_Loss(synth_3,real_3)

     return mse_loss,perceptual_loss


def project(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    verbose                    = False,
    device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    
    w_out = []

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore
    g_synthesis = G.synthesis

    MSE_Loss = nn.MSELoss(reduction="mean")
    upsample2d = torch.nn.Upsample(scale_factor=256/G.img_resolution, mode='bilinear', align_corners=True)
    perceptual_net = VGG16_for_Perceptual(n_layers=[2,4,14,21]).to(device)
    
    img = target.float().div(255)
    img = img.to(device).unsqueeze(0)
    img_p = img.clone() #Perceptual loss
    img_p = upsample2d(img_p)

    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples.cpu().numpy().astype(np.float32)       # [N, L, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, L, C]
    
    dlatent = torch.tensor(w_avg, dtype = torch.float32, requires_grad = True, device=device)
    optimizer = torch.optim.Adam({dlatent}, lr = initial_learning_rate, betas=(0.9, 0.999), eps = 1e-8)

    print("Start")
    loss_list=[]
    for i in range(num_steps):
        optimizer.zero_grad()

        t = i / num_steps
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        synth_img = g_synthesis(dlatent, noise_mode='const')
        synth_img = (synth_img + 1) * (255/2)
        synth_img = synth_img.float().div(255)
        mse_loss,perceptual_loss = caluclate_loss(synth_img,img,perceptual_net,img_p,MSE_Loss,upsample2d)
        loss = 255 * mse_loss + perceptual_loss

        loss.backward()
        optimizer.step()

        loss_np=loss.detach().cpu().numpy()
        loss_p=perceptual_loss.detach().cpu().numpy()
        loss_m=mse_loss.detach().cpu().numpy()

        loss_list.append(loss_np)
        if i%10==0:
            print("iter{}: loss -- {},  mse_loss --{},  percep_loss --{}".format(i,loss_np,loss_m,loss_p))
        w_out.append(dlatent.clone().detach())

    return w_out

def getProjection(
    network_pkl,
    target_fname,
    num_steps
):

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

    # Load target image.
    target_pil = PIL.Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    # Optimize projection.
    projected_w_steps = project(
        G,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
        num_steps=num_steps,
        device=device,
        verbose=True
    )
    return projected_w_steps[-1]

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target', 'target_fname', help='Target image file to project to', required=True, metavar='FILE')
@click.option('--num-steps',              help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
def run_projection(
    network_pkl: str,
    target_fname: str,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int
):
    """Project given image to the latent space of pretrained network pickle.

    Examples:

    \b
    python projector.py --outdir=out --target=~/mytargetimg.png \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

    # Load target image.
    target_pil = PIL.Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps = project(
        G,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
        num_steps=num_steps,
        device=device,
        verbose=True
    )
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)
    if save_video:
        video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{outdir}/proj.mp4"')
        for projected_w in projected_w_steps:
            synth_image = G.synthesis(projected_w, noise_mode='const')
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
        video.close()

    # Save final projected frame and W vector.
    target_pil.save(f'{outdir}/target.png')
    projected_w = projected_w_steps[-1]
    synth_image = G.synthesis(projected_w, noise_mode='const')
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')
    np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
