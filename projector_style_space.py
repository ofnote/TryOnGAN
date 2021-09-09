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
from torch_utils import misc

from projector import VGG16_for_Perceptual, caluclate_loss

import torch.nn as nn

class HookedStyleGenerator(nn.Module):
    def __init__(self, network_pkl, resolution):
        super(HookedStyleGenerator, self).__init__()

        self.device = torch.device('cuda')

        G_kwargs = dnnlib.EasyDict(class_name='training.networks.Generator', z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
        if resolution == 256: factor = 0.5
        else : factor = 1
        G_kwargs.synthesis_kwargs.channel_base = int(factor * 32768)
        G_kwargs.synthesis_kwargs.channel_max = 512
        G_kwargs.mapping_kwargs.num_layers = 2
        G_kwargs.synthesis_kwargs.num_fp16_res = 4
        G_kwargs.synthesis_kwargs.conv_clamp = 256

        common_kwargs = dict(c_dim=0, img_resolution=resolution, img_channels=3)
        self.G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(self.device)

        print('Loading networks from "%s"...' % network_pkl)

        with dnnlib.util.open_url(network_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        
        for name, module in [('G', self.G)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

        self.activation = {}


        featureLayer = None
        for i, layer in enumerate(self.G.synthesis.children()):
            featureLayer = layer

            if hasattr(featureLayer, 'conv0'):
                h = featureLayer.conv0.affine.register_forward_hook(self.getActivation('comp'+str(i)+'conv0'))

            if hasattr(featureLayer, 'conv1'):
                h = featureLayer.conv1.affine.register_forward_hook(self.getActivation('comp'+str(i)+'conv1'))

            if hasattr(featureLayer, 'torgb'):
                h = featureLayer.torgb.affine.register_forward_hook(self.getActivation('comp_'+str(i)+'torgb'))

    def getActivation(self, name):
        def hook(model, input, output):
            self.activation[name] = output
        return hook

    def forward(self, w):
        img = self.G.synthesis(w)
        outputs=[]
        for i, layer in enumerate(self.G.synthesis.children()):
            featureLayer = layer
            if hasattr(featureLayer, 'conv0'):
                output = self.activation['comp'+str(i)+'conv0']
                outputs.append(output)

            if hasattr(featureLayer, 'conv1'):
                output = self.activation['comp'+str(i)+'conv1']
                outputs.append(output)

            if hasattr(featureLayer, 'torgb'):
                output = self.activation['comp_'+str(i)+'torgb'] * featureLayer.torgb.weight_gain
                outputs.append(output)
        return img, outputs

class MyUpsample(nn.Module):
    def __init__(self, size, mode, align_corners):
        super(MyUpsample, self).__init__()
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, img):
        
        out = nn.functional.interpolate(img,
                                        size=self.size,
                                        mode=self.mode,
                                        align_corners=self.align_corners)
        return out

def projectStyle(
    HSG,
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

    G = HSG.G
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    MSE_Loss = nn.MSELoss(reduction="mean")
    upsample2d = MyUpsample(size = (256, 256), mode='bilinear', align_corners=True)
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
    style_out = []
    w_out = []
    for i in range(num_steps):
        optimizer.zero_grad()
        
        t = i / num_steps
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        synth_img, styles = HSG(dlatent)
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
        
        style_temp = [style.clone().detach() for style in styles]
        style_out.append(style_temp)
        w_out.append(dlatent.clone().detach())

    return w_out, style_out


def getStyleProjection(
    network_pkl,
    target_fname,
    num_steps,
    resolution
):

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    HSG = HookedStyleGenerator(network_pkl, resolution).eval().requires_grad_(False)
    
    # Load target image.
    target_pil = PIL.Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((HSG.G.img_resolution, HSG.G.img_resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    # Optimize projection.
    projected_w_steps, projected_style_steps = projectStyle(
        HSG,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
        num_steps=num_steps,
        device=device,
        verbose=True
    )
    return projected_w_steps[-1], projected_style_steps[-1]

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target', 'target_fname', help='Target image file to project to', required=True, metavar='FILE')
@click.option('--num-steps',              help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
@click.option('--resolution',             required=True, type=int)
def run_projection(
    network_pkl: str,
    target_fname: str,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int,
    resolution: int

):
    """Project given image to the latent space of pretrained network pickle.

    Examples:

    \b
    
    !(python3 projector_style_space.py \
    --network=/gdrive/MyDrive/ada-tryongan/best_checkpoints/UC-scratch-network-snapshot-002404.pkl \
    --target=/content/male17_front \
    --outdir=/content \
    --save-video=True \
    --num-steps=1000 \
    --resolution=256)
    
    """
    np.random.seed(seed)    
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    HSG = HookedStyleGenerator(network_pkl, resolution).eval().requires_grad_(False)
    
    # Load target image.
    target_pil = PIL.Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((HSG.G.img_resolution, HSG.G.img_resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    # Optimize projection.
    
    start_time = perf_counter()
    projected_w_steps, projected_style_steps = projectStyle(
        HSG,
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
        for projected_style in projected_style_steps:

            #the projected_w_steps[0] is dummy, is not used when projected_style is provided.
            synth_image = HSG.G.synthesis(projected_w_steps[0], projected_style, noise_mode='const')
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
        video.close()

    # Save final projected frame and W vector.
    target_pil.save(f'{outdir}/target.png')
    projected_w = projected_w_steps[-1]
    synth_image = HSG.G.synthesis(projected_w, noise_mode='const')
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')
    np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
