import torch
from torch import nn
import numpy as np

'''
We take in pose as a 17x64x64 heatmap. 17 channels for 17 pose keypoints.
We output 64x64, 32x32, 16x16, 8x8 and 4x4 tensors to input at multiple
style blocks, starting from the first.
'''

class PoseEncoder(nn.Module):
    def __init__(self, imageSize = 256, channel_base = 32768):
        super().__init__()

        self.img_resolution_log2 = int(np.log2(imageSize))
        self.img_channels = 3
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels = {res: min(channel_base // res, 512) for res in self.block_resolutions}

        self.conv64x64 = nn.Conv2d(17, channels[64], kernel_size=3, padding=1)
        
        self.conv32x32 = nn.ModuleList([nn.Conv2d(channels[64], channels[32], kernel_size=3, padding=1),
                                        nn.MaxPool2d(2)])
        
        self.conv16x16 = nn.ModuleList([nn.Conv2d(channels[32], channels[16], kernel_size=3, padding=1),
                                        nn.MaxPool2d(2)])
        
        self.conv8x8 = nn.ModuleList([nn.Conv2d(channels[16], channels[8], kernel_size=3, padding=1),
                                        nn.MaxPool2d(2)])
        
        self.conv4x4 = nn.ModuleList([nn.Conv2d(channels[8], channels[4], kernel_size=3, padding=1),
                                        nn.MaxPool2d(2)])
        
        self.conv4x4_2 = nn.Conv2d(channels[4], channels[4], kernel_size=3, padding=1)
        self.act = nn.LeakyReLU()
        
    def forward(self, x, only4 = False):

        ret = {}
        x64 = self.act(self.conv64x64(x))
        ret[64] = x64
        
        x32 = x64
        for layer in self.conv32x32:
            x32 = self.act(layer(x32))
            ret[32] = x32
        
        x16 = x32
        for layer in self.conv16x16:
            x16 = self.act(layer(x16))
            ret[16] = x16
        
        x8 = x16
        for layer in self.conv8x8:
            x8 = self.act(layer(x8))
            ret[8] = x8
        
        x4 = x8
        for layer in self.conv4x4:
            x4 = self.act(layer(x4))
        x4 = x4 + self.act(self.conv4x4_2(x4))
        x4 = torch.div(x4, torch.sqrt(torch.tensor(2)))
        ret[4] = x4

        if only4:
            return x4
        else:
            return ret
