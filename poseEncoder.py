import torch
from torch import nn

'''
We take in pose as a 17x64x64 heatmap. 17 channels for 17 pose keypoints.
We output 64x64, 32x32, 16x16, 8x8 and 4x4 tensors to input at multiple
style blocks, starting from the first.
'''

class PoseEncoder(nn.Module):
    def __init__(self, imageSize = 256):
        super().__init__()

        channels = [int(min(imageSize/2, 512))]
        for i in range(4):
            channels.append(min(channels[-1]*2, 512))

        self.conv64x64 = nn.Conv2d(17, channels[0], kernel_size=3, padding=1)
        
        self.conv32x32 = nn.ModuleList([nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
                                        nn.MaxPool2d(2)])
        
        self.conv16x16 = nn.ModuleList([nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),
                                        nn.MaxPool2d(2)])
        
        self.conv8x8 = nn.ModuleList([nn.Conv2d(channels[2], channels[3], kernel_size=3, padding=1),
                                        nn.MaxPool2d(2)])
        
        self.conv4x4 = nn.ModuleList([nn.Conv2d(channels[3], channels[4], kernel_size=3, padding=1),
                                        nn.MaxPool2d(2)])
        
        self.conv4x4_2 = nn.Conv2d(channels[4], channels[4], kernel_size=3, padding=1)
        self.act = nn.LeakyReLU()
        
    def forward(self, x, only4 = False):
        x64 = self.act(self.conv64x64(x))
        
        x32 = x64
        for layer in self.conv32x32:
            x32 = self.act(layer(x32))
        
        x16 = x32
        for layer in self.conv16x16:
            x16 = self.act(layer(x16))
        
        x8 = x16
        for layer in self.conv8x8:
            x8 = self.act(layer(x8))
        
        x4 = x8
        for layer in self.conv4x4:
            x4 = self.act(layer(x4))
        x4 = x4 + self.act(self.conv4x4_2(x4))
        x4 = torch.div(x4, torch.sqrt(torch.tensor(2)))

        if only4:
            return x4
        else:
            return x64, x32, x16, x8, x4
