import torch
import torch.nn as nn
from model.layers import ConvBlock


class RSGNet(nn.Module):
    """UNet: Event Voxel (5,H,W) → Structure Feature Map (3,H,W)"""
    def __init__(self, in_ch=5, out_ch=3, base_ch=32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch,     base_ch)
        self.enc2 = ConvBlock(base_ch,   base_ch*2)
        self.enc3 = ConvBlock(base_ch*2, base_ch*4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_ch*4, base_ch*8)
        self.up3  = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch*8, base_ch*4)
        self.up2  = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch*4, base_ch*2)
        self.up1  = nn.ConvTranspose2d(base_ch*2, base_ch,   2, stride=2)
        self.dec1 = ConvBlock(base_ch*2, base_ch)
        self.out_conv = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out_conv(d1))
