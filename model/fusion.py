import torch
import torch.nn as nn
from .layers import IlluminationGate


class ChannelAttention(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        mid = max(ch // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, mid, bias=False),
            nn.ELU(inplace=True),
            nn.Linear(mid, ch, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x).flatten(1))
        mx  = self.fc(self.max_pool(x).flatten(1))
        w   = self.sigmoid(avg + mx)
        return x * w.view(x.shape[0], x.shape[1], 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size,
                                 padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        sa  = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * sa


class CBAM(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(ch, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class CBAMFusion(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ill_gate = IlluminationGate()
        self.cbam     = CBAM(ch * 2)
        self.proj     = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ELU(inplace=True),
        )

    def forward(self, rgb_feat, evt_feat, rgb_img):
        alpha    = self.ill_gate(rgb_img).view(-1, 1, 1, 1)
        weighted = torch.cat([
            rgb_feat * alpha,
            evt_feat * (1 - alpha),
        ], dim=1)
        attended = self.cbam(weighted)
        return self.proj(attended)


class DetectionHead(nn.Module):
    def __init__(self, in_ch, num_classes=4, num_anchors=3):
        super().__init__()
        out_ch = num_anchors * (5 + num_classes)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ELU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),
        )
    def forward(self, x):
        return self.head(x)