import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, act='elu'):
        super().__init__()
        act_fn = nn.ELU(inplace=True) if act=='elu' \
                 else nn.GELU()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            act_fn,
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            act_fn,
        )
    def forward(self, x):
        return self.block(x)


class BackboneBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, act='elu'):
        super().__init__()
        act_fn = nn.ELU(inplace=True) if act=='elu' \
                 else nn.GELU()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            act_fn,
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            act_fn,
        )
    def forward(self, x):
        return self.block(x)


class SEBlock(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, max(ch//reduction, 4)),
            nn.ELU(inplace=True),
            nn.Linear(max(ch//reduction, 4), ch),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(x).view(x.shape[0], x.shape[1], 1, 1)


class IlluminationGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3, 16),
            nn.ELU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
    def forward(self, rgb):
        return self.fc(rgb)
