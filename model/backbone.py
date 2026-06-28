import torch.nn as nn
from model.layers import BackboneBlock


class DualBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_b1 = BackboneBlock(3,   32)
        self.rgb_b2 = BackboneBlock(32,  64)
        self.rgb_b3 = BackboneBlock(64,  128)
        self.rgb_b4 = BackboneBlock(128, 256)

        self.evt_b1 = BackboneBlock(3,   32)
        self.evt_b2 = BackboneBlock(32,  64)
        self.evt_b3 = BackboneBlock(64,  128)
        self.evt_b4 = BackboneBlock(128, 256)

    def forward(self, rgb, sif):
        r1 = self.rgb_b1(rgb)
        r2 = self.rgb_b2(r1)
        r3 = self.rgb_b3(r2)
        r4 = self.rgb_b4(r3)

        e1 = self.evt_b1(sif)
        e2 = self.evt_b2(e1)
        e3 = self.evt_b3(e2)
        e4 = self.evt_b4(e3)

        return (r3, r4), (e3, e4)
