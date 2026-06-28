import torch.nn as nn
from model.rsgnet   import RSGNet
from model.backbone import DualBackbone
from model.fusion   import CBAMFusion, DetectionHead


class NightFusionModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.rsgnet    = RSGNet(in_ch=5, out_ch=3)
        self.backbone  = DualBackbone()
        self.fusion_p3 = CBAMFusion(ch=128)
        self.fusion_p4 = CBAMFusion(ch=256)
        self.head_p3   = DetectionHead(128, num_classes)
        self.head_p4   = DetectionHead(256, num_classes)

    def forward(self, rgb, voxel):
        sif              = self.rsgnet(voxel)
        (r3, r4),(e3, e4)= self.backbone(rgb, sif)
        f3               = self.fusion_p3(r3, e3, rgb)
        f4               = self.fusion_p4(r4, e4, rgb)
        return self.head_p3(f3), self.head_p4(f4), sif
