import torch
import torch.nn as nn
from ultralytics import YOLO
from v4.model.event_transformer import EventMAETransformer
from v4.model.fusion import CrossModalCBAM


class NightFusionV4(nn.Module):
    """
    Version 4: Feature-level RGB–Event fusion.

    RGB branch  : YOLOv8 CSPDarknet backbone
    Event branch: Event MAE Transformer
    Fusion      : Cross-modal CBAM at P3/P4 scale
    Head        : YOLOv8 detection head
    """
    def __init__(self, num_classes=4,
                 event_embed_dim=256,
                 event_depth=6):
        super().__init__()

        # ── Event MAE Transformer ──
        self.event_encoder = EventMAETransformer(
            in_ch=5, img_size=640,
            patch_size=16,
            embed_dim=event_embed_dim,
            depth=event_depth,
            n_heads=8,
            mask_ratio=0.75,
        )

        # ── YOLOv8 backbone (we extract intermediate features) ──
        yolo = YOLO('yolov8n.pt')
        self.yolo_model = yolo.model
        # freeze detection head, only train backbone+neck in joint training
        # (unfreeze later if needed)

        # ── Cross-modal fusion at P3 (ch=128) and P4 (ch=256) ──
        self.fusion_p3 = CrossModalCBAM(
            rgb_ch=128, event_dim=event_embed_dim, out_ch=128)
        self.fusion_p4 = CrossModalCBAM(
            rgb_ch=256, event_dim=event_embed_dim, out_ch=256)

        self.num_classes = num_classes

    def get_rgb_features(self, rgb):
        """Extract P3 and P4 features from YOLOv8 backbone."""
        x = rgb
        features = []
        for i, layer in enumerate(self.yolo_model.model):
            x = layer(x)
            if i == 4:   # P3: stride 8,  ch=128
                features.append(x)
            if i == 6:   # P4: stride 16, ch=256
                features.append(x)
            if i >= 6:
                break
        return features[0], features[1]   # p3, p4

    def forward(self, rgb, voxel, pretrain_event=False):
        """
        pretrain_event=True  → return MAE outputs for L_mae
        pretrain_event=False → return YOLO predictions
        """
        # ── Event Transformer ──
        if pretrain_event:
            pred_patches, mask = self.event_encoder(
                voxel, pretrain=True)
            return pred_patches, mask

        event_tokens = self.event_encoder(voxel, pretrain=False)
        # event_tokens: (B, N, embed_dim)

        # ── RGB backbone ──
        p3, p4 = self.get_rgb_features(rgb)

        # ── Fusion ──
        f3 = self.fusion_p3(p3, event_tokens, rgb)
        f4 = self.fusion_p4(p4, event_tokens, rgb)

        return f3, f4, event_tokens, p3, p4
