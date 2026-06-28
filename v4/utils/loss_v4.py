import torch
import torch.nn as nn
import torch.nn.functional as F


class MAELoss(nn.Module):
    """
    Loss 1: Masked Autoencoder reconstruction loss.
    Only compute loss on masked patches (where mask==1).

    pred  : (B, N, patch_size^2 * in_ch)  reconstructed patches
    target: (B, N, patch_size^2 * in_ch)  original patches
    mask  : (B, N)  1=masked, 0=visible
    """
    def forward(self, pred, target, mask):
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss.mean(dim=-1)           # (B, N)
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss


class FeatureAlignLoss(nn.Module):
    """
    Loss 2: Feature alignment between RGB and Event features.
    Forces the two modalities to share a common semantic space.

    rgb_feat   : (B, N_rgb, D)  or (B, C, H, W)
    event_feat : (B, N_evt, D)
    """
    def __init__(self, proj_dim=256):
        super().__init__()
        # small projection heads to match dimensions if needed
        self.rgb_proj   = None
        self.event_proj = None
        self.proj_dim   = proj_dim

    def _build_proj(self, in_dim, device):
        return nn.Sequential(
            nn.Linear(in_dim, self.proj_dim),
            nn.GELU(),
            nn.Linear(self.proj_dim, self.proj_dim),
        ).to(device)

    def forward(self, rgb_feat, event_feat):
        # flatten spatial dims if needed
        if rgb_feat.dim() == 4:
            B, C, H, W = rgb_feat.shape
            rgb_feat = rgb_feat.flatten(2).transpose(1, 2)  # (B,N,C)

        B, Nr, Dr = rgb_feat.shape
        B, Ne, De = event_feat.shape

        # build projection heads lazily
        if self.rgb_proj is None or self.rgb_proj[0].in_features != Dr:
            self.rgb_proj   = self._build_proj(Dr, rgb_feat.device)
        if self.event_proj is None or self.event_proj[0].in_features != De:
            self.event_proj = self._build_proj(De, event_feat.device)

        # project both to same dim
        rgb_z   = self.rgb_proj(rgb_feat)      # (B, Nr, proj_dim)
        event_z = self.event_proj(event_feat)  # (B, Ne, proj_dim)

        # global average pool → (B, proj_dim)
        rgb_g   = rgb_z.mean(dim=1)
        event_g = event_z.mean(dim=1)

        # normalize
        rgb_g   = F.normalize(rgb_g,   dim=-1)
        event_g = F.normalize(event_g, dim=-1)

        # cosine similarity loss: 1 - sim (want sim → 1)
        sim  = (rgb_g * event_g).sum(dim=-1)   # (B,)
        loss = (1 - sim).mean()
        return loss


class FusionLossV4(nn.Module):
    """
    Total loss for Version 4:
      L_total = L_yolo + lambda_mae * L_mae + lambda_align * L_align

    L_yolo is computed by ultralytics trainer internally.
    This module provides L_mae and L_align as auxiliary losses.
    """
    def __init__(self, lambda_mae=0.3, lambda_align=0.2,
                 proj_dim=256):
        super().__init__()
        self.lambda_mae   = lambda_mae
        self.lambda_align = lambda_align
        self.mae_loss     = MAELoss()
        self.align_loss   = FeatureAlignLoss(proj_dim)

    def forward(self, mae_pred, mae_target, mae_mask,
                rgb_feat, event_feat):
        """
        mae_pred   : (B, N, patch_pixels)  decoder output
        mae_target : (B, N, patch_pixels)  original patches
        mae_mask   : (B, N)                1=masked
        rgb_feat   : (B, C, H, W) or (B,N,D)
        event_feat : (B, N, D)
        """
        l_mae   = self.mae_loss(mae_pred, mae_target, mae_mask)
        l_align = self.align_loss(rgb_feat, event_feat)
        total   = self.lambda_mae * l_mae + \
                  self.lambda_align * l_align
        return total, l_mae, l_align


def patchify(voxel, patch_size=16):
    """
    Convert voxel (B, C, H, W) to patch tokens (B, N, C*p*p)
    for use as MAE reconstruction target.
    """
    B, C, H, W = voxel.shape
    p = patch_size
    h, w = H // p, W // p
    x = voxel.reshape(B, C, h, p, w, p)
    x = x.permute(0, 2, 4, 1, 3, 5)    # (B, h, w, C, p, p)
    x = x.reshape(B, h*w, C*p*p)
    return x
