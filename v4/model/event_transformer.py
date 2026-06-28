import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbed(nn.Module):
    """
    Event voxel (B, 5, H, W) → patch tokens (B, N, embed_dim)
    patch_size=16: 640/16=40, so N=40*40=1600 patches
    """
    def __init__(self, in_ch=5, img_size=640,
                 patch_size=16, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches  = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                     # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)     # (B, N, D)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, n_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(
            torch.zeros(1, n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        return x + self.pos_embed


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=256, n_heads=8,
                 mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim, n_heads,
            dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim    = int(embed_dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class EventMAETransformer(nn.Module):
    """
    Masked Autoencoder for Event Voxel data.

    Two modes:
      pretrain=True  → mask patches, reconstruct them (L_mae)
      pretrain=False → encode all patches, return features for fusion

    Args:
        in_ch      : voxel channels (5)
        img_size   : input resolution (640)
        patch_size : patch size (16)
        embed_dim  : transformer hidden dim (256)
        depth      : number of transformer blocks (6)
        n_heads    : attention heads (8)
        mask_ratio : fraction of patches to mask during pretraining (0.75)
    """
    def __init__(self, in_ch=5, img_size=640,
                 patch_size=16, embed_dim=256,
                 depth=6, n_heads=8, mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.embed_dim  = embed_dim

        self.patch_embed = PatchEmbed(in_ch, img_size,
                                      patch_size, embed_dim)
        n_patches = self.patch_embed.n_patches
        self.pos_enc = PositionalEncoding(n_patches, embed_dim)

        # Encoder
        self.encoder = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads)
            for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # Decoder (lightweight, only for pretraining)
        self.mask_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder_pos = nn.Parameter(
            torch.zeros(1, n_patches, embed_dim))
        self.decoder = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads // 2)
            for _ in range(2)
        ])
        self.decoder_norm = nn.LayerNorm(embed_dim)
        # reconstruct original patch pixels
        self.decoder_pred = nn.Linear(
            embed_dim, in_ch * patch_size * patch_size)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos, std=0.02)

    def random_mask(self, x):
        """
        Randomly mask (mask_ratio) of patches.
        Returns: x_masked, mask (True=masked), ids_restore
        """
        B, N, D = x.shape
        n_keep = int(N * (1 - self.mask_ratio))

        noise   = torch.rand(B, N, device=x.device)
        ids_shuffle  = torch.argsort(noise, dim=1)
        ids_restore  = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :n_keep]
        x_masked = torch.gather(
            x, 1,
            ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, N, device=x.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)  # 1=masked
        return x_masked, mask, ids_restore

    def encode(self, x):
        """Encode all patches (inference / fusion mode)"""
        x = self.patch_embed(x)
        x = self.pos_enc(x)
        for blk in self.encoder:
            x = blk(x)
        x = self.encoder_norm(x)
        return x   # (B, N, D)

    def forward(self, x, pretrain=False):
        """
        pretrain=False → return encoded features (B, N, embed_dim)
        pretrain=True  → return (pred_patches, mask) for L_mae
        """
        tokens = self.patch_embed(x)
        tokens = self.pos_enc(tokens)

        if not pretrain:
            for blk in self.encoder:
                tokens = blk(tokens)
            return self.encoder_norm(tokens)

        # ── Pretraining path ──
        x_vis, mask, ids_restore = self.random_mask(tokens)

        for blk in self.encoder:
            x_vis = blk(x_vis)
        x_vis = self.encoder_norm(x_vis)

        # Decoder: fill mask tokens back
        B, N, D = tokens.shape
        n_keep  = x_vis.shape[1]
        mask_tokens = self.mask_token.expand(
            B, N - n_keep, -1)
        x_full = torch.cat([x_vis, mask_tokens], dim=1)
        x_full = torch.gather(
            x_full, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, D))
        x_full = x_full + self.decoder_pos

        for blk in self.decoder:
            x_full = blk(x_full)
        x_full = self.decoder_norm(x_full)
        pred   = self.decoder_pred(x_full)   # (B, N, p*p*C)

        return pred, mask
