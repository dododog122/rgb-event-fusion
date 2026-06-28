import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        mid = max(ch // reduction, 4)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(ch, mid, bias=False),
            nn.GELU(),
            nn.Linear(mid, ch, bias=False),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        a = self.fc(self.avg(x).flatten(1))
        m = self.fc(self.max(x).flatten(1))
        w = self.sig(a + m).view(x.shape[0], x.shape[1], 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(1, keepdim=True)
        mx  = x.max(1, keepdim=True).values
        sa  = self.sig(self.conv(torch.cat([avg, mx], 1)))
        return x * sa


class IlluminationGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, rgb):
        return self.fc(rgb)   # (B, 1) — high=bright, low=dark


class CrossModalCBAM(nn.Module):
    """
    Cross-modal CBAM Fusion:
      1. Illumination gate: weight RGB vs Event by scene brightness
      2. Channel attention on concatenated features
      3. Spatial attention
      4. Project back to ch
    """
    def __init__(self, rgb_ch, event_dim, out_ch):
        super().__init__()
        self.ill_gate  = IlluminationGate()
        # project event tokens to spatial feature map
        self.evt_proj  = nn.Linear(event_dim, out_ch)
        self.ca        = ChannelAttention(rgb_ch + out_ch)
        self.sa        = SpatialAttention()
        self.proj      = nn.Sequential(
            nn.Conv2d(rgb_ch + out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, rgb_feat, event_tokens, rgb_img):
        """
        rgb_feat     : (B, rgb_ch, H, W)
        event_tokens : (B, N, event_dim)   N = (H/p)*(W/p)
        rgb_img      : (B, 3, H_orig, W_orig)  for illumination gate
        """
        B, C, H, W = rgb_feat.shape
        B, N, D    = event_tokens.shape

        # project event tokens → spatial map
        evt = self.evt_proj(event_tokens)          # (B, N, out_ch)
        h   = w = int(N ** 0.5)
        evt = evt.transpose(1, 2).reshape(B, -1, h, w)  # (B, out_ch, h, w)
        evt = F.interpolate(evt, size=(H, W), mode='bilinear',
                            align_corners=False)

        # illumination-aware weighting
        alpha = self.ill_gate(rgb_img).view(B, 1, 1, 1)
        fused = torch.cat([rgb_feat * alpha,
                           evt * (1 - alpha)], dim=1)

        fused = self.ca(fused)
        fused = self.sa(fused)
        return self.proj(fused)
