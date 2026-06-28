"""
Membrane Potential Event Representation
V(t) = V(t-1) * decay + event(t)

不是圖片，是時序訊號
保留 event 的時間因果關係
正負極性分開累積
"""
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

H, W   = 720, 1280
DECAY  = 0.9   # temporal decay factor


def events_to_membrane(events, H=H, W=W, decay=DECAY):
    """
    Convert raw events to membrane potential maps.
    
    events: structured array with (x, y, t, pol)
    returns: (2, H, W) float32
             [0] = V_pos (brightness increase accumulation)
             [1] = V_neg (brightness decrease accumulation)
    """
    V_pos = np.zeros((H, W), dtype=np.float32)
    V_neg = np.zeros((H, W), dtype=np.float32)

    if len(events) == 0:
        return np.stack([V_pos, V_neg])

    # Sort by time
    idx  = np.argsort(events['t'])
    evts = events[idx]

    xs   = np.clip(evts['x'].astype(np.int32), 0, W-1)
    ys   = np.clip(evts['y'].astype(np.int32), 0, H-1)
    ts   = evts['t'].astype(np.float64)
    pols = evts['pol'].astype(np.bool_)

    # Normalize timestamps to [0, 1]
    t_min, t_max = ts.min(), ts.max()
    t_range = t_max - t_min
    if t_range > 0:
        ts_norm = (ts - t_min) / t_range
    else:
        ts_norm = np.zeros_like(ts)

    # Process events with temporal decay
    # Batch process for efficiency
    t_prev = 0.0
    for i in range(len(evts)):
        dt = ts_norm[i] - t_prev
        # Apply decay based on time elapsed
        d  = decay ** (dt * 100)  # scale dt for meaningful decay
        V_pos *= d
        V_neg *= d
        t_prev = ts_norm[i]

        x, y = xs[i], ys[i]
        if pols[i]:
            V_pos[y, x] += 1.0
        else:
            V_neg[y, x] += 1.0

    # Normalize to [0, 1]
    p_max = V_pos.max()
    n_max = V_neg.max()
    if p_max > 0: V_pos /= p_max
    if n_max > 0: V_neg /= n_max

    return np.stack([V_pos, V_neg])  # (2, H, W)


def events_to_membrane_fast(events, H=H, W=W, decay=DECAY, n_bins=10):
    """
    Faster version: process events in temporal bins.
    Each bin applies decay, then accumulates events.
    
    Much faster than per-event processing.
    """
    V_pos = np.zeros((H, W), dtype=np.float32)
    V_neg = np.zeros((H, W), dtype=np.float32)

    if len(events) == 0:
        return np.stack([V_pos, V_neg])

    # Sort by time
    idx  = np.argsort(events['t'])
    evts = events[idx]

    xs   = np.clip(evts['x'].astype(np.int32), 0, W-1)
    ys   = np.clip(evts['y'].astype(np.int32), 0, H-1)
    pols = evts['pol'].astype(np.bool_)

    # Split into temporal bins
    n    = len(evts)
    bins = np.array_split(np.arange(n), n_bins)

    for b_idx, bin_idx in enumerate(bins):
        if len(bin_idx) == 0:
            continue

        # Apply decay between bins
        V_pos *= decay
        V_neg *= decay

        # Accumulate events in this bin
        bx = xs[bin_idx]
        by = ys[bin_idx]
        bp = pols[bin_idx]

        np.add.at(V_pos, (by[bp],  bx[bp]),  1.0)
        np.add.at(V_neg, (by[~bp], bx[~bp]), 1.0)

    # Normalize
    p_max = V_pos.max()
    n_max = V_neg.max()
    if p_max > 0: V_pos /= p_max
    if n_max > 0: V_neg /= n_max

    return np.stack([V_pos, V_neg])  # (2, H, W)


# ── CNN Encoder for Membrane Potential ──
class MembranePotentialEncoder(nn.Module):
    """
    Lightweight CNN to encode (2, H, W) membrane maps
    into feature maps compatible with YOLOv8 P3/P4.
    
    No ViT needed — CNN naturally handles spatial structure.
    """
    def __init__(self):
        super().__init__()
        # Shared stem
        self.stem = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1),  # /2
            nn.BatchNorm2d(16), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # /4
            nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # /8
            nn.BatchNorm2d(64), nn.SiLU(),
        )
        # P3 branch (stride 8, ch=64 matching YOLOv8 P3)
        self.p3_branch = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.SiLU(),
        )
        # P4 branch (stride 16, ch=128 matching YOLOv8 P4)
        self.p4_branch = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # /16
            nn.BatchNorm2d(128), nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.SiLU(),
        )

    def forward(self, membrane):
        """
        membrane: (B, 2, H, W)  [V_pos, V_neg]
        returns: feat_p3 (B, 64, H/8, W/8)
                 feat_p4 (B, 128, H/16, W/16)
        """
        x   = self.stem(membrane)
        p3  = self.p3_branch(x)
        p4  = self.p4_branch(x)
        return p3, p4


# ── Illumination-Aware Cross-Attention ──
class MembraneFusion(nn.Module):
    """
    Fuse membrane potential features with RGB features.
    dark scene → more membrane contribution
    """
    def __init__(self, rgb_ch, mem_ch, n_heads=4):
        super().__init__()
        self.proj_mem = nn.Conv2d(mem_ch, rgb_ch, 1)  # match channels
        self.ca = nn.MultiheadAttention(rgb_ch, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(rgb_ch)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(rgb_ch, 1), nn.Sigmoid()
        )

    def forward(self, rgb_feat, mem_feat):
        """
        rgb_feat: (B, C, H, W)
        mem_feat: (B, C_mem, H, W)
        """
        B, C, H, W = rgb_feat.shape

        # Project membrane to same channel count
        mem = self.proj_mem(mem_feat)          # (B, C, H, W)
        mem = F.interpolate(mem, (H, W),
                            mode='bilinear', align_corners=False)

        # Illumination gate
        gate = 1.0 - self.gate(rgb_feat)      # dark=1, bright=0
        gate = gate.view(B, 1, 1)

        # Cross-attention: RGB attends to membrane
        q = rgb_feat.flatten(2).transpose(1,2)  # (B, HW, C)
        k = mem.flatten(2).transpose(1,2)
        v = k

        attn_out, _ = self.ca(q, k, v)
        fused = self.norm(q + attn_out * gate)
        return fused.transpose(1,2).reshape(B, C, H, W)


# ── Test ──
if __name__ == '__main__':
    import time

    # Load sample events
    evts = np.load('/home/ivlab/carla_ws/examples/'
                   'morning_3_cameras/dvs/frame_000000.npy')
    print(f'Events: {len(evts)}')

    # Test fast version
    t0 = time.time()
    mp = events_to_membrane_fast(evts)
    t1 = time.time()
    print(f'Membrane shape: {mp.shape}')
    print(f'V_pos range: [{mp[0].min():.3f}, {mp[0].max():.3f}]')
    print(f'V_neg range: [{mp[1].min():.3f}, {mp[1].max():.3f}]')
    print(f'Time: {t1-t0:.2f}s')

    # Visualize
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    vis[:,:,1] = (mp[0] * 255).astype(np.uint8)  # green = pos
    vis[:,:,2] = (mp[1] * 255).astype(np.uint8)  # red   = neg
    cv2.imwrite('/tmp/membrane_vis.png', vis)
    print('Saved /tmp/membrane_vis.png')

    # Test encoder
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc = MembranePotentialEncoder().to(device)
    x   = torch.from_numpy(mp).unsqueeze(0).to(device)
    p3, p4 = enc(x)
    print(f'P3 feature: {p3.shape}')
    print(f'P4 feature: {p4.shape}')
    print('All OK!')
