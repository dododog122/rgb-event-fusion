# RGB–Event Camera Feature-Level Fusion for Nighttime Autonomous Driving

> Bachelor's thesis — ivlab, National Taiwan University
> Authors: Hsu Tzu-Fang, Lim Zhi-Xuan, Hsieh Chia-Yu

## Overview

This repository implements **Membrane Potential Feature-Level Fusion (V_mp)**, an RGB–Event camera fusion architecture for nighttime object detection in autonomous driving.

Unlike prior methods that convert event data into image overlays, V_mp injects event features **directly into YOLOv8's intermediate feature layers** (P3/P4), preserving temporal causality and polarity physics of DVS signals.

## Key Results

| Metric | Value |
|--------|-------|
| CARLA mAP@50 (V_mp) | 0.400 |
| DSEC mAP@50 improvement | +50% vs RGB-only |
| DSEC pedestrian AP | +32.5× (0.002→0.065) |
| SLAM keypoints (night scene) | 96× advantage (31→3,012) |
| 3D depth range (Event vs RGB) | 19.7m vs 11.4m (+8.3m) |

**Core finding:** Image-space fusion *hurts* performance in very dark scenes (−0.226 F1). Feature-level membrane fusion addresses this root cause.

## Method: V_mp Architecture

### Membrane Potential Accumulation
\`\`\`python
V_pos(t) = V_pos(t-1) * 0.9 + max(event(t),  0)
V_neg(t) = V_neg(t-1) * 0.9 + max(-event(t), 0)
\`\`\`
Inspired by the **Leaky Integrate-and-Fire (LIF)** neuron model. DVS hardware itself is designed to mimic retinal ganglion cells, so processing its output with the same biological logic is physically motivated.

### Fusion Flow
\`\`\`
RGB Image ──► YOLOv8 Backbone ──► P3 (80×80) ──► MAF ──► FPN/PAN ──► Detection
                                   P4 (40×40) ──► MAF ──►
                                        ▲
DVS Events ──► Membrane CNN Encoder ────┘
               (V_pos, V_neg → feature maps)
\`\`\`

**MembraneAdaptiveFusion (MAF):**
- Illumination gate: \`σ(Linear(AvgPool(RGB_feat)))\` — auto-weights event vs RGB by brightness
- Channel attention: selects informative fused channels
- Residual connection: guarantees floor ≥ RGB-only (safety guarantee)

## Model Versions

| Version | Description | CARLA mAP@50 |
|---------|-------------|---------------|
| Baseline | RGB-only YOLOv8n | 0.462 |
| V1 | Fixed-weight image overlay | 0.357 |
| V2 | Adaptive-weight overlay | 0.320 |
| V3 | Raw polarity overlay (CLAHE+voxel) | 0.399 |
| V4 | MAE ViT attention-guided overlay | 0.374 |
| V5 | Polarity-aware dual embedding (image-space) | 0.395 |
| **V_mp** | **Membrane potential feature injection** | **0.400** |

## Repository Structure

\`\`\`
rgb_event_fusion/
├── v5/
│   ├── membrane_potential.py     # Core model: V_pos/V_neg accumulation + MAF
│   ├── train_membrane_ultra.py   # Main training script (best result)
│   ├── dsec_finetune_v2.py       # DSEC domain adaptation
│   ├── dsec_eval.py              # Evaluation pipeline
│   └── slam_full.py              # SLAM keypoint analysis
├── v4/
│   ├── model/                    # V4 Transformer-based fusion
│   └── training/                 # V4 training scripts
├── model/                        # V1–V3 base models
├── scripts/                      # Data preparation tools
├── utils/                        # Dataset, loss, metrics
├── requirements.txt
└── README.md
\`\`\`

## Usage

### Train V_mp (main model)
\`\`\`bash
python v5/train_membrane_ultra.py \
    --data /path/to/carla/morning_new \
    --epochs 50 \
    --batch 16
\`\`\`

### Fine-tune on DSEC
\`\`\`bash
python v5/dsec_finetune_v2.py \
    --weights runs/fusion_membrane_ultra/weights/best.pt \
    --dsec-root /path/to/dsec_detection \
    --seq zurich_city_04_a
\`\`\`

### Evaluate on DSEC
\`\`\`bash
python v5/dsec_eval.py \
    --weights runs/fusion_membrane_ultra/weights/best.pt \
    --dsec-root /path/to/dsec_detection
\`\`\`

## Key Technical Notes

- **Custom detection loss is broken**: \`pred_cls[:,:NC]\` channel mismatch invalidates V4/V5 feature-loss variants. Only ultralytics official loss gives valid mAP.
- **DSEC GT label space**: Labels are in event camera space (640×480). Alignment uses \`W_evt/W_rgb\` scaling ratio, NOT rectify_map remapping.
- **DSEC "night" ≠ true darkness**: Luminance 76–103, closer to dusk. True extreme darkness tested via CARLA synthetic degradation.
- **Sim-to-real gap**: CARLA DVS (frame-differencing) ≠ real DVS hardware — fusion effects are significantly stronger on real DSEC data.

## Citation

\`\`\`bibtex
@misc{hsu2025rgbevent,
  author = {Hsu, Tzu-Fang and Lim, Zhi-Xuan and Hsieh, Chia-Yu},
  title  = {RGB--Event Camera Feature-Level Fusion for Nighttime Autonomous Driving Perception},
  year   = {2025},
  school = {National Taiwan University},
}
\`\`\`

## License
MIT License
