import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from model.model   import NightFusionModel
from utils.dataset import get_dataloaders

DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE      = '/home/ivlab/carla_ws/fusion_model'
CKPT      = 'checkpoints/best.pt'
IMG_SIZE  = 640
NUM_CLS   = 4
NA        = 3
CLS_NAMES = ['vehicle','pedestrian','traffic_sign','traffic_light']

CFG = {
    'train_rgb':    f'{BASE}/morning/rgb',
    'train_voxel':  f'{BASE}/morning/voxel',
    'train_labels': f'{BASE}/morning/yolo/labels',
    'val_rgb':      f'{BASE}/night/rgb',
    'val_voxel':    f'{BASE}/night/voxel',
    'val_labels':   f'{BASE}/night/yolo/labels',
    'img_size':     IMG_SIZE,
    'batch_size':   8,
}


def decode(pred, conf_thr=0.01):
    """
    pred: (B, NA*(5+NC), H, W)
    returns list of (N,6) tensors: [cx,cy,w,h,conf,cls]
    """
    B, _, H, W = pred.shape
    p = pred.view(B, NA, 5+NUM_CLS, H, W).permute(0,1,3,4,2)
    # p: (B, NA, H, W, 5+NC)

    results = []
    for b in range(B):
        boxes = []
        obj   = p[b,:,:,:,4].sigmoid()          # (NA,H,W)
        mask  = obj >= conf_thr

        if mask.sum() == 0:
            results.append(torch.zeros((0,6)))
            continue

        for a in range(NA):
            for j in range(H):
                for i in range(W):
                    if not mask[a,j,i]:
                        continue
                    conf = obj[a,j,i].item()
                    raw  = p[b,a,j,i]

                    # decode cx,cy,w,h
                    cx = (i + torch.sigmoid(raw[0]).item()) / W
                    cy = (j + torch.sigmoid(raw[1]).item()) / H
                    w  = torch.exp(raw[2].clamp(-4,4)).item() * 0.1
                    h  = torch.exp(raw[3].clamp(-4,4)).item() * 0.1
                    w  = min(w, 1.0)
                    h  = min(h, 1.0)
                    cls = raw[5:].argmax().item()

                    boxes.append([cx, cy, w, h, conf, cls])

        if boxes:
            results.append(torch.tensor(boxes))
        else:
            results.append(torch.zeros((0,6)))
    return results


def box_iou_single(b1, b2):
    """b1, b2: [cx,cy,w,h]"""
    b1x1, b1y1 = b1[0]-b1[2]/2, b1[1]-b1[3]/2
    b1x2, b1y2 = b1[0]+b1[2]/2, b1[1]+b1[3]/2
    b2x1, b2y1 = b2[0]-b2[2]/2, b2[1]-b2[3]/2
    b2x2, b2y2 = b2[0]+b2[2]/2, b2[1]+b2[3]/2

    ix1 = max(b1x1, b2x1); iy1 = max(b1y1, b2y1)
    ix2 = min(b1x2, b2x2); iy2 = min(b1y2, b2y2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    area1 = (b1x2-b1x1)*(b1y2-b1y1)
    area2 = (b2x2-b2x1)*(b2y2-b2y1)
    return inter / (area1+area2-inter+1e-6)


def compute_map(all_preds, all_targets, iou_thr=0.5):
    APs = []
    for cls in range(NUM_CLS):
        tp_list, conf_list, n_gt = [], [], 0

        for preds, targets in zip(all_preds, all_targets):
            # GT for this class
            if len(targets) == 0:
                continue
            gt = targets[targets[:,0]==cls][:,1:]  # (M,4)
            n_gt += len(gt)

            # preds for this class
            pr = preds[preds[:,5]==cls] if len(preds)>0 \
                 else torch.zeros((0,6))
            if len(pr) == 0:
                continue

            # sort by conf descending
            pr = pr[pr[:,4].argsort(descending=True)]
            matched = [False]*len(gt)

            for pred_box in pr:
                conf = pred_box[4].item()
                conf_list.append(conf)

                if len(gt) == 0:
                    tp_list.append(0)
                    continue

                ious = [box_iou_single(
                    pred_box[:4].tolist(),
                    gt[g].tolist()) for g in range(len(gt))]
                best_idx = int(np.argmax(ious))
                best_iou = ious[best_idx]

                if best_iou >= iou_thr and not matched[best_idx]:
                    tp_list.append(1)
                    matched[best_idx] = True
                else:
                    tp_list.append(0)

        if n_gt == 0 or len(tp_list) == 0:
            APs.append(0.0)
            continue

        tp   = np.array(tp_list)
        conf = np.array(conf_list)
        idx  = np.argsort(-conf)
        tp   = tp[idx]

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(1-tp)
        prec   = cum_tp / (cum_tp+cum_fp+1e-6)
        rec    = cum_tp / (n_gt+1e-6)
        APs.append(float(np.trapz(prec, rec)))

    return float(np.mean(APs)), APs


def main():
    _, val_loader = get_dataloaders(CFG)

    model = NightFusionModel(num_classes=NUM_CLS).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()
    print(f'Loaded: {CKPT}\n')

    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            rgb    = batch['rgb'].to(DEVICE)
            voxel  = batch['voxel'].to(DEVICE)
            labels = batch['labels']

            p3, p4, _ = model(rgb, voxel)
            dec3 = decode(p3)
            dec4 = decode(p4)

            for i in range(len(labels)):
                parts = []
                if len(dec3[i]) > 0: parts.append(dec3[i])
                if len(dec4[i]) > 0: parts.append(dec4[i])
                p = torch.cat(parts) if parts \
                    else torch.zeros((0,6))
                all_preds.append(p)
                all_targets.append(labels[i])

    mAP, APs = compute_map(all_preds, all_targets)

    print('\n' + '='*48)
    print('  Fusion Model — Night Scene Evaluation')
    print('='*48)
    for name, ap in zip(CLS_NAMES, APs):
        print(f'  {name:20s}  AP@50 = {ap:.4f}')
    print('-'*48)
    print(f'  mAP@50                = {mAP:.4f}')
    print('='*48)
    print(f'\n  RGB-only baseline     = 0.533')
    print(f'  Fusion model (CBAM)   = {mAP:.4f}')
    diff = mAP - 0.533
    print(f'  Difference            = {diff:+.4f}')

if __name__ == '__main__':
    main()
