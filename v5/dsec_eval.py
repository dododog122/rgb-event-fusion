"""
Phase 1: DSEC Sim-to-Real Transfer Evaluation
用 CARLA 訓練的模型在真實 DSEC 夜間資料上測試
"""
import sys
sys.path.insert(0, '/home/ivlab/carla_ws/fusion_model')

import h5py, hdf5plugin
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from torchvision.ops import box_iou

DSEC_BASE  = '/media/ivlab/8AC24E8FC24E7F85/dsec_detection'
MODEL_BASE = '/home/ivlab/carla_ws/fusion_model/runs/detect/runs'
NC         = 4
CONF       = 0.25
IOU_THR    = 0.5

# DSEC class mapping → our class
# DSEC: 0=pedestrian, 1=large_vehicle, 2=car
# Ours: 0=vehicle, 1=pedestrian, 2=sign, 3=light
DSEC_TO_OURS = {0: 1, 1: 0, 2: 0}  # ped→1, car/large→0

NIGHT_SEQS = ['zurich_city_04_a', 'zurich_city_04_b', 'zurich_city_04_c']


def load_events_for_frame(h5_file, t_offset, t_label_us, window_us=50000):
    """
    Load events in time window [t_label - window, t_label]
    t_label_us: absolute timestamp in microseconds
    """
    t_rel_start = max(0, (t_label_us - t_offset) - window_us)
    t_rel_end   = t_label_us - t_offset

    # Use ms_to_idx for fast lookup
    ms_start = int(t_rel_start / 1000)
    ms_end   = min(int(t_rel_end / 1000) + 1,
                   len(h5_file['ms_to_idx']) - 1)

    idx_start = int(h5_file['ms_to_idx'][ms_start])
    idx_end   = int(h5_file['ms_to_idx'][ms_end])

    if idx_end <= idx_start:
        return None

    x = h5_file['events/x'][idx_start:idx_end].astype(np.int32)
    y = h5_file['events/y'][idx_start:idx_end].astype(np.int32)
    t = h5_file['events/t'][idx_start:idx_end].astype(np.float32)
    p = h5_file['events/p'][idx_start:idx_end].astype(np.bool_)

    return x, y, t, p


def events_to_membrane(x, y, t, p, H, W, decay=0.9, n_bins=10):
    """Convert events to membrane potential."""
    V_pos = np.zeros((H, W), dtype=np.float32)
    V_neg = np.zeros((H, W), dtype=np.float32)
    if len(x) == 0:
        return np.stack([V_pos, V_neg])

    xs = np.clip(x, 0, W-1)
    ys = np.clip(y, 0, H-1)

    for bi in np.array_split(np.arange(len(x)), n_bins):
        if len(bi) == 0: continue
        V_pos *= decay; V_neg *= decay
        bx=xs[bi]; by=ys[bi]; bp=p[bi]
        np.add.at(V_pos,(by[bp], bx[bp]),  1.0)
        np.add.at(V_neg,(by[~bp],bx[~bp]), 1.0)

    pm=V_pos.max(); nm=V_neg.max()
    if pm>0: V_pos/=pm
    if nm>0: V_neg/=nm
    return np.stack([V_pos, V_neg])


def make_v3_fusion(rgb, membrane_2hw, img_size=640):
    """Create V3-style fusion image from RGB + membrane."""
    rgb_r = cv2.resize(rgb, (img_size, img_size))
    lab   = cv2.cvtColor(rgb_r, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    rgb_enh = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    mp_pos = cv2.resize(membrane_2hw[0], (img_size, img_size))
    mp_neg = cv2.resize(membrane_2hw[1], (img_size, img_size))

    evt = np.zeros((img_size, img_size, 3), dtype=np.float32)
    evt[:,:,1] = mp_pos * 255
    evt[:,:,2] = mp_neg * 255

    fused = cv2.addWeighted(
        rgb_enh.astype(np.float32), 0.65,
        evt, 0.35, 0).clip(0,255).astype(np.uint8)
    return fused, rgb_r


def compute_ap(rec, prec):
    rec  = np.concatenate([[0.], rec,  [1.]])
    prec = np.concatenate([[0.], prec, [0.]])
    for i in range(len(prec)-2,-1,-1):
        prec[i]=max(prec[i],prec[i+1])
    idx = np.where(rec[1:]!=rec[:-1])[0]
    return np.sum((rec[idx+1]-rec[idx])*prec[idx+1])


def evaluate_model(model, seq_data, model_type='rgb'):
    """
    Evaluate model on DSEC sequence.
    model_type: 'rgb' or 'v3'
    """
    all_preds = {0:[], 1:[]}   # vehicle, pedestrian
    all_ngt   = {0:0,  1:0}

    for item in tqdm(seq_data, desc=f'  {model_type}'):
        rgb_path   = item['rgb_path']
        membrane   = item['membrane']
        gt_boxes   = item['gt_boxes']   # (N,4) xyxy normalized
        gt_cls     = item['gt_cls']     # (N,)

        # Count GT
        for c in gt_cls:
            if c in all_ngt:
                all_ngt[c] += 1

        if gt_boxes.shape[0] == 0:
            continue

        # Prepare input
        if model_type == 'rgb':
            inp = str(rgb_path)
        else:  # v3 fusion
            rgb = cv2.imread(str(rgb_path))
            fused, _ = make_v3_fusion(rgb, membrane)
            tmp = '/tmp/dsec_fused.png'
            cv2.imwrite(tmp, fused)
            inp = tmp

        # Predict
        results = model.predict(inp, conf=CONF, verbose=False,
                                imgsz=640)[0]

        pred_boxes, pred_cls, pred_conf = [], [], []
        for box in results.boxes:
            c = int(box.cls[0])
            if c < NC:
                x1,y1,x2,y2 = box.xyxyn[0].tolist()
                pred_boxes.append([x1,y1,x2,y2])
                pred_cls.append(c)
                pred_conf.append(float(box.conf[0]))

        if not pred_boxes:
            continue

        pb = torch.tensor(pred_boxes)
        gb = torch.tensor(gt_boxes)
        iou = box_iou(pb, gb)

        gt_matched = set()
        order = np.argsort(pred_conf)[::-1]

        for idx in order:
            c    = pred_cls[idx]
            conf = pred_conf[idx]
            if c not in all_preds:
                continue

            gt_mask = np.array(gt_cls) == c
            if gt_mask.sum() == 0:
                all_preds[c].append((conf, 0))
                continue

            gt_ids = np.where(gt_mask)[0]
            ious_c = iou[idx][gt_ids]
            best_j = ious_c.argmax().item()
            best_iou = ious_c[best_j].item()
            best_gt  = gt_ids[best_j]

            if best_iou >= IOU_THR and best_gt not in gt_matched:
                gt_matched.add(best_gt)
                all_preds[c].append((conf, 1))
            else:
                all_preds[c].append((conf, 0))

    # Compute AP per class
    aps = []
    cls_names = {0:'vehicle', 1:'pedestrian'}
    for c, name in cls_names.items():
        ngt = all_ngt[c]
        if ngt == 0 or len(all_preds[c]) == 0:
            print(f'    {name}: AP=0.000 (GT={ngt})')
            aps.append(0.)
            continue
        preds_c = sorted(all_preds[c], key=lambda x:-x[0])
        tp = np.array([p[1] for p in preds_c])
        fp = 1 - tp
        tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
        rec  = tp_c / (ngt + 1e-6)
        prec = tp_c / (tp_c + fp_c + 1e-6)
        ap   = compute_ap(rec, prec)
        aps.append(ap)
        print(f'    {name}: AP@50={ap:.3f} (GT={ngt})')

    mAP = np.mean(aps)
    print(f'    mAP@50 = {mAP:.3f}')
    return mAP


def prepare_sequence(seq, max_frames=200):
    """Load and align RGB + events + labels for a sequence."""
    print(f'  Preparing {seq}...')
    rgb_dir   = Path(f'{DSEC_BASE}/{seq}')
    track     = np.load(f'{DSEC_BASE}/train/{seq}/object_detections/left/tracks.npy')
    h5_path   = f'{DSEC_BASE}/{seq}/events.h5'

    items = []
    with h5py.File(h5_path, 'r') as f:
        t_offset = float(f['t_offset'][()])

        # Get unique timestamps from labels
        unique_ts = np.unique(track['t'])[:max_frames]

        for t_label in tqdm(unique_ts, desc='    loading', leave=False):
            # Get labels at this timestamp
            mask    = track['t'] == t_label
            frame_t = track[mask]

            # Find closest RGB frame index
            # DSEC images are at ~20Hz = 50ms intervals
            t_rel  = t_label - t_offset
            img_idx = int(t_rel / 1e6 * 20)  # 20fps
            img_idx = min(img_idx, 698)

            rgb_path = rgb_dir / f'{img_idx:06d}.png'
            if not rgb_path.exists():
                continue

            # Load events for this frame
            evts = load_events_for_frame(f, t_offset, t_label)
            if evts is None:
                membrane = np.zeros((2, 480, 640), dtype=np.float32)
            else:
                x,y,t_e,p = evts
                rgb_tmp = cv2.imread(str(rgb_path))
                H,W = rgb_tmp.shape[:2]
                membrane = events_to_membrane(x,y,t_e,p,H,W)

            # GT boxes (normalized xyxy)
            gt_boxes, gt_cls = [], []
            W_img = 1440; H_img = 1080
            for det in frame_t:
                dsec_cls = int(det['class_id'])
                if dsec_cls not in DSEC_TO_OURS:
                    continue
                our_cls = DSEC_TO_OURS[dsec_cls]
                x1 = det['x'] / W_img
                y1 = det['y'] / H_img
                x2 = (det['x'] + det['w']) / W_img
                y2 = (det['y'] + det['h']) / H_img
                gt_boxes.append([x1,y1,x2,y2])
                gt_cls.append(our_cls)

            if not gt_boxes:
                continue

            items.append({
                'rgb_path': rgb_path,
                'membrane': membrane,
                'gt_boxes': np.array(gt_boxes),
                'gt_cls':   gt_cls,
            })

    print(f'    {len(items)} frames with GT')
    return items


def main():
    print('\n=== Phase 1: DSEC Sim-to-Real Transfer ===')
    print('Models trained on CARLA, tested on real DSEC night data\n')

    # Load models
    rgb_model = YOLO(f'{MODEL_BASE}/rgb_only_yolov8/weights/best.pt')
    v3_model  = YOLO(f'{MODEL_BASE}/fusion_yolov8_v3/weights/best.pt')

    results = {}

    for seq in NIGHT_SEQS:
        print(f'\n--- {seq} ---')
        seq_data = prepare_sequence(seq, max_frames=150)
        if not seq_data:
            continue

        print(f'  RGB-only:')
        rgb_mAP = evaluate_model(rgb_model, seq_data, 'rgb')

        print(f'  V3 Fusion:')
        v3_mAP  = evaluate_model(v3_model,  seq_data, 'v3')

        results[seq] = {'rgb': rgb_mAP, 'v3': v3_mAP}

    # Summary
    print('\n=== DSEC Night Transfer Results ===')
    print(f'{"Sequence":<25} {"RGB-only":>10} {"V3 Fusion":>10} {"Diff":>8}')
    print('-'*55)
    for seq, res in results.items():
        diff = res['v3'] - res['rgb']
        marker = '← V3 wins' if diff > 0 else ''
        print(f'{seq:<25} {res["rgb"]:>10.3f} {res["v3"]:>10.3f} {diff:>+8.3f} {marker}')

    avg_rgb = np.mean([r['rgb'] for r in results.values()])
    avg_v3  = np.mean([r['v3']  for r in results.values()])
    print('-'*55)
    print(f'{"AVERAGE":<25} {avg_rgb:>10.3f} {avg_v3:>10.3f} {avg_v3-avg_rgb:>+8.3f}')

    print('\n=== Comparison ===')
    print(f'CARLA (simulated):')
    print(f'  RGB-only mAP@50 = 0.462')
    print(f'  V3 Fusion mAP@50 = 0.399')
    print(f'DSEC (real world):')
    print(f'  RGB-only mAP@50 = {avg_rgb:.3f}')
    print(f'  V3 Fusion mAP@50 = {avg_v3:.3f}')
    print(f'Sim-to-Real gap (RGB): {0.462 - avg_rgb:.3f}')


if __name__ == '__main__':
    main()
