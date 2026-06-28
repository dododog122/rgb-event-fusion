"""
Fine-tune on DSEC: RGB-only baseline + V3 fusion
用真實 DSEC 資料訓練，看 fusion 在真實 DVS 上有沒有幫助
"""
import sys
sys.path.insert(0, '/home/ivlab/carla_ws/fusion_model')

import h5py, hdf5plugin
import numpy as np
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

DSEC_BASE = '/media/ivlab/8AC24E8FC24E7F85/dsec_detection'
OUT_BASE  = '/media/ivlab/8AC24E8FC24E7F85/dsec_yolo'
NIGHT_SEQS_TRAIN = ['zurich_city_04_a', 'zurich_city_04_b']
NIGHT_SEQS_VAL   = ['zurich_city_04_c']

# DSEC class → our class
# 0=pedestrian→1, 1=large_vehicle→0, 2=car→0
DSEC_TO_OURS = {0:1, 1:0, 2:0}
W_EVT, H_EVT = 640, 480


def load_rectify_map(seq):
    with h5py.File(f'{DSEC_BASE}/{seq}/rectify_map.h5','r') as f:
        rect_map = f['rectify_map'][:]
    return rect_map[:,:,0].astype(np.float32), \
           rect_map[:,:,1].astype(np.float32)


def events_to_membrane(h5_file, t_offset, t_label, W=W_EVT, H=H_EVT,
                        window_us=50000, decay=0.9, n_bins=10):
    t_rel_end   = t_label - t_offset
    t_rel_start = max(0, t_rel_end - window_us)

    ms_start = int(t_rel_start / 1000)
    ms_end   = min(int(t_rel_end / 1000)+1,
                   len(h5_file['ms_to_idx'])-1)

    idx_s = int(h5_file['ms_to_idx'][ms_start])
    idx_e = int(h5_file['ms_to_idx'][ms_end])

    V_pos = np.zeros((H,W), dtype=np.float32)
    V_neg = np.zeros((H,W), dtype=np.float32)

    if idx_e <= idx_s:
        return np.stack([V_pos, V_neg])

    x = h5_file['events/x'][idx_s:idx_e].astype(np.int32)
    y = h5_file['events/y'][idx_s:idx_e].astype(np.int32)
    p = h5_file['events/p'][idx_s:idx_e].astype(np.bool_)

    xs = np.clip(x, 0, W-1)
    ys = np.clip(y, 0, H-1)

    for bi in np.array_split(np.arange(len(x)), n_bins):
        if len(bi)==0: continue
        V_pos *= decay; V_neg *= decay
        bx=xs[bi]; by=ys[bi]; bp=p[bi]
        np.add.at(V_pos,(by[bp], bx[bp]),  1.0)
        np.add.at(V_neg,(by[~bp],bx[~bp]), 1.0)

    pm=V_pos.max(); nm=V_neg.max()
    if pm>0: V_pos/=pm
    if nm>0: V_neg/=nm
    return np.stack([V_pos, V_neg])


def prepare_dsec_split(seqs, split_name, use_fusion=False):
    """
    Prepare YOLO format data from DSEC sequences.
    use_fusion: if True, create V3-style membrane overlay
    """
    suffix = '_fusion' if use_fusion else '_rgb'
    out_img = Path(f'{OUT_BASE}{suffix}/images/{split_name}')
    out_lbl = Path(f'{OUT_BASE}{suffix}/labels/{split_name}')
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    for seq in seqs:
        print(f'  Processing {seq}...')
        track  = np.load(f'{DSEC_BASE}/train/{seq}/object_detections/left/tracks.npy')
        map_x, map_y = load_rectify_map(seq)
        unique_ts = np.unique(track['t'])

        if use_fusion:
            h5 = h5py.File(f'{DSEC_BASE}/{seq}/events.h5','r')
            t_offset = float(h5['t_offset'][()])

        done = set(f.stem for f in out_img.glob('*.png'))

        for i, t_label in enumerate(tqdm(unique_ts, desc=f'    {seq}')):
            stem = f'{seq}_{i:04d}'
            if stem in done:
                continue

            # GT labels
            gt = track[track['t']==t_label]
            labels = []
            for d in gt:
                c = int(d['class_id'])
                if c not in DSEC_TO_OURS:
                    continue
                our_c = DSEC_TO_OURS[c]
                # normalize to event camera coords
                cx = (d['x'] + d['w']/2) / W_EVT
                cy = (d['y'] + d['h']/2) / H_EVT
                bw = d['w'] / W_EVT
                bh = d['h'] / H_EVT
                if 0<cx<1 and 0<cy<1 and bw>0.01 and bh>0.01:
                    labels.append(f'{our_c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')

            if not labels:
                continue

            # Remap RGB to event camera view
            img_path = f'{DSEC_BASE}/{seq}/{i:06d}.png'
            img_rgb  = cv2.imread(img_path)
            if img_rgb is None:
                continue

            img_evt = cv2.remap(img_rgb, map_x, map_y, cv2.INTER_LINEAR)

            if use_fusion:
                # Add membrane overlay
                mp = events_to_membrane(h5, t_offset, t_label)
                mp_pos = mp[0]; mp_neg = mp[1]

                # CLAHE
                lab = cv2.cvtColor(img_evt, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                lab[:,:,0] = clahe.apply(lab[:,:,0])
                img_enh = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

                # Membrane overlay
                mem = np.zeros((H_EVT,W_EVT,3),dtype=np.float32)
                mem[:,:,1] = mp_pos*255
                mem[:,:,2] = mp_neg*255

                img_out = cv2.addWeighted(
                    img_enh.astype(np.float32), 0.65,
                    mem, 0.35, 0).clip(0,255).astype(np.uint8)
            else:
                img_out = img_evt

            cv2.imwrite(str(out_img/f'{stem}.png'), img_out)
            with open(out_lbl/f'{stem}.txt','w') as f:
                f.write('\n'.join(labels))

        if use_fusion:
            h5.close()

    return out_img.parent.parent


def write_yaml(out_base, suffix):
    yaml_path = f'{OUT_BASE}{suffix}/data.yaml'
    with open(yaml_path,'w') as f:
        f.write(f"""path: {OUT_BASE}{suffix}
train: images/train
val:   images/val
nc: 2
names: ['vehicle','pedestrian']
""")
    return yaml_path


def train_on_dsec(use_fusion=False):
    suffix = '_fusion' if use_fusion else '_rgb'
    name   = f'dsec_v3{suffix}' if use_fusion else 'dsec_rgb'

    print(f'\n=== Preparing DSEC {name} ===')
    prepare_dsec_split(NIGHT_SEQS_TRAIN, 'train', use_fusion)
    prepare_dsec_split(NIGHT_SEQS_VAL,   'val',   use_fusion)

    yaml_path = write_yaml(OUT_BASE, suffix)

    print(f'\n=== Training {name} ===')
    model = YOLO('yolov8n.pt')
    model.train(
        data    = yaml_path,
        epochs  = 30,
        imgsz   = 640,
        batch   = 8,
        device  = 0,
        project = '/home/ivlab/carla_ws/fusion_model/runs',
        name    = name,
        exist_ok= True,
        workers = 0,
    )
    print(f'{name} done!')


if __name__ == '__main__':
    # Step 1: RGB-only baseline on DSEC
    train_on_dsec(use_fusion=False)
    # Step 2: V3 fusion on DSEC
    train_on_dsec(use_fusion=True)
