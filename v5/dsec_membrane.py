import sys
sys.path.insert(0, '/home/ivlab/carla_ws/fusion_model')

import h5py, hdf5plugin
import numpy as np, cv2, shutil
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

DSEC_BASE  = '/media/ivlab/8AC24E8FC24E7F85/dsec_detection'
OUT_BASE   = '/media/ivlab/8AC24E8FC24E7F85/dsec_membrane'
TRAIN_SEQS = ['zurich_city_04_a', 'zurich_city_04_b']
VAL_SEQS   = ['zurich_city_04_c']
DSEC_TO_OURS = {0:1, 1:0, 2:0}
W_EVT,H_EVT  = 640, 480


def events_to_membrane(h5f, t_offset, t_label,
                        window_us=50000, decay=0.9, n_bins=10):
    t_end   = t_label - t_offset
    t_start = max(0, t_end - window_us)
    ms_s = int(t_start/1000)
    ms_e = min(int(t_end/1000)+1, len(h5f['ms_to_idx'])-1)
    i_s  = int(h5f['ms_to_idx'][ms_s])
    i_e  = int(h5f['ms_to_idx'][ms_e])
    V_pos = np.zeros((H_EVT,W_EVT),dtype=np.float32)
    V_neg = np.zeros((H_EVT,W_EVT),dtype=np.float32)
    if i_e<=i_s: return V_pos, V_neg
    x = h5f['events/x'][i_s:i_e].astype(np.int32)
    y = h5f['events/y'][i_s:i_e].astype(np.int32)
    p = h5f['events/p'][i_s:i_e].astype(np.bool_)
    xs=np.clip(x,0,W_EVT-1); ys=np.clip(y,0,H_EVT-1)
    for bi in np.array_split(np.arange(len(x)),n_bins):
        if len(bi)==0: continue
        V_pos*=decay; V_neg*=decay
        bx=xs[bi]; by=ys[bi]; bp=p[bi]
        np.add.at(V_pos,(by[bp], bx[bp]),  1.0)
        np.add.at(V_neg,(by[~bp],bx[~bp]), 1.0)
    pm=V_pos.max(); nm=V_neg.max()
    if pm>0: V_pos/=pm
    if nm>0: V_neg/=nm
    return V_pos, V_neg


def prepare_split(seqs, split):
    oi = Path(f'{OUT_BASE}/images/{split}')
    ol = Path(f'{OUT_BASE}/labels/{split}')
    oi.mkdir(parents=True,exist_ok=True)
    ol.mkdir(parents=True,exist_ok=True)

    for seq in seqs:
        print(f'  {seq}...')
        track = np.load(
            f'{DSEC_BASE}/train/{seq}/object_detections/left/tracks.npy')
        unique_ts = np.unique(track['t'])
        h5f   = h5py.File(f'{DSEC_BASE}/{seq}/events.h5','r')
        t_off = float(h5f['t_offset'][()])
        done  = set(f.stem for f in oi.glob('*.png'))

        for i, t_lbl in enumerate(tqdm(unique_ts, desc=f'    {seq}')):
            stem = f'{seq}_{i:04d}'
            if stem in done: continue

            gt = track[track['t']==t_lbl]
            labels = []
            for d in gt:
                c = int(d['class_id'])
                if c not in DSEC_TO_OURS: continue
                oc = DSEC_TO_OURS[c]
                cx = (d['x']+d['w']/2)/W_EVT
                cy = (d['y']+d['h']/2)/H_EVT
                bw = d['w']/W_EVT
                bh = d['h']/H_EVT
                if 0<cx<1 and 0<cy<1 and bw>0.005 and bh>0.005:
                    labels.append(
                        f'{oc} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
            if not labels: continue

            img = cv2.imread(f'{DSEC_BASE}/{seq}/{i:06d}.png')
            if img is None: continue
            img_r = cv2.resize(img,(W_EVT,H_EVT))

            # CLAHE
            lab = cv2.cvtColor(img_r, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            img_enh = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # Membrane potential (真實 DVS！)
            V_pos, V_neg = events_to_membrane(h5f, t_off, t_lbl)

            # Polarity-aware overlay (V5 style)
            # pos → green channel (亮度增加)
            # neg → red channel (亮度減少)
            mem = np.zeros((H_EVT,W_EVT,3),dtype=np.float32)
            mem[:,:,1] = V_pos * 255  # green
            mem[:,:,2] = V_neg * 255  # red

            # Adaptive alpha based on event density
            evt_density = (V_pos + V_neg).mean()
            alpha_evt = min(0.5, evt_density * 3 + 0.2)
            alpha_rgb = 1.0 - alpha_evt

            img_out = cv2.addWeighted(
                img_enh.astype(np.float32), alpha_rgb,
                mem, alpha_evt, 0).clip(0,255).astype(np.uint8)

            cv2.imwrite(str(oi/f'{stem}.png'), img_out)
            with open(ol/f'{stem}.txt','w') as f:
                f.write('\n'.join(labels))

        h5f.close()


if __name__=='__main__':
    print('\n=== DSEC Membrane V5 Fusion ===')
    print('Using real DVS membrane potential (V_pos + V_neg)')
    print('Adaptive alpha based on event density')

    prepare_split(TRAIN_SEQS,'train')
    prepare_split(VAL_SEQS,  'val')

    yaml = f'{OUT_BASE}/data.yaml'
    with open(yaml,'w') as f:
        f.write(f"""path: {OUT_BASE}
train: images/train
val:   images/val
nc: 2
names: ['vehicle','pedestrian']
""")

    model = YOLO('yolov8n.pt')
    model.train(
        data=yaml, epochs=30, imgsz=640,
        batch=8, device=0,
        project='/home/ivlab/carla_ws/fusion_model/runs',
        name='dsec_membrane_v5',
        exist_ok=True, workers=0)
    print('Done!')
