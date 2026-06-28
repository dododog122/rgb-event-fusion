import sys
sys.path.insert(0, '/home/ivlab/carla_ws/fusion_model')

import numpy as np, cv2, shutil
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

BASE = '/media/ivlab/8AC24E8FC24E7F85/fusion_data'
H,W  = 720, 1280

def events_to_membrane(events, decay=0.9, n_bins=10):
    V_pos=np.zeros((H,W),dtype=np.float32)
    V_neg=np.zeros((H,W),dtype=np.float32)
    if len(events)==0: return V_pos, V_neg
    idx=np.argsort(events['t']); evts=events[idx]
    xs=np.clip(evts['x'].astype(np.int32),0,W-1)
    ys=np.clip(evts['y'].astype(np.int32),0,H-1)
    pols=evts['pol'].astype(np.bool_)
    for bi in np.array_split(np.arange(len(evts)),n_bins):
        if len(bi)==0: continue
        V_pos*=decay; V_neg*=decay
        bx=xs[bi]; by=ys[bi]; bp=pols[bi]
        np.add.at(V_pos,(by[bp], bx[bi[bp-bi[0]]  if False else bp]),1.0)
        np.add.at(V_neg,(by[~bp],bx[~bp]),1.0)
    pm=V_pos.max(); nm=V_neg.max()
    if pm>0: V_pos/=pm
    if nm>0: V_neg/=nm
    return V_pos, V_neg

def gen(split, rgb_dir, dvs_dir, lbl_dir, out_base):
    out_img = Path(f'{out_base}/images/{split}')
    out_lbl = Path(f'{out_base}/labels/{split}')
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(rgb_dir).glob('frame_*.png'))
    done  = set(f.stem for f in out_img.glob('*.png'))
    todo  = [f for f in files if f.stem not in done]
    print(f'{split}: {len(todo)} remaining')

    for rgb_path in tqdm(todo):
        stem = rgb_path.stem
        dp   = Path(dvs_dir)/f'{stem}.npy'
        lp   = Path(lbl_dir)/f'{stem}.txt'
        if not dp.exists() or not lp.exists(): continue

        # CLAHE RGB
        rgb = cv2.imread(str(rgb_path))
        rgb = cv2.resize(rgb,(640,640))
        lab = cv2.cvtColor(rgb,cv2.COLOR_BGR2LAB)
        clahe=cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
        lab[:,:,0]=clahe.apply(lab[:,:,0])
        rgb_enh=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)

        # Membrane potential
        events=np.load(str(dp))
        V_pos,V_neg=events_to_membrane(events)
        mp_pos=cv2.resize(V_pos,(640,640))
        mp_neg=cv2.resize(V_neg,(640,640))

        # Membrane overlay
        mem=np.zeros((640,640,3),dtype=np.float32)
        mem[:,:,1]=mp_pos*255  # green=V_pos
        mem[:,:,2]=mp_neg*255  # red=V_neg

        fused=cv2.addWeighted(
            rgb_enh.astype(np.float32),0.65,
            mem,0.35,0).clip(0,255).astype(np.uint8)

        cv2.imwrite(str(out_img/f'{stem}.png'),fused)
        shutil.copy(str(lp),str(out_lbl/f'{stem}.txt'))

out_base = '/home/ivlab/carla_ws/fusion_model/fusion_membrane'

gen('train',
    f'{BASE}/morning/rgb',
    f'{BASE}/morning/dvs',
    f'{BASE}/morning/yolo/labels',
    out_base)

gen('val',
    f'{BASE}/night/rgb',
    f'{BASE}/night/dvs',
    f'{BASE}/night/yolo/labels',
    out_base)

with open(f'{out_base}/data.yaml','w') as f:
    f.write(f"""path: {out_base}
train: images/train
val:   images/val
nc: 4
names: ['vehicle','pedestrian','traffic_sign','traffic_light']
""")

model=YOLO('yolov8n.pt')
model.train(
    data=f'{out_base}/data.yaml',
    epochs=50, imgsz=640, batch=8, device=0,
    project='/home/ivlab/carla_ws/fusion_model/runs',
    name='fusion_membrane_ultra',
    exist_ok=True, workers=0)
print('Done!')
