"""
Full SLAM Comparison Video
Sequences: zurich_city_04_b + 04_c + 04_d
Total: ~2400 frames, pedestrian-rich scenes
左：RGB-only ORB 特徵追蹤
右：RGB+Event Membrane Fusion 特徵追蹤
底：特徵點歷史圖 + 統計 + 環境重建對比
"""
import cv2, numpy as np, h5py, hdf5plugin
from pathlib import Path
from tqdm import tqdm

BASE = '/media/ivlab/8AC24E8FC24E7F85/dsec_detection'
OUT  = '/home/ivlab/carla_ws/fusion_model/slam_full.mp4'
H, W = 480, 640
FPS  = 15   # 流暢

SEQS = [
    ('zurich_city_04_b', 269),
    ('zurich_city_04_c', 1181),
    ('zurich_city_04_d', 957),
]

def events_to_membrane(h5f, t_offset, t_label, decay=0.9, n_bins=10):
    t_end   = t_label - t_offset
    t_start = max(0, t_end - 50000)
    ms_s = int(t_start/1000)
    ms_e = min(int(t_end/1000)+1, len(h5f['ms_to_idx'])-1)
    i_s  = int(h5f['ms_to_idx'][ms_s])
    i_e  = int(h5f['ms_to_idx'][ms_e])
    Vp = np.zeros((H,W),dtype=np.float32)
    Vn = np.zeros((H,W),dtype=np.float32)
    if i_e<=i_s: return Vp,Vn
    x=h5f['events/x'][i_s:i_e].astype(np.int32)
    y=h5f['events/y'][i_s:i_e].astype(np.int32)
    p=h5f['events/p'][i_s:i_e].astype(np.bool_)
    xs=np.clip(x,0,W-1); ys=np.clip(y,0,H-1)
    for bi in np.array_split(np.arange(len(x)),n_bins):
        if len(bi)==0: continue
        Vp*=decay; Vn*=decay
        bx=xs[bi]; by=ys[bi]; bp=p[bi]
        np.add.at(Vp,(by[bp],bx[bp]),1.)
        np.add.at(Vn,(by[~bp],bx[~bp]),1.)
    pm=Vp.max(); nm=Vn.max()
    if pm>0: Vp/=pm
    if nm>0: Vn/=nm
    return Vp,Vn

def make_fusion(rgb, Vp, Vn):
    lab=cv2.cvtColor(rgb,cv2.COLOR_BGR2LAB)
    clahe=cv2.createCLAHE(3.0,(8,8))
    lab[:,:,0]=clahe.apply(lab[:,:,0])
    enh=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
    mem=np.zeros((H,W,3),dtype=np.float32)
    mem[:,:,1]=Vp*255; mem[:,:,2]=Vn*255
    return cv2.addWeighted(enh.astype(np.float32),0.65,
                           mem,0.35,0).clip(0,255).astype(np.uint8)

# ── ORB: 不設上限，讓暗部差異自然出現 ──
orb_rgb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8,
                          edgeThreshold=31, fastThreshold=20)
orb_fus = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8,
                          edgeThreshold=10, fastThreshold=7)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# ── Loop Closure Detector ──
class LCDetector:
    def __init__(self, thr=0.65):
        self.db  = []
        self.thr = thr
        self.bfm = cv2.BFMatcher(cv2.NORM_HAMMING)
    def add(self, idx, des):
        if des is not None and len(des)>20:
            self.db.append((idx, des))
    def detect(self, idx, des, gap=40):
        if des is None or len(des)<20 or len(self.db)<gap:
            return None, 0.0
        best_score=0; best_idx=None
        cands=[(i,d) for i,d in self.db if idx-i>=gap]
        for prev_idx,prev_des in cands[-15:]:
            try:
                ms=self.bfm.knnMatch(des,prev_des,k=2)
                good=[m for m,n in ms if len([m,n])==2
                      and m.distance<self.thr*n.distance]
                score=len(good)/max(len(des),len(prev_des))
                if score>best_score:
                    best_score=score; best_idx=prev_idx
            except: pass
        return (best_idx,best_score) if best_score>0.12 else (None,0.0)

# ── Dense Reconstruction Map ──
class ReconMap:
    """Bird's-eye-view cumulative depth map"""
    def __init__(self, size=200):
        self.s  = size
        self.mp = np.zeros((size,size,3),dtype=np.uint8)+25

    def update(self, depth_row, color_row, offset_x, label):
        # depth_row: array of depth values for bottom of image
        # Project onto bird-eye-view
        for j, (d, c) in enumerate(zip(depth_row, color_row)):
            if d <= 0 or d > 25: continue
            px = int(offset_x % self.s)
            py = int((1 - d/25) * (self.s-10)) + 5
            px = max(0, min(self.s-1, px))
            py = max(0, min(self.s-1, py))
            self.mp[py, px] = c[::-1]  # BGR→RGB

    def render(self, title):
        vis = self.mp.copy()
        cv2.putText(vis, title, (4,14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220,220,220), 1)
        return vis

recon_rgb = ReconMap(200)
recon_fus = ReconMap(200)

# ── Depth from disparity ──
def get_depth_row(disp_path, row=420):
    """Get depth for one horizontal row"""
    if not Path(disp_path).exists():
        return None, None
    di = cv2.imread(disp_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)/256.0
    di_r = cv2.resize(di,(W,H),interpolation=cv2.INTER_NEAREST)
    row_disp = di_r[row,:]
    depth = np.zeros_like(row_disp)
    valid = row_disp > 0
    depth[valid] = 0.6*223.6/row_disp[valid]
    return depth, row

# ── Video writer ──
VW, VH = W*2, H+160
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUT, fourcc, FPS, (VW, VH))

kp_h_rgb=[]; kp_h_fus=[]
mt_h_rgb=[]; mt_h_fus=[]
lc_rgb=[]; lc_fus=[]
global_frame = 0

lc_det_rgb = LCDetector()
lc_det_fus = LCDetector()
prev_des_rgb=None; prev_kp_rgb=None
prev_des_fus=None; prev_kp_fus=None

print("Processing sequences: 04_b + 04_c + 04_d")

for seq_name, seq_len in SEQS:
    print(f"\n=== {seq_name} ({seq_len} frames) ===")
    img_dir  = Path(f'{BASE}/{seq_name}')
    h5_path  = f'{BASE}/{seq_name}/events.h5'
    lbl_path = f'{BASE}/train/{seq_name}/object_detections/left/tracks.npy'

    if not Path(h5_path).exists():
        print(f"  No events.h5, skipping"); continue

    h5f      = h5py.File(h5_path,'r')
    t_offset = float(h5f['t_offset'][()])
    track    = np.load(lbl_path)
    unique_ts = np.unique(track['t'])

    disp_evt_dir = Path(f'{BASE}/{seq_name}/disparity_event')
    disp_img_dir = Path(f'{BASE}/{seq_name}/disparity_image')
    has_disp = disp_evt_dir.exists() and disp_img_dir.exists()

    for i in tqdm(range(min(seq_len, len(unique_ts))), desc=seq_name):
        t_lbl = unique_ts[i]
        img_path = img_dir/f'{i:06d}.png'
        if not img_path.exists(): continue

        img = cv2.imread(str(img_path))
        if img is None: continue
        img = cv2.resize(img,(W,H))

        Vp,Vn = events_to_membrane(h5f, t_offset, t_lbl)
        fus   = make_fusion(img.copy(), Vp, Vn)

        gray_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_fus = cv2.cvtColor(fus, cv2.COLOR_BGR2GRAY)

        kp_rgb, des_rgb = orb_rgb.detectAndCompute(gray_rgb, None)
        kp_fus, des_fus = orb_fus.detectAndCompute(gray_fus, None)

        n_rgb=len(kp_rgb); n_fus=len(kp_fus)
        kp_h_rgb.append(n_rgb); kp_h_fus.append(n_fus)

        # Match
        n_m_rgb=0; n_m_fus=0; mr=[]; mf=[]
        if prev_des_rgb is not None and des_rgb is not None and len(des_rgb)>5:
            try:
                m=bf.match(prev_des_rgb,des_rgb)
                mr=sorted(m,key=lambda x:x.distance)[:80]; n_m_rgb=len(mr)
            except: pass
        if prev_des_fus is not None and des_fus is not None and len(des_fus)>5:
            try:
                m=bf.match(prev_des_fus,des_fus)
                mf=sorted(m,key=lambda x:x.distance)[:80]; n_m_fus=len(mf)
            except: pass
        mt_h_rgb.append(n_m_rgb); mt_h_fus.append(n_m_fus)

        # Loop closure
        lc_det_rgb.add(global_frame, des_rgb)
        lc_det_fus.add(global_frame, des_fus)
        lci_r, lcs_r = lc_det_rgb.detect(global_frame, des_rgb)
        lci_f, lcs_f = lc_det_fus.detect(global_frame, des_fus)
        if lci_r: lc_rgb.append((global_frame, lcs_r))
        if lci_f: lc_fus.append((global_frame, lcs_f))

        # Depth reconstruction
        if has_disp and i%2==0:
            disp_e = disp_evt_dir/f'{i:06d}.png'
            disp_i = disp_img_dir/f'{i:06d}.png'
            if disp_e.exists():
                de=cv2.imread(str(disp_e),cv2.IMREAD_ANYDEPTH).astype(np.float32)/256.0
                di=cv2.imread(str(disp_i),cv2.IMREAD_ANYDEPTH).astype(np.float32)/256.0
                de_r=cv2.resize(de,(W,H),interpolation=cv2.INTER_NEAREST)
                di_r=cv2.resize(di,(W,H),interpolation=cv2.INTER_NEAREST)
                # bottom row depth
                row=420
                row_rgb_img=img[row,:]
                row_fus_img=fus[row,:]
                depth_e = np.where(de_r[row]>0, 0.6*223.6/de_r[row], 0)
                depth_i = np.where(di_r[row]>0, 0.6*223.6/di_r[row], 0)
                recon_fus.update(depth_e, row_fus_img, i, 'Fusion')
                recon_rgb.update(depth_i, row_rgb_img, i, 'RGB')

        # ── Visualize ──
        vis_rgb = img.copy()
        vis_fus = fus.copy()

        # Track lines
        if mr and prev_kp_rgb:
            for m in mr[:60]:
                p1=tuple(map(int,prev_kp_rgb[m.queryIdx].pt))
                p2=tuple(map(int,kp_rgb[m.trainIdx].pt))
                cv2.line(vis_rgb,p1,p2,(0,180,0),1,cv2.LINE_AA)
        if mf and prev_kp_fus:
            for m in mf[:60]:
                p1=tuple(map(int,prev_kp_fus[m.queryIdx].pt))
                p2=tuple(map(int,kp_fus[m.trainIdx].pt))
                cv2.line(vis_fus,p1,p2,(60,60,220),1,cv2.LINE_AA)

        # Keypoints (size by response)
        for k in kp_rgb:
            r=max(2,min(int(k.response*4),6))
            cv2.circle(vis_rgb,tuple(map(int,k.pt)),r,(0,255,0),-1,cv2.LINE_AA)
        for k in kp_fus:
            r=max(2,min(int(k.response*4),6))
            cv2.circle(vis_fus,tuple(map(int,k.pt)),r,(80,80,255),-1,cv2.LINE_AA)

        # Loop closure indicator
        if lci_r:
            cv2.rectangle(vis_rgb,(2,2),(W-2,H-2),(0,255,255),4)
            cv2.putText(vis_rgb,f'LOOP CLOSURE! {lcs_r:.2f}',
                        (8,H-12),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        if lci_f:
            cv2.rectangle(vis_fus,(2,2),(W-2,H-2),(0,255,255),4)
            cv2.putText(vis_fus,f'LOOP CLOSURE! {lcs_f:.2f}',
                        (8,H-12),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

        # Membrane thumbnail
        mem_vis=np.zeros((H,W,3),dtype=np.uint8)
        mem_vis[:,:,1]=(Vp*220).astype(np.uint8)
        mem_vis[:,:,2]=(Vn*220).astype(np.uint8)
        th=cv2.resize(mem_vis,(150,85))
        vis_fus[H-90:H-5,W-155:W-5]=th
        cv2.rectangle(vis_fus,(W-155,H-90),(W-5,H-5),(80,80,80),1)
        cv2.putText(vis_fus,'DVS Membrane',(W-152,H-92),
                    cv2.FONT_HERSHEY_SIMPLEX,0.3,(160,160,160),1)

        # Info bars
        for vis,name,nkp,nm,col in [
            (vis_rgb,'RGB-only',n_rgb,n_m_rgb,(50,220,50)),
            (vis_fus,'RGB+Event Fusion',n_fus,n_m_fus,(80,100,255))]:
            cv2.rectangle(vis,(0,0),(W,62),(12,12,12),-1)
            cv2.putText(vis,name,(8,20),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2)
            cv2.putText(vis,f'KP: {nkp:4d}  Match: {nm:3d}',(8,42),
                        cv2.FONT_HERSHEY_SIMPLEX,0.42,(210,210,210),1)
            cv2.putText(vis,f'{seq_name} | {i:04d}',(W-160,20),
                        cv2.FONT_HERSHEY_SIMPLEX,0.36,(130,130,130),1)
            bw=int(min(nkp/3000,1.)*(W-4))
            cv2.rectangle(vis,(2,60),(2+bw,68),col,-1)
            cv2.rectangle(vis,(2,60),(W-2,68),(40,40,40),1)

        # ── Bottom panel (160px) ──
        bot = np.zeros((160,VW,3),dtype=np.uint8)+15

        # KP graph (left 600px)
        GW,GH=580,130
        g=np.zeros((GH,GW,3),dtype=np.uint8)+22
        recent=100
        h_r=kp_h_rgb[-recent:]; h_f=kp_h_fus[-recent:]
        mx=max(max(h_r+[1]),max(h_f+[1]))
        for j in range(1,min(len(h_r),GW)):
            yr =GH-4-int(h_r[j-1]/mx*(GH-8))
            yr2=GH-4-int(h_r[j  ]/mx*(GH-8)) if j<len(h_r) else yr
            yf =GH-4-int(h_f[j-1]/mx*(GH-8))
            yf2=GH-4-int(h_f[j  ]/mx*(GH-8)) if j<len(h_f) else yf
            cv2.line(g,(j,yr),(j+1,yr2),(50,200,50),1)
            cv2.line(g,(j,yf),(j+1,yf2),(80,80,220),1)
        # LC markers
        offset=max(0,global_frame-recent)
        for lf,_ in lc_rgb:
            x=lf-offset
            if 0<=x<GW: cv2.line(g,(x,0),(x,GH),(0,200,200),1)
        for lf,_ in lc_fus:
            x=lf-offset
            if 0<=x<GW: cv2.line(g,(x,0),(x,GH),(0,150,200),1)
        cv2.putText(g,'Keypoints (last 100 frames)',(4,11),
                    cv2.FONT_HERSHEY_SIMPLEX,0.33,(170,170,170),1)
        cv2.putText(g,f'RGB:{np.mean(kp_h_rgb):.0f} avg',(4,GH-3),
                    cv2.FONT_HERSHEY_SIMPLEX,0.32,(50,200,50),1)
        cv2.putText(g,f'Fus:{np.mean(kp_h_fus):.0f} avg',(130,GH-3),
                    cv2.FONT_HERSHEY_SIMPLEX,0.32,(80,80,220),1)
        cv2.putText(g,f'LC RGB:{len(lc_rgb)} Fus:{len(lc_fus)}',(280,GH-3),
                    cv2.FONT_HERSHEY_SIMPLEX,0.32,(0,200,200),1)
        bot[15:15+GH,8:8+GW]=g

        # Recon maps (middle)
        rc_r=cv2.resize(recon_rgb.render('RGB Depth Map'),(150,140))
        rc_f=cv2.resize(recon_fus.render('Fusion Depth Map'),(150,140))
        bot[10:150,GW+20:GW+170]=rc_r
        bot[10:150,GW+180:GW+330]=rc_f

        # Stats (right)
        sx=GW+340
        diff=n_fus-n_rgb
        sign='+' if diff>=0 else ''
        cv2.putText(bot,'Current:',(sx,16),
                    cv2.FONT_HERSHEY_SIMPLEX,0.4,(190,190,190),1)
        cv2.putText(bot,f'RGB  KP:{n_rgb:4d}',(sx,32),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,(50,200,50),1)
        cv2.putText(bot,f'Fus  KP:{n_fus:4d}',(sx,48),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,(80,80,220),1)
        cv2.putText(bot,f'Diff:  {sign}{diff}',(sx,64),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,(200,200,80),1)
        cv2.putText(bot,f'Match RGB:{n_m_rgb}',(sx,80),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,(50,200,50),1)
        cv2.putText(bot,f'Match Fus:{n_m_fus}',(sx,96),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,(80,80,220),1)
        fus_x=np.mean(kp_h_fus)/max(np.mean(kp_h_rgb),1)
        cv2.putText(bot,f'Fusion x{fus_x:.2f} KP',(sx,115),
                    cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,200,80),1)
        cv2.putText(bot,f'LC: RGB={len(lc_rgb)} Fus={len(lc_fus)}',(sx,135),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,200,200),1)

        frame=np.vstack([np.hstack([vis_rgb,vis_fus]),bot])
        writer.write(frame)

        prev_des_rgb=des_rgb; prev_kp_rgb=kp_rgb
        prev_des_fus=des_fus; prev_kp_fus=kp_fus
        global_frame+=1

    h5f.close()

writer.release()

# Convert to H264
import subprocess
out_h264 = OUT.replace('.mp4','_h264.mp4')
subprocess.run(['ffmpeg','-i',OUT,'-vcodec','libx264',
                '-pix_fmt','yuv420p','-crf','23','-preset','fast',
                out_h264,'-y'], check=True)

print(f'\nDone! {out_h264}')
print(f'Total frames:    {global_frame}')
print(f'RGB  avg KP:     {np.mean(kp_h_rgb):.1f}')
print(f'Fusion avg KP:   {np.mean(kp_h_fus):.1f}')
print(f'RGB  avg Match:  {np.mean(mt_h_rgb):.1f}')
print(f'Fusion avg Match:{np.mean(mt_h_fus):.1f}')
print(f'Loop Closure RGB:    {len(lc_rgb)}')
print(f'Loop Closure Fusion: {len(lc_fus)}')
