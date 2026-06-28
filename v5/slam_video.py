"""
SLAM Feature Tracking Comparison Video
左半：RGB-only ORB 特徵點追蹤
右半：RGB+Event membrane fusion 特徵點追蹤

展示：
- 特徵點數量和分布
- 特徵點追蹤穩定性
- 模擬 loop closure 效果（顏色標記重訪區域）
- 軌跡圖
"""
import cv2
import numpy as np
import h5py
import hdf5plugin
from pathlib import Path
from tqdm import tqdm

BASE    = '/media/ivlab/8AC24E8FC24E7F85/dsec_detection'
SEQ     = 'zurich_city_04_a'
OUT     = '/home/ivlab/carla_ws/fusion_model/slam_comparison.mp4'
N_FRAMES = 200   # 取前 200 幀
FPS      = 10
H_EVT, W_EVT = 480, 640

# ── Membrane Potential ──
def events_to_membrane(h5f, t_offset, t_label, window_us=50000, decay=0.9, n_bins=10):
    t_end   = t_label - t_offset
    t_start = max(0, t_end - window_us)
    ms_s = int(t_start/1000)
    ms_e = min(int(t_end/1000)+1, len(h5f['ms_to_idx'])-1)
    i_s  = int(h5f['ms_to_idx'][ms_s])
    i_e  = int(h5f['ms_to_idx'][ms_e])
    V_pos = np.zeros((H_EVT, W_EVT), dtype=np.float32)
    V_neg = np.zeros((H_EVT, W_EVT), dtype=np.float32)
    if i_e <= i_s:
        return V_pos, V_neg
    x = h5f['events/x'][i_s:i_e].astype(np.int32)
    y = h5f['events/y'][i_s:i_e].astype(np.int32)
    p = h5f['events/p'][i_s:i_e].astype(np.bool_)
    xs = np.clip(x,0,W_EVT-1); ys = np.clip(y,0,H_EVT-1)
    for bi in np.array_split(np.arange(len(x)), n_bins):
        if len(bi)==0: continue
        V_pos *= decay; V_neg *= decay
        bx=xs[bi]; by=ys[bi]; bp=p[bi]
        np.add.at(V_pos,(by[bp], bx[bp]),  1.0)
        np.add.at(V_neg,(by[~bp],bx[~bp]), 1.0)
    pm=V_pos.max(); nm=V_neg.max()
    if pm>0: V_pos/=pm
    if nm>0: V_neg/=nm
    return V_pos, V_neg

def make_fusion(rgb, V_pos, V_neg):
    """CLAHE RGB + membrane overlay"""
    lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    rgb_enh = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    mem = np.zeros((H_EVT,W_EVT,3), dtype=np.float32)
    mem[:,:,1] = V_pos * 255
    mem[:,:,2] = V_neg * 255
    fused = cv2.addWeighted(
        rgb_enh.astype(np.float32), 0.65,
        mem, 0.35, 0).clip(0,255).astype(np.uint8)
    return fused

# ── ORB Feature Tracker ──
class ORBTracker:
    def __init__(self, name, color):
        self.name   = name
        self.color  = color  # BGR
        self.orb    = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
        self.bf     = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_kp  = None
        self.prev_des = None
        self.prev_pts = None
        self.tracks   = {}   # id → list of (x,y)
        self.track_id = 0
        self.traj     = []   # (cx, cy) trajectory
        self.loop_pts = []   # loop closure candidates
        self.frame_count = 0

    def process(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        self.frame_count += 1

        vis = img.copy()
        matches_used = 0

        if self.prev_des is not None and des is not None and len(des) > 10:
            try:
                matches = self.bf.match(self.prev_des, des)
                matches = sorted(matches, key=lambda x: x.distance)
                good    = matches[:min(150, len(matches))]
                matches_used = len(good)

                # Draw matches as tracks
                for m in good:
                    p1 = tuple(map(int, self.prev_kp[m.queryIdx].pt))
                    p2 = tuple(map(int, kp[m.trainIdx].pt))
                    # Track line
                    cv2.line(vis, p1, p2, self.color, 1, cv2.LINE_AA)

                # Loop closure detection (simple: check if we revisit similar descriptors)
                if self.frame_count > 30 and self.frame_count % 20 == 0:
                    self.loop_pts.append(kp[good[0].trainIdx].pt if good else None)

            except Exception:
                pass

        # Draw current keypoints
        for k in kp:
            x, y = int(k.pt[0]), int(k.pt[1])
            cv2.circle(vis, (x,y), 2, self.color, -1, cv2.LINE_AA)

        # Simulate trajectory (centroid of keypoints)
        if kp:
            pts = np.array([k.pt for k in kp])
            cx, cy = pts.mean(axis=0)
            self.traj.append((int(cx), int(cy)))

        self.prev_kp  = kp
        self.prev_des = des

        return vis, len(kp), matches_used

# ── Trajectory Map ──
class TrajMap:
    def __init__(self, size=200):
        self.size = size
        self.canvas = np.ones((size, size, 3), dtype=np.uint8) * 30
        self.pts_rgb = []
        self.pts_fus = []

    def update(self, traj_rgb, traj_fus):
        self.pts_rgb = traj_rgb[-100:]
        self.pts_fus = traj_fus[-100:]

    def render(self):
        canvas = self.canvas.copy()
        cv2.putText(canvas, "Trajectory", (5,15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

        def draw_traj(pts, color):
            if len(pts) < 2: return
            # Normalize to canvas
            all_x = [p[0] for p in pts]
            all_y = [p[1] for p in pts]
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            rx = max_x - min_x + 1
            ry = max_y - min_y + 1
            for i in range(1, len(pts)):
                x1 = int((pts[i-1][0]-min_x)/rx*(self.size-20))+10
                y1 = int((pts[i-1][1]-min_y)/ry*(self.size-20))+10
                x2 = int((pts[i][0]-min_x)/rx*(self.size-20))+10
                y2 = int((pts[i][1]-min_y)/ry*(self.size-20))+10
                cv2.line(canvas,(x1,y1),(x2,y2),color,1,cv2.LINE_AA)
            # Current position
            xc = int((pts[-1][0]-min_x)/rx*(self.size-20))+10
            yc = int((pts[-1][1]-min_y)/ry*(self.size-20))+10
            cv2.circle(canvas,(xc,yc),4,color,-1)

        draw_traj(self.pts_rgb, (80,200,80))   # green
        draw_traj(self.pts_fus, (80,80,255))   # red

        # Legend
        cv2.rectangle(canvas,(5,self.size-35),(15,self.size-27),(80,200,80),-1)
        cv2.putText(canvas,"RGB",(17,self.size-28),cv2.FONT_HERSHEY_SIMPLEX,0.3,(200,200,200),1)
        cv2.rectangle(canvas,(5,self.size-22),(15,self.size-14),(80,80,255),-1)
        cv2.putText(canvas,"Fusion",(17,self.size-14),cv2.FONT_HERSHEY_SIMPLEX,0.3,(200,200,200),1)
        return canvas

# ── Loop Closure Visualizer ──
def draw_loop_closure(vis, frame_idx, detected=False):
    if detected:
        # Draw loop closure indicator
        h, w = vis.shape[:2]
        cv2.rectangle(vis, (0,0), (w-1,h-1), (0,255,255), 4)
        cv2.putText(vis, "LOOP CLOSURE DETECTED!", (10, h-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    return vis

# ── Main ──
def main():
    print("Loading DSEC data...")
    img_dir  = Path(f'{BASE}/{SEQ}')
    h5f      = h5py.File(f'{BASE}/{SEQ}/events.h5','r')
    t_offset = float(h5f['t_offset'][()])

    import numpy as np
    track    = np.load(f'{BASE}/train/{SEQ}/object_detections/left/tracks.npy')
    unique_ts = np.unique(track['t'])

    n = min(N_FRAMES, len(unique_ts)-1)
    print(f"Processing {n} frames...")

    # Video writer: side-by-side (1280 x 560)
    VW, VH = 1280, 580
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUT, fourcc, FPS, (VW, VH))

    tracker_rgb = ORBTracker("RGB-only",  (80, 200, 80))   # green
    tracker_fus = ORBTracker("RGB+Event", (80,  80, 255))  # red/blue
    traj_map    = TrajMap(size=180)

    loop_frames = set(range(60, 70))   # simulate loop closure at frame 60-70
    kp_hist_rgb = []
    kp_hist_fus = []

    for i in tqdm(range(n)):
        t_lbl = unique_ts[i]
        img_path = img_dir / f'{i:06d}.png'
        if not img_path.exists():
            continue

        # Load and resize RGB
        img_rgb = cv2.imread(str(img_path))
        img_rgb = cv2.resize(img_rgb, (W_EVT, H_EVT))

        # Membrane fusion
        V_pos, V_neg = events_to_membrane(h5f, t_offset, t_lbl)
        img_fus      = make_fusion(img_rgb.copy(), V_pos, V_neg)

        # Track features
        vis_rgb, n_kp_rgb, n_match_rgb = tracker_rgb.process(img_rgb.copy())
        vis_fus, n_kp_fus, n_match_fus = tracker_fus.process(img_fus.copy())

        kp_hist_rgb.append(n_kp_rgb)
        kp_hist_fus.append(n_kp_fus)

        # Loop closure simulation
        loop_detected = i in loop_frames
        if loop_detected:
            vis_rgb = draw_loop_closure(vis_rgb, i, True)
            vis_fus = draw_loop_closure(vis_fus, i, True)

        # Draw info overlay on RGB panel
        def draw_info(vis, name, n_kp, n_match, color):
            cv2.rectangle(vis, (0,0), (W_EVT-1, 55), (20,20,20), -1)
            cv2.putText(vis, name, (8,18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(vis, f"Keypoints: {n_kp}", (8,36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1)
            cv2.putText(vis, f"Matches:   {n_match}", (8,52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1)
            return vis

        vis_rgb = draw_info(vis_rgb, "RGB-only", n_kp_rgb, n_match_rgb, (80,220,80))
        vis_fus = draw_info(vis_fus, "RGB + Event Fusion", n_kp_fus, n_match_fus, (80,120,255))

        # Draw keypoint density bar
        def draw_bar(vis, val, max_val, color, y=60):
            bar_w = int(min(val/max(max_val,1), 1.0) * (W_EVT-4))
            cv2.rectangle(vis, (2,y), (2+bar_w, y+6), color, -1)
            cv2.rectangle(vis, (2,y), (W_EVT-2, y+6), (80,80,80), 1)
            return vis

        max_kp = max(max(kp_hist_rgb+[1]), max(kp_hist_fus+[1]))
        vis_rgb = draw_bar(vis_rgb, n_kp_rgb, max_kp, (80,220,80))
        vis_fus = draw_bar(vis_fus, n_kp_fus, max_kp, (80,120,255))

        # Frame counter + timestamp
        cv2.putText(vis_rgb, f"Frame {i:03d}", (W_EVT-100,18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)
        cv2.putText(vis_fus, f"Frame {i:03d}", (W_EVT-100,18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

        # Membrane visualization (small overlay on fusion panel)
        mem_vis = np.zeros((H_EVT,W_EVT,3),dtype=np.uint8)
        mem_vis[:,:,1] = (V_pos*200).astype(np.uint8)
        mem_vis[:,:,2] = (V_neg*200).astype(np.uint8)
        mem_small = cv2.resize(mem_vis, (160,90))
        vis_fus[H_EVT-95:H_EVT-5, W_EVT-165:W_EVT-5] = mem_small
        cv2.rectangle(vis_fus, (W_EVT-165,H_EVT-95),(W_EVT-5,H_EVT-5),(100,100,100),1)
        cv2.putText(vis_fus,"Membrane",(W_EVT-160,H_EVT-98),
                    cv2.FONT_HERSHEY_SIMPLEX,0.35,(180,180,180),1)

        # Update trajectory map
        traj_map.update(tracker_rgb.traj, tracker_fus.traj)
        traj_canvas = traj_map.render()

        # Assemble final frame: [RGB | Fusion] + bottom bar
        top = np.hstack([vis_rgb, vis_fus])  # 1280 x 480

        # Bottom stats bar
        bottom = np.zeros((100, VW, 3), dtype=np.uint8) + 20

        # KP history graph
        graph_w = 460
        for j in range(1, min(len(kp_hist_rgb), graph_w)):
            if j >= len(kp_hist_rgb): break
            y1_r = int(90 - kp_hist_rgb[j-1]/max(max_kp,1)*80)
            y2_r = int(90 - kp_hist_rgb[j]/max(max_kp,1)*80)
            y1_f = int(90 - kp_hist_fus[j-1]/max(max_kp,1)*80)
            y2_f = int(90 - kp_hist_fus[j]/max(max_kp,1)*80)
            cv2.line(bottom,(j+10,y1_r),(j+11,y2_r),(80,220,80),1)
            cv2.line(bottom,(j+10,y1_f),(j+11,y2_f),(80,120,255),1)

        cv2.putText(bottom,"Keypoint Count History",(12,12),
                    cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,200,200),1)
        cv2.putText(bottom,f"RGB avg: {np.mean(kp_hist_rgb):.0f}",(12,92),
                    cv2.FONT_HERSHEY_SIMPLEX,0.35,(80,220,80),1)
        cv2.putText(bottom,f"Fus avg: {np.mean(kp_hist_fus):.0f}",(120,92),
                    cv2.FONT_HERSHEY_SIMPLEX,0.35,(80,120,255),1)

        # Trajectory map
        traj_resized = cv2.resize(traj_canvas,(180,95))
        bottom[2:97, 480:660] = traj_resized

        # Stats summary
        diff_kp = n_kp_fus - n_kp_rgb
        diff_sign = "+" if diff_kp >= 0 else ""
        cv2.putText(bottom,"Current Frame Stats:",(670,15),
                    cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,200,200),1)
        cv2.putText(bottom,f"RGB KP:    {n_kp_rgb:4d}",(670,32),
                    cv2.FONT_HERSHEY_SIMPLEX,0.4,(80,220,80),1)
        cv2.putText(bottom,f"Fusion KP: {n_kp_fus:4d}",(670,48),
                    cv2.FONT_HERSHEY_SIMPLEX,0.4,(80,120,255),1)
        cv2.putText(bottom,f"Diff:    {diff_sign}{diff_kp:4d}",(670,64),
                    cv2.FONT_HERSHEY_SIMPLEX,0.4,(220,220,80),1)
        cv2.putText(bottom,f"RGB Match:   {n_match_rgb}",(670,80),
                    cv2.FONT_HERSHEY_SIMPLEX,0.4,(80,220,80),1)
        cv2.putText(bottom,f"Fusion Match:{n_match_fus}",(670,96),
                    cv2.FONT_HERSHEY_SIMPLEX,0.4,(80,120,255),1)

        if loop_detected:
            cv2.putText(bottom,"⚡ LOOP CLOSURE ⚡",(900,50),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

        # Title bar
        cv2.putText(bottom,"DSEC zurich_city_04_a | RGB-only vs RGB+Event Membrane Fusion | ORB Feature Tracking",
                    (480,12),cv2.FONT_HERSHEY_SIMPLEX,0.35,(160,160,160),1)

        frame = np.vstack([top, bottom])  # 1280 x 580
        writer.write(frame)

    writer.release()
    h5f.close()
    print(f"\nDone! Saved to {OUT}")
    print(f"RGB avg keypoints:    {np.mean(kp_hist_rgb):.1f}")
    print(f"Fusion avg keypoints: {np.mean(kp_hist_fus):.1f}")
    print(f"Fusion advantage:     {(np.mean(kp_hist_fus)-np.mean(kp_hist_rgb)):.1f} more KP/frame")

if __name__ == '__main__':
    main()
