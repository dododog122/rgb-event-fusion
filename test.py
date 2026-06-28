import torch
import cv2
import numpy as np
from pathlib import Path
from model.model import NightFusionModel

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT     = 'checkpoints/best.pt'
IMG_SIZE = 640
NUM_CLS  = 4
CLS_NAMES= ['vehicle','pedestrian','traffic_sign','traffic_light']


def load_model():
    model = NightFusionModel(num_classes=NUM_CLS).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()
    return model


def preprocess_rgb(path):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    t   = torch.from_numpy(img.astype(np.float32)/255.0).permute(2,0,1)
    return t.unsqueeze(0).to(DEVICE), img


def preprocess_voxel(path):
    v    = np.load(str(path)).astype(np.float32)
    bins = [cv2.resize(v[b], (IMG_SIZE, IMG_SIZE)) for b in range(v.shape[0])]
    t    = torch.from_numpy(np.stack(bins))
    return t.unsqueeze(0).to(DEVICE)


def draw_boxes(img, pred, conf_thresh=0.3):
    H, W = img.shape[:2]
    out  = img.copy()
    pred = pred[0]  # remove batch dim
    C, Hg, Wg = pred.shape
    na, nc = 3, NUM_CLS
    pred = pred.view(na, 5+nc, Hg, Wg).permute(0,2,3,1)

    for a in range(na):
        for j in range(Hg):
            for i in range(Wg):
                obj = pred[a,j,i,4].sigmoid().item()
                if obj < conf_thresh: continue
                cls  = pred[a,j,i,5:].argmax().item()
                cx   = (i + 0.5) / Wg
                cy   = (j + 0.5) / Hg
                x1   = int((cx - 0.05) * W)
                y1   = int((cy - 0.05) * H)
                x2   = int((cx + 0.05) * W)
                y2   = int((cy + 0.05) * H)
                cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(out, f'{CLS_NAMES[cls]} {obj:.2f}',
                            (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,255,0), 1)
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb',   required=True)
    parser.add_argument('--voxel', required=True)
    parser.add_argument('--out',   default='result.png')
    args = parser.parse_args()

    model = load_model()
    rgb_t, rgb_img = preprocess_rgb(args.rgb)
    voxel_t        = preprocess_voxel(args.voxel)

    with torch.no_grad():
        p3, p4, sif = model(rgb_t, voxel_t)

    result = draw_boxes(rgb_img, p3.cpu())
    cv2.imwrite(args.out, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f'Result saved to {args.out}')

if __name__ == '__main__':
    main()
