import sys
sys.path.insert(0, '/home/ivlab/carla_ws/fusion_model')

import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from v4.training.train_v4_feature import V4FeatureModel, V4Dataset
from torch.utils.data import DataLoader

DEVICE    = 'cuda'
BASE      = '/home/ivlab/carla_ws/fusion_model'
CKPT_DIR  = Path(f'{BASE}/checkpoints')
NC        = 4
CONF      = 0.1
IOU_THR   = 0.5
IMG_SIZE  = 640
CLS_NAMES = ['vehicle','pedestrian','traffic_sign','traffic_light']


def xywh2xyxy(b):
    x,y,w,h = b[:,0],b[:,1],b[:,2],b[:,3]
    return torch.stack([x-w/2,y-h/2,x+w/2,y+h/2],dim=1)

def box_iou(b1,b2):
    a1=(b1[:,2]-b1[:,0])*(b1[:,3]-b1[:,1])
    a2=(b2[:,2]-b2[:,0])*(b2[:,3]-b2[:,1])
    ix1=torch.max(b1[:,None,0],b2[None,:,0])
    iy1=torch.max(b1[:,None,1],b2[None,:,1])
    ix2=torch.min(b1[:,None,2],b2[None,:,2])
    iy2=torch.min(b1[:,None,3],b2[None,:,3])
    inter=(ix2-ix1).clamp(0)*(iy2-iy1).clamp(0)
    return inter/(a1[:,None]+a2[None,:]-inter+1e-6)

def compute_ap(rec,prec):
    rec =np.concatenate([[0.],rec,[1.]])
    prec=np.concatenate([[0.],prec,[0.]])
    for i in range(len(prec)-2,-1,-1):
        prec[i]=max(prec[i],prec[i+1])
    idx=np.where(rec[1:]!=rec[:-1])[0]
    return np.sum((rec[idx+1]-rec[idx])*prec[idx+1])

def decode_pred(pred_out, device):
    """
    eval mode: pred_out = (tensor(B,84,8400), dict)
    returns: scores (B,8400,NC), boxes_xywh (B,8400,4)
    """
    p = pred_out[0]                    # (B,84,8400)
    pred_box = p[:,:4,:].permute(0,2,1) / 640.0    # (B,8400,4) normalize to 0~1
    pred_cls = p[:,4:4+NC,:].permute(0,2,1)         # (B,8400,NC)
    return pred_cls, pred_box

def evaluate():
    print('\n=== V4 Feature-level mAP Evaluation ===')

    model = V4FeatureModel().to(DEVICE)
    model.load_state_dict(
        torch.load(CKPT_DIR/'v4_feature_best.pt', map_location=DEVICE))
    model.eval()
    print('Loaded: v4_feature_best.pt')

    def collate_fn(batch):
        r,v,b,c=zip(*batch)
        return torch.stack(r),torch.stack(v),list(b),list(c)

    val_ds = V4Dataset(f'{BASE}/night/rgb',
                       f'{BASE}/night/voxel',
                       f'{BASE}/night/yolo/labels')
    val_ld = DataLoader(val_ds, batch_size=4, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)

    # anchor centres
    anchors = []
    for sz in [80,40,20]:
        gy,gx=torch.meshgrid(
            torch.arange(sz,dtype=torch.float32,device=DEVICE),
            torch.arange(sz,dtype=torch.float32,device=DEVICE),
            indexing='ij')
        grid=torch.stack([(gx+0.5)/sz,(gy+0.5)/sz],dim=-1)
        anchors.append(grid.reshape(-1,2))
    anchors=torch.cat(anchors)  # (8400,2)

    all_preds=[[] for _ in range(NC)]
    all_ngt  =[0]*NC

    with torch.no_grad():
        for rgb,vox,boxes_list,cls_list in tqdm(val_ld,desc='Eval'):
            rgb=rgb.to(DEVICE); vox=vox.to(DEVICE)
            pred_out=model(rgb,vox)
            scores,pred_box=decode_pred(pred_out,DEVICE)
            # scores: (B,8400,NC)  pred_box: (B,8400,4)

            B=rgb.shape[0]
            for b in range(B):
                gt_b=boxes_list[b].to(DEVICE)
                cl_b=cls_list[b].to(DEVICE)

                for c in range(NC):
                    all_ngt[c]+=(cl_b==c).sum().item()

                conf,cls_pred=scores[b].max(dim=-1)  # (8400,)
                keep=conf>CONF
                if keep.sum()==0: continue

                conf_k    =conf[keep]
                cls_k     =cls_pred[keep]
                anc_k     =anchors[keep]
                box_k     =pred_box[b][keep]  # (K,4)

                # combine anchor centre with predicted wh
                pred_xywh=torch.cat([anc_k, box_k[:,2:4]],dim=1)
                pred_xyxy=xywh2xyxy(pred_xywh)

                if gt_b.shape[0]==0: continue
                gt_xyxy=xywh2xyxy(gt_b)

                iou=box_iou(pred_xyxy,gt_xyxy)  # (K,M)
                gt_matched=torch.zeros(
                    gt_b.shape[0],dtype=torch.bool,device=DEVICE)

                order=conf_k.argsort(descending=True)
                for idx in order:
                    c=cls_k[idx].item()
                    sc=conf_k[idx].item()
                    gt_mask=(cl_b==c)
                    if gt_mask.sum()==0:
                        all_preds[c].append((sc,0))
                        continue
                    ious_c=iou[idx][gt_mask]
                    gt_ids=torch.where(gt_mask)[0]
                    best_iou,best_j=ious_c.max(dim=0)
                    best_gt=gt_ids[best_j]
                    if best_iou>=IOU_THR and not gt_matched[best_gt]:
                        gt_matched[best_gt]=True
                        all_preds[c].append((sc,1))
                    else:
                        all_preds[c].append((sc,0))

    print('\n=== Results ===')
    aps=[]
    for c in range(NC):
        ngt=all_ngt[c]
        if ngt==0 or len(all_preds[c])==0:
            print(f'  {CLS_NAMES[c]:15s}: AP=0.000  (GT={ngt})')
            aps.append(0.); continue
        preds_c=sorted(all_preds[c],key=lambda x:-x[0])
        tp=np.array([p[1] for p in preds_c])
        fp=1-tp
        tp_c=np.cumsum(tp); fp_c=np.cumsum(fp)
        rec =tp_c/(ngt+1e-6)
        prec=tp_c/(tp_c+fp_c+1e-6)
        ap=compute_ap(rec,prec)
        aps.append(ap)
        print(f'  {CLS_NAMES[c]:15s}: AP@50={ap:.3f}  '
              f'(GT={ngt}, pred={len(preds_c)})')

    mAP=np.mean(aps)
    print(f'\n  mAP@50 = {mAP:.3f}')
    print(f'\n  Comparison:')
    print(f'    RGB-only  : 0.462')
    print(f'    Fusion V3 : 0.399')
    print(f'    Fusion V4 : {mAP:.3f}  ← true feature-level')

if __name__=='__main__':
    evaluate()
