import sys
sys.path.insert(0, '/home/ivlab/carla_ws/fusion_model')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from ultralytics import YOLO

DEVICE   = 'cuda'
BASE     = '/media/ivlab/8AC24E8FC24E7F85/fusion_data'
CKPT_DIR = Path('/home/ivlab/carla_ws/fusion_model/checkpoints')
IMG_SIZE = 640
NC       = 4
H, W     = 720, 1280


def events_to_membrane(events, H=H, W=W, decay=0.9, n_bins=10):
    V_pos = np.zeros((H,W), dtype=np.float32)
    V_neg = np.zeros((H,W), dtype=np.float32)
    if len(events)==0: return np.stack([V_pos,V_neg])
    idx=np.argsort(events['t']); evts=events[idx]
    xs=np.clip(evts['x'].astype(np.int32),0,W-1)
    ys=np.clip(evts['y'].astype(np.int32),0,H-1)
    pols=evts['pol'].astype(np.bool_)
    for bi in np.array_split(np.arange(len(evts)),n_bins):
        if len(bi)==0: continue
        V_pos*=decay; V_neg*=decay
        bx=xs[bi]; by=ys[bi]; bp=pols[bi]
        np.add.at(V_pos,(by[bp], bx[bp]),  1.0)
        np.add.at(V_neg,(by[~bp],bx[~bp]), 1.0)
    pm=V_pos.max(); nm=V_neg.max()
    if pm>0: V_pos/=pm
    if nm>0: V_neg/=nm
    return np.stack([V_pos,V_neg])


class MembraneEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Sequential(
            nn.Conv2d(2,16,3,stride=2,padding=1),nn.BatchNorm2d(16),nn.SiLU(),
            nn.Conv2d(16,32,3,stride=2,padding=1),nn.BatchNorm2d(32),nn.SiLU(),
            nn.Conv2d(32,64,3,stride=2,padding=1),nn.BatchNorm2d(64),nn.SiLU())
        self.p3=nn.Sequential(
            nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64),nn.SiLU(),
            nn.Conv2d(64,64,1),nn.BatchNorm2d(64),nn.SiLU())
        self.p4=nn.Sequential(
            nn.Conv2d(64,128,3,stride=2,padding=1),nn.BatchNorm2d(128),nn.SiLU(),
            nn.Conv2d(128,128,1),nn.BatchNorm2d(128),nn.SiLU())
    def forward(self,x):
        f=self.stem(x)
        return self.p3(f),self.p4(f)


class MembraneAdaptiveFusion(nn.Module):
    def __init__(self,ch):
        super().__init__()
        self.ca=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),nn.Flatten(),
            nn.Linear(ch*2,ch//2),nn.GELU(),
            nn.Linear(ch//2,ch*2),nn.Sigmoid())
        self.illum=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),nn.Flatten(),
            nn.Linear(ch,1),nn.Sigmoid())
        self.norm=nn.BatchNorm2d(ch)
        self.proj=nn.Conv2d(ch*2,ch,1)

    def forward(self,rgb_f,mem_f):
        illum=self.illum(rgb_f)
        mem_w=(1.0-illum).view(-1,1,1,1)
        mem_f=mem_f*mem_w
        combined=torch.cat([rgb_f,mem_f],dim=1)
        ca_w=self.ca(combined).view(combined.shape[0],combined.shape[1],1,1)
        fused=self.proj(combined*ca_w)
        return self.norm(fused+rgb_f)


class MembraneV5(nn.Module):
    def __init__(self):
        super().__init__()
        self.mem_enc=MembraneEncoder()
        self.fuse_p3=MembraneAdaptiveFusion(64)
        self.fuse_p4=MembraneAdaptiveFusion(128)
        m=YOLO('yolov8n.pt').model.model
        self.l0=m[0];self.l1=m[1];self.l2=m[2];self.l3=m[3]
        self.l4=m[4];self.l5=m[5];self.l6=m[6];self.l7=m[7]
        self.l8=m[8];self.l9=m[9];self.l10=m[10];self.l11=m[11]
        self.l12=m[12];self.l13=m[13];self.l14=m[14];self.l15=m[15]
        self.l16=m[16];self.l17=m[17];self.l18=m[18];self.l19=m[19]
        self.l20=m[20];self.l21=m[21];self.l22=m[22]

    def forward(self,rgb,membrane):
        mp3,mp4=self.mem_enc(membrane)
        x0=self.l0(rgb);x1=self.l1(x0);x2=self.l2(x1)
        x3=self.l3(x2);x4=self.l4(x3)
        x5=self.l5(x4);x6=self.l6(x5)
        x4f=self.fuse_p3(x4,mp3)
        x6f=self.fuse_p4(x6,mp4)
        x7=self.l7(x6);x8=self.l8(x7);x9=self.l9(x8)
        x10=self.l10(x9)
        x11=self.l11([x10,x6f])
        x12=self.l12(x11);x13=self.l13(x12)
        x14=self.l14([x13,x4f])
        x15=self.l15(x14);x16=self.l16(x15)
        x17=self.l17([x16,x12])
        x18=self.l18(x17);x19=self.l19(x18)
        x20=self.l20([x19,x9])
        x21=self.l21(x20)
        return self.l22([x15,x18,x21])


class MembraneDataset(Dataset):
    def __init__(self,rgb_dir,dvs_dir,label_dir,split='train'):
        self.dvs_dir=Path(dvs_dir)
        self.lbl_dir=Path(label_dir)
        self.files=sorted([
            f for f in Path(rgb_dir).glob('frame_*.png')
            if (self.dvs_dir/f'{f.stem}.npy').exists()
            and (self.lbl_dir/f'{f.stem}.txt').exists()])
        print(f'  [{split}] {len(self.files)} frames')

    def __len__(self): return len(self.files)

    def __getitem__(self,idx):
        stem=self.files[idx].stem; S=IMG_SIZE
        rgb=cv2.imread(str(self.files[idx]))
        rgb=cv2.resize(rgb,(S,S))
        rgb=torch.from_numpy(rgb).permute(2,0,1).float()/255.
        events=np.load(str(self.dvs_dir/f'{stem}.npy'))
        mp=events_to_membrane(events)
        membrane=torch.from_numpy(np.stack([
            cv2.resize(mp[0],(S,S)),
            cv2.resize(mp[1],(S,S))]))
        boxes,cls_ids=[],[]
        with open(self.lbl_dir/f'{stem}.txt') as f:
            for line in f:
                p=line.strip().split()
                if len(p)==5:
                    c,x,y,w,h=map(float,p)
                    cls_ids.append(int(c)); boxes.append([x,y,w,h])
        boxes=torch.tensor(boxes,dtype=torch.float32) \
              if boxes else torch.zeros((0,4))
        cls_ids=torch.tensor(cls_ids,dtype=torch.long) \
                if cls_ids else torch.zeros((0,),dtype=torch.long)
        return rgb,membrane,boxes,cls_ids

def collate_fn(batch):
    r,m,b,c=zip(*batch)
    return torch.stack(r),torch.stack(m),list(b),list(c)


def compute_loss(pred_out,gt_boxes_list,gt_cls_list,device='cuda'):
    if isinstance(pred_out,dict):
        pred_cls=pred_out['scores'].permute(0,2,1)
        pred_box=pred_out['boxes'].permute(0,2,1)
    elif isinstance(pred_out,(list,tuple)):
        p=pred_out[0]
        pred_box=p[:,:4,:].permute(0,2,1)
        pred_cls=p[:,4:,:].permute(0,2,1)
    else:
        return torch.tensor(0.,device=device,requires_grad=True)
    B=pred_cls.shape[0]; N=pred_cls.shape[1]
    pred_cls_nc=pred_cls[:,:,:NC]
    anchors=[]
    for sz in [80,40,20]:
        gy,gx=torch.meshgrid(
            torch.arange(sz,dtype=torch.float32,device=device),
            torch.arange(sz,dtype=torch.float32,device=device),
            indexing='ij')
        anchors.append(torch.stack(
            [(gx+0.5)/sz,(gy+0.5)/sz],dim=-1).reshape(-1,2))
    anchors=torch.cat(anchors)
    all_losses=[]
    for b in range(B):
        gt_b=gt_boxes_list[b].to(device)
        cl_b=gt_cls_list[b].to(device)
        obj_tgt=torch.zeros(N,device=device)
        obj_l=F.binary_cross_entropy_with_logits(
            pred_cls_nc[b].max(-1).values,obj_tgt)
        if gt_b.shape[0]==0:
            all_losses.append(obj_l); continue
        dist=torch.cdist(anchors,gt_b[:,:2])
        _,topk=dist.topk(10,dim=0,largest=False)
        pos_idx=topk.reshape(-1).unique()
        obj_tgt[pos_idx]=1.
        obj_l2=F.binary_cross_entropy_with_logits(
            pred_cls_nc[b].max(-1).values,obj_tgt)
        box_tgt=torch.zeros(len(pos_idx),4,device=device)
        cls_tgt=torch.zeros(len(pos_idx),NC,device=device)
        for n in range(gt_b.shape[0]):
            mask=torch.isin(pos_idx,topk[:,n])
            box_tgt[mask]=gt_b[n]
            cls_tgt[mask,cl_b[n]]=1.
        box_l=F.mse_loss(pred_box[b][pos_idx][:,:4].sigmoid(),box_tgt)
        cls_l=F.binary_cross_entropy_with_logits(
            pred_cls_nc[b][pos_idx],cls_tgt)
        all_losses.append(obj_l2+box_l+cls_l)
    return torch.stack(all_losses).mean()


def train(epochs=50):
    print('\n=== Membrane Potential Feature-level Fusion ===')
    print('V_pos + V_neg directly injected into P3/P4')

    train_ds=MembraneDataset(
        f'{BASE}/morning/rgb',
        f'{BASE}/morning/dvs',
        f'{BASE}/morning/yolo/labels',
        split='train')
    val_ds=MembraneDataset(
        f'{BASE}/night/rgb',
        f'{BASE}/night/dvs',
        f'{BASE}/night/yolo/labels',
        split='val')

    train_ld=DataLoader(train_ds,batch_size=2,shuffle=True,
                        num_workers=0,collate_fn=collate_fn)
    val_ld=DataLoader(val_ds,batch_size=2,shuffle=False,
                      num_workers=0,collate_fn=collate_fn)

    model=MembraneV5().to(DEVICE)
    print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')

    opt=AdamW([
        {'params':model.mem_enc.parameters(), 'lr':1e-4},
        {'params':model.fuse_p3.parameters(), 'lr':1e-4},
        {'params':model.fuse_p4.parameters(), 'lr':1e-4},
        {'params':[p for n,p in model.named_parameters()
                   if n.startswith('l')],     'lr':5e-5},
    ],weight_decay=0.01)
    sch=CosineAnnealingLR(opt,T_max=epochs)
    best=float('inf')

    for ep in range(1,epochs+1):
        model.train(); tr=0
        for rgb,mem,boxes,cls_ids in tqdm(
                train_ld,desc=f'Ep{ep:02d}/{epochs}'):
            rgb=rgb.to(DEVICE); mem=mem.to(DEVICE)
            opt.zero_grad()
            pred=model(rgb,mem)
            loss=compute_loss(pred,boxes,cls_ids,DEVICE)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            tr+=loss.item()

        model.eval(); vl=0
        with torch.no_grad():
            for rgb,mem,boxes,cls_ids in val_ld:
                rgb=rgb.to(DEVICE); mem=mem.to(DEVICE)
                pred=model(rgb,mem)
                loss=compute_loss(pred,boxes,cls_ids,DEVICE)
                vl+=loss.item()

        sch.step()
        tl=tr/len(train_ld); vl=vl/len(val_ld)
        print(f'Ep {ep:02d} | train={tl:.4f}  val={vl:.4f}')
        if vl<best:
            best=vl
            torch.save(model.state_dict(),
                       CKPT_DIR/'membrane_v5_best.pt')
            print(f'  Saved! val={vl:.4f}')

    print(f'\nDone! Best val={best:.4f}')

if __name__=='__main__':
    train(epochs=50)
