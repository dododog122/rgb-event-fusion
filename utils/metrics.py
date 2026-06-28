import torch
import numpy as np


def box_iou(box1, box2):
    """
    box format: [cx, cy, w, h] normalized
    returns IoU scalar
    """
    def to_xyxy(b):
        x1 = b[0] - b[2]/2
        y1 = b[1] - b[3]/2
        x2 = b[0] + b[2]/2
        y2 = b[1] + b[3]/2
        return x1, y1, x2, y2

    b1x1,b1y1,b1x2,b1y2 = to_xyxy(box1)
    b2x1,b2y1,b2x2,b2y2 = to_xyxy(box2)

    ix1 = max(b1x1, b2x1)
    iy1 = max(b1y1, b2y1)
    ix2 = min(b1x2, b2x2)
    iy2 = min(b1y2, b2y2)

    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    area1 = (b1x2-b1x1) * (b1y2-b1y1)
    area2 = (b2x2-b2x1) * (b2y2-b2y1)
    union = area1 + area2 - inter + 1e-6
    return inter / union


def compute_map(all_preds, all_targets,
                num_classes=4, iou_thresh=0.5):
    """
    all_preds:   list of [N, 6] tensors (cx,cy,w,h,conf,cls)
    all_targets: list of [M, 5] tensors (cls,cx,cy,w,h)
    returns: mAP@iou_thresh
    """
    APs = []

    for cls in range(num_classes):
        tp_list, conf_list = [], []
        n_gt = 0

        for preds, targets in zip(all_preds, all_targets):
            # filter by class
            gt = targets[targets[:,0] == cls][:,1:]  # (M,4)
            pr = preds[preds[:,5] == cls]            # (N,6)
            n_gt += len(gt)

            if len(pr) == 0:
                continue

            # sort by confidence
            pr = pr[pr[:,4].argsort(descending=True)]
            matched = torch.zeros(len(gt))

            for p in pr:
                conf = p[4].item()
                conf_list.append(conf)

                if len(gt) == 0:
                    tp_list.append(0)
                    continue

                ious = torch.tensor([
                    box_iou(p[:4].tolist(), g.tolist()) for g in gt
                ])
                best_iou, best_idx = ious.max(0)

                if best_iou >= iou_thresh and not matched[best_idx]:
                    tp_list.append(1)
                    matched[best_idx] = 1
                else:
                    tp_list.append(0)

        if n_gt == 0 or len(tp_list) == 0:
            APs.append(0.0)
            continue

        # compute precision-recall
        tp   = np.array(tp_list)
        conf = np.array(conf_list)
        idx  = np.argsort(-conf)
        tp   = tp[idx]

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(1 - tp)
        precision = cum_tp / (cum_tp + cum_fp + 1e-6)
        recall    = cum_tp / (n_gt + 1e-6)

        # AP = area under PR curve
        ap = np.trapz(precision, recall)
        APs.append(ap)

    mAP = np.mean(APs)
    return mAP, APs
