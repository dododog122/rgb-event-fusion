import torch
import torch.nn as nn


def rsgnet_loss(sif, voxel):
    target = voxel.abs().mean(dim=1, keepdim=True).expand_as(sif)
    mn, mx = target.min(), target.max()
    target = (target - mn) / (mx - mn + 1e-6)
    return nn.MSELoss()(sif, target)


def detection_loss(pred, targets, num_anchors=3, num_classes=4):
    B, C, H, W = pred.shape
    pred = pred.view(B, num_anchors, 5+num_classes, H, W).permute(0,1,3,4,2)
    obj_pred   = pred[..., 4].sigmoid()
    obj_target = torch.zeros_like(obj_pred)

    for b_idx, boxes in enumerate(targets):
        for box in boxes:
            gi = min(int(box[1].item() * W), W-1)
            gj = min(int(box[2].item() * H), H-1)
            obj_target[b_idx, :, gj, gi] = 1.0

    return nn.BCELoss()(obj_pred, obj_target)
