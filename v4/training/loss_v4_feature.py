import torch
import torch.nn as nn
import torch.nn.functional as F

NC = 4

def compute_loss(pred_tuple, gt_boxes_list, gt_cls_list, device='cuda'):
    """
    pred_tuple[0]: (B, 84, 8400)
      84 = 4(box) + 80(cls) for COCO
      但我們 NC=4，所以是 4+4=8? 不對
      YOLOv8n 預設輸出 84 = 4 + 80
      我們需要看實際的格式
    """
    # 取第一個元素
    if isinstance(pred_tuple, (list, tuple)):
        pred = pred_tuple[0]  # (B, 84, 8400)
    else:
        pred = pred_tuple

    B, CH, N = pred.shape
    # CH = 4 (box) + num_classes
    # 8400 = 80*80 + 40*40 + 20*20 = 6400+1600+400

    pred_box = pred[:, :4, :]        # (B, 4, 8400)
    pred_cls = pred[:, 4:4+NC, :]   # (B, NC, 8400)

    # flatten to (B, 8400, ...)
    pred_box = pred_box.permute(0, 2, 1)  # (B, 8400, 4)
    pred_cls = pred_cls.permute(0, 2, 1)  # (B, 8400, NC)

    # anchor points (normalized 0~1)
    # 8400 = 6400+1600+400
    # stride 8: 80x80=6400, stride 16: 40x40=1600, stride 32: 20x20=400
    strides = [8, 16, 32]
    sizes   = [80, 40, 20]
    anchors = []
    for s, sz in zip(strides, sizes):
        grid_y, grid_x = torch.meshgrid(
            torch.arange(sz), torch.arange(sz), indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
        grid = (grid + 0.5) / (sz)  # normalize to 0~1
        anchors.append(grid)
    anchors = torch.cat(anchors, dim=0).to(device)  # (8400, 2)

    total = torch.tensor(0., device=device, requires_grad=True)
    n_pos_total = 0

    for b in range(B):
        gt_b = gt_boxes_list[b].to(device)  # (N, 4) xywh normalized
        cl_b = gt_cls_list[b].to(device)    # (N,)

        obj_tgt = torch.zeros(8400, device=device)
        box_loss = torch.tensor(0., device=device)
        cls_loss = torch.tensor(0., device=device)

        if gt_b.shape[0] == 0:
            obj_loss = F.binary_cross_entropy_with_logits(
                pred_cls[b].max(-1).values, obj_tgt)
            total = total + obj_loss
            continue

        # assign each GT to nearest anchor point
        gt_xy = gt_b[:, :2]  # (M, 2)
        # dist from each anchor to each GT center
        dist = torch.cdist(
            anchors.unsqueeze(0),
            gt_xy.unsqueeze(0)
        ).squeeze(0)  # (8400, M)

        # for each GT, find top-k nearest anchors
        topk = min(10, 8400)
        _, topk_idx = dist.topk(topk, dim=0, largest=False)  # (topk, M)

        for n in range(gt_b.shape[0]):
            pos_idx = topk_idx[:, n]  # (topk,)
            obj_tgt[pos_idx] = 1.

            # box loss on positive anchors
            pb = pred_box[b, pos_idx]  # (topk, 4)
            gb = gt_b[n].unsqueeze(0).expand(topk, -1)
            box_loss = box_loss + F.mse_loss(pb.sigmoid(), gb)

            # cls loss
            ct = torch.zeros(NC, device=device)
            ct[cl_b[n]] = 1.
            cls_loss = cls_loss + F.binary_cross_entropy_with_logits(
                pred_cls[b, pos_idx].mean(0), ct)
            n_pos_total += topk

        obj_loss = F.binary_cross_entropy_with_logits(
            pred_cls[b].max(-1).values, obj_tgt)

        if n_pos_total > 0:
            total = total + obj_loss + box_loss + cls_loss
        else:
            total = total + obj_loss

    return total / B
