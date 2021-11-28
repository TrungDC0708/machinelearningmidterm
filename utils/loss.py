import math

import torch
import torch.nn as nn

from .utils import to_cpu



def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    box2 = box2.T

    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)
        else:
            c_area = cw * ch + eps
            return iou - (c_area - union) / c_area
    else:
        return iou


def compute_loss(predictions, targets, model):
    device = targets.device

    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

    tcls, tbox, indices, anchors = build_targets(predictions, targets, model)  # targets

    BCEcls = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.0], device=device))
    BCEobj = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.0], device=device))

    for layer_index, layer_predictions in enumerate(predictions):
        b, anchor, grid_j, grid_i = indices[layer_index]
        tobj = torch.zeros_like(layer_predictions[..., 0], device=device)  # target obj
        num_targets = b.shape[0]
        if num_targets:
            ps = layer_predictions[b, anchor, grid_j, grid_i]
            pxy = ps[:, :2].sigmoid()
            pwh = torch.exp(ps[:, 2:4]) * anchors[layer_index]
            pbox = torch.cat((pxy, pwh), 1)
            iou = bbox_iou(pbox.T, tbox[layer_index], x1y1x2y2=False, CIoU=True)
            lbox += (1.0 - iou).mean()  # iou loss

            tobj[b, anchor, grid_j, grid_i] = iou.detach().clamp(0).type(tobj.dtype)


            if ps.size(1) - 5 > 1:
                t = torch.zeros_like(ps[:, 5:], device=device)
                t[range(num_targets), tcls[layer_index]] = 1
                lcls += BCEcls(ps[:, 5:], t)


        lobj += BCEobj(layer_predictions[..., 4], tobj)

    lbox *= 0.05
    lobj *= 1.0
    lcls *= 0.5

    loss = lbox + lobj + lcls

    return loss, to_cpu(torch.cat((lbox, lobj, lcls, loss)))


def build_targets(p, targets, model):
    na, nt = 3, targets.shape[0]
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(7, device=targets.device)
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

    for i, yolo_layer in enumerate(model.yolo_layers):
        anchors = yolo_layer.anchors / yolo_layer.stride
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]
        t = targets * gain

        if nt:
            r = t[:, :, 4:6] / anchors[:, None]
            j = torch.max(r, 1. / r).max(2)[0] < 4
            t = t[j]
        else:
            t = targets[0]
        b, c = t[:, :2].long().T
        gxy = t[:, 2:4]
        gwh = t[:, 4:6]
        gij = gxy.long()
        gi, gj = gij.T
        a = t[:, 6].long()
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])
        tcls.append(c)

    return tcls, tbox, indices, anch
