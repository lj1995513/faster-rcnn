import torch
from bbox_iou import bbox_iou
def nms(bbox, thresh, score=None, limit=None):
    '''bbox(y1x1y2x2)按score从高往低的顺序排列，后面的框如果和前面的框重合率大于阈值就丢弃
    '''
    if len(bbox) == 0:
        return torch.zeros((0,), dtype=torch.int32)

    if score is not None:
        order = score.argsort()[::-1]
        bbox = bbox[order]
    bbox_area = torch.prod(bbox[:, 2:]-bbox[:, :2], dim=1)#所有框的面积

    selec = torch.zeros(bbox.shape[0], dtype=torch.uint8)
    selec[0]=1
    for i, b in enumerate(bbox):
        tl = torch.max(b[:2].unsqueeze(0), bbox[selec, :2])#框b和所有框相交区域的左上角
        br = torch.min(b[2:].unsqueeze(0), bbox[selec, 2:])#框b和所有框相交区域的右下角
        area = torch.prod(br - tl, dim=1) * (tl < br).all(dim=1).float()#相交区域面积

        iou = area / (bbox_area[i] + bbox_area[selec] - area)
        if (iou >= thresh).any():
            continue

        selec[i] = 1
        if limit is not None and len(selec.nonzero()) >= limit:
            break

    selec = selec.nonzero().reshape(-1)
    if score is not None:
        selec = order[selec]
    return selec.long()