import torch
def bbox_iou(bbox_a,bbox_b):
    #bbox_b有可能有多个框，每个a需要与每个b计算
    tl = torch.max(bbox_a[:,:2].unsqueeze(1), bbox_b[:, :2])#左上角
    br = torch.min(bbox_a[:,2:].unsqueeze(1), bbox_b[:, 2:])#右下角
    area = torch.prod(br - tl, dim=2) * (tl < br).all(dim=2).float()#相交区域面积
    area_a = torch.prod(bbox_a[:,2:]-bbox_a[:,:2],dim=1)
    area_b = torch.prod(bbox_b[:,2:]-bbox_b[:,:2],dim=1)
    return area / (area_a[:,None]+area_b - area)
