import torch
def bbox2loc(src_bbox,dst_bbox):
    '''把框变为回归目标
    '''
    height = src_bbox[:,2]-src_bbox[:,0]
    width = src_bbox[:,3]-src_bbox[:,1]
    ctr_y = src_bbox[:,0]+0.5*height
    ctr_x = src_bbox[:,1]+0.5*width

    base_height = dst_bbox[:,2]-dst_bbox[:,0]
    base_width = dst_bbox[:,3]-dst_bbox[:,1]
    base_ctr_y = dst_bbox[:,0]+0.5*base_height
    base_ctr_x = dst_bbox[:,1]+0.5*base_width

    eps = torch.zeros_like(height)+1e-5
    height = torch.max(height,eps)
    eps = torch.zeros_like(height)+1e-5
    width = torch.max(width,eps)

    dy = (base_ctr_y-ctr_y)/height
    dx = (base_ctr_x-ctr_x)/width
    dh = torch.log(base_height/height)
    dw = torch.log(base_width/width)
    loc = torch.stack((dy,dx,dh,dw),dim=1)
    return loc