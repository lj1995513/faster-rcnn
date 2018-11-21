import torch
def loc2bbox(src_bbox,loc):
    '''把回归目标变为真实坐标
    Args:
    src_bbox (array): Anchor
        ymin, xmin, ymax, xmax
    loc (array): RPN的回归目标
        t_y, t_x, t_h, t_w
    '''
    #anchor bbox转为宽高中心点
    src_height = src_bbox[:,2]-src_bbox[:,0]
    src_width = src_bbox[:,3]-src_bbox[:,1]
    src_ctr_y = src_bbox[:,0]+0.5*src_height
    src_ctr_x = src_bbox[:,1]+0.5*src_width
    #rpn输出的正则化的宽高中心点
    dy = loc[:,0::4]
    dx = loc[:,1::4]
    dh = loc[:,2::4]
    dw = loc[:,3::4]
    #rpn预测的bbox宽高中心点
    ctr_y = dy * src_height.unsqueeze(-1) + src_ctr_y.unsqueeze(-1)
    ctr_x = dx * src_width.unsqueeze(-1) + src_ctr_x.unsqueeze(-1)
    h = torch.exp(dh) * src_height.unsqueeze(-1)
    w = torch.exp(dw) * src_width.unsqueeze(-1)
    #宽高中心点改为左上角右下角
    dst_bbox = torch.zeros_like(loc)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w
    return dst_bbox