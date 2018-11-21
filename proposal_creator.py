import torch
from nms import nms
from loc2bbox import loc2bbox
class ProposalCreator(torch.nn.Module):
    '''对一个batch中的每一个样本提出proposal: 首先把rpn的回归目标变为bbox的坐标然后nms
    '''
    def __init__(self,nms_thresh=0.7,min_size=16):
        super(ProposalCreator,self).__init__()
        self.nms_thresh = nms_thresh
        self.min_size = min_size
    def forward(self,loc,score,anchor,img_size):
        roi = loc2bbox(anchor,loc)# y1x1y2x2
        #裁剪所有超出图像边缘的框
        roi[:,slice(0,4,2)].clamp_(0,img_size[0])
        roi[:,slice(1,4,2)].clamp_(0,img_size[1])
        #删除太小的框
        hs = roi[:,2]-roi[:,0]
        ws = roi[:,3]-roi[:,1]
        keep = (hs>=self.min_size) & (ws>=self.min_size)
        roi = roi[keep]#keep是uint8类型0,1为mask作用
        score = score[keep]
        #nms
        _,order = score.sort(descending=True)
        if self.training:
            n_pre_nms = 12000
            n_post_nms = 2000
        else:
            n_pre_nms = 6000
            n_post_nms = 300
        order = order[:n_pre_nms]#int64为索引作用
        roi = roi[order]
        keep = nms(roi,self.nms_thresh)
        keep = keep[:n_post_nms]
        roi = roi[keep]
        return roi
