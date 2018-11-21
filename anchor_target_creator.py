import torch
import numpy as np
from random import sample
from bbox2loc import bbox2loc
from bbox_iou import bbox_iou
class AnchorTargetCreator:
    '''
    给每个anchor分配ground truth
        Args:
        n_sample (int): The number of regions to produce.
        pos_iou_thresh (float): Anchors with IoU above this
            threshold will be assigned as positive.
        neg_iou_thresh (float): Anchors with IoU below this
            threshold will be assigned as negative.
        pos_ratio (float): Ratio of positive regions in the
            sampled regions.
    '''
    def __init__(self,n_sample=256,pos_iou_thresh=0.7,neg_iou_thresh=0.3,pos_ratio=0.5,device=torch.device('cpu')):
        self.n_sample=n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio
        self.device = device
    def __call__(self,bbox,anchor,img_size):
        img_w,img_h = img_size
        n_anchor = len(anchor)
        #删除超出图像边缘的anchor
        inside_index = torch.stack((anchor[:,0]>=0,anchor[:,1]>=0,anchor[:,2]<=img_h,anchor[:,3]<=img_w)).all(dim=0).nonzero().reshape(-1)
        anchor = anchor[inside_index]
        #给每个anchor生成类别标签
        argmax_ious, label = self._create_label(inside_index,anchor,bbox)
        #回归目标
        loc = bbox2loc(anchor,bbox[argmax_ious])
        label = _unmap(label,n_anchor,inside_index,fill=-1,device=self.device)
        loc = _unmap(loc,n_anchor,inside_index,fill=0,device=self.device)
        return loc,label
    def _create_label(self,inside_index,anchor,bbox):
        #1为正例，0为负例，-1忽略
        label = torch.full((len(inside_index),),-1,dtype=torch.long,device=self.device)
        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor,bbox,inside_index)
        label[max_ious<self.neg_iou_thresh]=0#负例
        label[gt_argmax_ious]=1#iou最高的为正例
        label[max_ious>=self.pos_iou_thresh]=1#正例

        #如果正例太多进行欠采样
        n_pos = int(self.pos_ratio*self.n_sample)
        pos_index = (label==1).nonzero().reshape(-1)
        pos_num = pos_index.shape[0]
        if pos_num>n_pos:
            disable_index = np.random.choice(
                pos_index.cpu().numpy(),size=pos_index.shape[0]-n_pos,replace=False
            )
            pos_num=n_pos
        #如果负例太多进行欠采样
        n_neg = self.n_sample-pos_num
        neg_index = (label==0).nonzero().reshape(-1)
        if neg_index.shape[0]>n_neg:
            disable_index = np.random.choice(
                neg_index.cpu().numpy(),size=neg_index.shape[0]-n_neg,replace=False
            )
            label[disable_index]=-1
        return argmax_ious,label
    def _calc_ious(self,anchor,bbox,inside_index):
        ious = bbox_iou(anchor,bbox)
        argmax_ious = ious.argmax(dim=1)#每一个anchor最近的object
        max_ious = ious[torch.arange(len(inside_index),device=self.device),argmax_ious]
        gt_argmax_ious = ious.argmax(dim=0)
        gt_max_ious = ious[gt_argmax_ious,torch.arange(ious.shape[1],device=self.device)]
        gt_argmax_ious = (ious==gt_max_ious).nonzero().reshape(-1)#可能有多个最大值
        return argmax_ious,max_ious,gt_argmax_ious
def _unmap(data,count,index,fill=0,device=torch.device('cpu')):
    if len(data.shape)==1:
        ret = torch.full((count,),fill,dtype=data.dtype,device=device)
        ret[index]=data
    else:
        ret = torch.full((count,)+data.shape[1:],fill,dtype=data.dtype,device=device)
        ret[index,:]=data
    return ret

