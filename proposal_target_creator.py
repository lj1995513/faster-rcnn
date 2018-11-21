import numpy as np
import torch
from bbox_iou import bbox_iou
from bbox2loc import bbox2loc
class ProposalTargetCreator:
    '''给rpn的proposal 分配ground truth,采样n_sample个roi
    '''
    def __init__(self,n_sample=128,pos_ratio=0.25,pos_iou_thresh=0.5,neg_iou_thresh_hi=0.5,neg_iou_thresh_lo=0.0,loc_normalize_mean=(0.,0.,0.,0.),loc_normalize_std=(0.1,0.1,0.2,0.2),device=torch.device('cpu')):
        self.n_sample=n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh=pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo
        self.loc_normalize_mean=torch.tensor(loc_normalize_mean,device=device)
        self.loc_normalize_std=torch.tensor(loc_normalize_std,device=device)
    def __call__(self,roi,bbox,label):
        iou = bbox_iou(roi,bbox)
        # gt_assignment = iou.argmax(dim=1)
        max_iou,gt_assignment = iou.max(dim=1)
        gt_roi_label = label[gt_assignment] + 1#把类别全部加一，背景由-1变为0

        #选择前景ROI
        pos_index = (max_iou>=self.pos_iou_thresh).nonzero().reshape(-1)
        pos_roi_per_this_image = self.pos_ratio*self.n_sample
        pos_roi_per_this_image = int(min(pos_roi_per_this_image,pos_index.shape[0]))
        if pos_index.shape[0]>0:
            pos_index = np.random.choice(pos_index.cpu().numpy(),size=pos_roi_per_this_image,replace=False)

        #选择背景ROI
        neg_index = ((max_iou<self.neg_iou_thresh_hi) & (max_iou>=self.neg_iou_thresh_lo)).nonzero().reshape(-1)
        neg_roi_per_this_image = self.n_sample-pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image,neg_index.shape[0]))
        if neg_index.shape[0]>0:
            neg_index=np.random.choice(neg_index.cpu().numpy(),size = neg_roi_per_this_image,replace=False)
        
        keep_index = np.concatenate([pos_index,neg_index])
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:]=0#负例为0
        sample_roi = roi[keep_index]

        gt_roi_loc = bbox2loc(sample_roi,bbox[gt_assignment[keep_index]])
        gt_roi_loc = ((gt_roi_loc-self.loc_normalize_mean)/self.loc_normalize_std)
        return sample_roi,gt_roi_loc,gt_roi_label

