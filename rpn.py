import torch
import torch.nn as  nn
import torch.nn.functional as F
from torchvision.models import vgg16

from proposal_creator import ProposalCreator
from generate_anchor_base import generate_anchor_base
class RPN(nn.Module):
    '''
    根据从图像中提取的特征，生成类别无关的回归框
    '''
    def __init__(self,feat_stride=16,ratios=(0.5,1,2),scales=(8,16,32),device=torch.device('cpu')):
        super(RPN,self).__init__()
        self.device = device
        self.feat_stride = feat_stride#feature map上的一个像素对应原图的多少像素
        anchor_base = generate_anchor_base(ratios=ratios,anchor_scales=scales)
        self.anchor_base = torch.tensor(anchor_base,dtype=torch.float)
        n_anchors = self.anchor_base.shape[0]

        self.feature_extractor = vgg16(pretrained=True).features[:30]
        # for l in self.feature_extractor:
        #     if isinstance(l,nn.MaxPool2d):
        #         l.ceil_mode=True

        for p in self.feature_extractor[:5].parameters():
            p.require_grad=False

        self.conv = nn.Conv2d(512,512,3,padding=1)
        self.score = nn.Conv2d(512,2*n_anchors,1)
        self.loc = nn.Conv2d(512,4*n_anchors,1)

        self.proposal_layer = ProposalCreator()
        self.to(device)
    def _enumerate_shifted_anchor(self,height,width):
        shift_y = torch.arange(0,height*self.feat_stride,self.feat_stride,dtype=torch.float)
        shift_x = torch.arange(0,width*self.feat_stride,self.feat_stride,dtype=torch.float)
        shift_x,shift_y = torch.meshgrid([shift_x,shift_y])#这个函数和numpy的不一样啊卧槽
        shift_x,shift_y = shift_x.t(),shift_y.t()
        shift = torch.stack((shift_y.reshape(-1),shift_x.reshape(-1),shift_y.reshape(-1),shift_x.reshape(-1)),dim=1)
        A = self.anchor_base.shape[0]
        K = shift.shape[0]
        anchor = self.anchor_base.reshape((1,A,4))+shift.reshape((K,1,4))
        anchor = anchor.reshape((K*A,4))
        return anchor
    def forward(self,x):
        img_size = x.shape[2:]
        feature = self.feature_extractor(x)
        n,_,hh,ww = feature.shape
        x = F.relu(self.conv(feature))
        
        rpn_locs = self.loc(x)
        rpn_locs = rpn_locs.permute(0,2,3,1).reshape(n,-1,4)

        rpn_scores = self.score(x)
        rpn_scores = rpn_scores.permute(0,2,3,1).reshape(n,-1,2)
        with torch.no_grad():
            rpn_fg_scores = rpn_scores[:,:,1]
            anchor = self._enumerate_shifted_anchor(hh,ww).to(self.device)
            rois = []
            roi_indices = []
            for i in range(n):
                roi = self.proposal_layer(rpn_locs[i],rpn_fg_scores[i],anchor,img_size)
                rois.append(roi)
                batch_index = i*torch.ones(roi.shape[0])
                roi_indices.append(batch_index)
            rois = torch.cat(rois)
            roi_indices = torch.cat(roi_indices)
        return feature,rpn_locs,rpn_scores,rois,roi_indices,anchor