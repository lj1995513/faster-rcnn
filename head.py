import torch
import torch.nn as nn
import torch.nn.functional as F
from roi_pooling import ROIPooling
class FasterRCNNHead(nn.Module):
    def __init__(self,n_class,roi_size,spatial_scale,device=torch.device('cpu')):
        super(FasterRCNNHead,self).__init__()
        self.fc6 = nn.Linear(25088,4096)
        self.fc7 = nn.Linear(4096,4096)
        self.cls_loc = nn.Linear(4096,n_class*4)
        self.score = nn.Linear(4096,n_class)
        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi_pooling = ROIPooling(roi_size,roi_size,spatial_scale,device)
        self.to(device)
    def forward(self,feature,rois,roi_indices):
        #TODO: ROI-POOLING
        rois = torch.cat((roi_indices,rois),dim=1)
        pool = self.roi_pooling(feature,rois)
        pool = pool.view(-1,512*7*7)
        fc6 = F.relu(self.fc6(pool))
        fc7 = F.relu(self.fc7(fc6))
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs,roi_scores