from rpn import RPN
from head import FasterRCNNHead
from anchor_target_creator import AnchorTargetCreator
from proposal_target_creator import ProposalTargetCreator
from dataset import VOC07,Rescale,RandomCrop,ToTensor,Normlize
from smooth_l1_loss import SmoothL1Loss

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
from torchvision import transforms

batch_size=1
epochs = 10
lr = 0.001
momentum=0.9
img_shape = (800,600)
ratios=(0.5,1,2)
scales=(8,16,32)
rpn_sigma=3
roi_sigma=1
feat_stride = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data = VOC07('VOCdevkit/VOC2007',split='train',transform=transforms.Compose([
    Rescale(801),
    RandomCrop(img_shape),
    ToTensor(),
    Normlize()
]))
data_loader = DataLoader(train_data,batch_size=batch_size,pin_memory=True)

rpn = RPN(feat_stride=feat_stride,ratios=ratios,scales=scales,device=device)
head = FasterRCNNHead(train_data.class_num+1,roi_size=7,spatial_scale=1/feat_stride,device=device)
anchor_target_creator = AnchorTargetCreator(device=device)
proposal_target_creator = ProposalTargetCreator(device=device)

rpn_cls_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
rpn_loc_loss = SmoothL1Loss(device)
roi_cls_loss = torch.nn.CrossEntropyLoss()
roi_loc_loss = SmoothL1Loss(device)
# param = [rpn.parameters(),head.parameters()]
# optim = SGD(rpn.parameters(),lr=lr,momentum=momentum)
optim = SGD([{'params':rpn.parameters(),'params':head.parameters()}],lr=lr,momentum=momentum)
for epoch in range(epochs):
    for i,data in enumerate(data_loader):
        optim.zero_grad()
        imgs,boxs,labels = data['image'],data['box'],data['label']
        box = boxs[0]
        label = labels[0]
        imgs = imgs.to(device)
        box = box.to(device)
        label = label.to(device)

        feature,rpn_locs,rpn_scores,rois,roi_indices,anchor = rpn(imgs)
        gt_rpn_loc,gt_rpn_label = anchor_target_creator(box,anchor,img_shape)
        rpn_closs = rpn_cls_loss(rpn_scores[0],gt_rpn_label)
        rpn_lloss = rpn_loc_loss(rpn_locs[0],gt_rpn_loc,gt_rpn_label,rpn_sigma)

        sample_roi,gt_roi_loc,gt_roi_label = proposal_target_creator(rois,box,label)
        sample_roi_index = torch.zeros((sample_roi.shape[0],1),dtype=torch.float).to(device)

        roi_cls_loc,roi_score = head(feature,sample_roi,sample_roi_index)
        roi_closs = roi_cls_loss(roi_score,gt_roi_label)
        # roi_closs.backward()
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.reshape((n_sample,-1,4))
        roi_loc = roi_cls_loc[torch.arange(n_sample),gt_roi_label]
        roi_lloss = roi_loc_loss(roi_loc,gt_roi_loc,gt_roi_label,roi_sigma)
        # roi_lloss.backward()
        loss = rpn_closs+rpn_lloss+roi_closs+roi_lloss
        loss.backward()
        optim.step()

        # print('epoch:%d/%d, %d/%d ,cls loss: %.3f, loc loss: %.3f'%(epoch,epochs,i,len(train_data),cls_loss.item(),loc_loss.item()))