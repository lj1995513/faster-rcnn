import torch
class SmoothL1Loss(torch.nn.Module):
    def __init__(self,device=torch.device('cpu')):
        super(SmoothL1Loss,self).__init__()
        self.device=device
    def forward(self,pred_loc,gt_loc,gt_cls,sigma=3):
        weight = torch.zeros_like(gt_loc,device=self.device)
        weight[gt_cls>0] = 1
        loc_loss = smooth_l1_loss(pred_loc,gt_loc,weight,sigma)
        loc_loss/=torch.sum(gt_cls>=0).float()
        return loc_loss
def smooth_l1_loss(pred_loc,gt_loc,weight,sigma):
    sigma2 = sigma**2
    diff = weight*(pred_loc-gt_loc)
    abs_diff = torch.abs(diff)
    flag = (abs_diff<1./sigma2).float()
    y = (flag*sigma2/2.)*torch.pow(diff,2)+(1-flag)*(abs_diff-0.5/sigma2)
    return torch.sum(y)
