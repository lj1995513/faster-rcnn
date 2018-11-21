import torch
import math
def _roi_pooling_slice(size,stride,max_size,roi_offset):
    start = math.floor(stride*size)
    end = math.ceil(stride*(size+1))
    start = min(max(start+roi_offset,0),max_size)
    end = min(max(end+roi_offset,0),max_size)
    return start,end,end-start

class ROIPooling:
    def __init__(self,outh,outw,spatial_scale,device=torch.device('cpu')):
        self.outh,self.outw = outh,outw
        self.spatial_scale = spatial_scale
        self.device=device
    def __call__(self,bottom_data,bottom_rois):
        channels,height,width = bottom_data.shape[1:]
        n_rois = bottom_rois.shape[0]
        top_data = torch.zeros((n_rois,channels,self.outh,self.outw),dtype=bottom_data.dtype,device=self.device)
        # self.argmax_data = torch.zeros(top_data.shape,dtype=torch.int32)
        for i in range(n_rois):#每个ROI max pooling 为512*7*7
            idx,ymin,xmin,ymax,xmax = bottom_rois[i]
            xmin = int(torch.round(xmin*self.spatial_scale))
            xmax = int(torch.round(xmax*self.spatial_scale))
            ymin = int(torch.round(ymin*self.spatial_scale))
            ymax = int(torch.round(ymax*self.spatial_scale))
            roi_width = max(xmax-xmin+1,1.)
            roi_height = max(ymax-ymin+1,1.)
            strideh = 1.*roi_height/self.outh
            stridew = 1.*roi_width/self.outw
            for outh in range(self.outh):
                starth,endh,lenh = _roi_pooling_slice(outh,strideh,height,ymin)
                if endh<=starth:
                    continue
                for outw in range(self.outw):
                    startw,endw,lenw = _roi_pooling_slice(outw,stridew,width,xmin)
                    if endw<=startw:
                        continue
                    roi_data = bottom_data[int(idx),:,starth:endh,startw:endw].reshape(channels,-1)
                    top_data[i,:,outh,outw],_=torch.max(roi_data,dim=1)

        return top_data
                    