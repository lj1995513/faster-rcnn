import cv2
import numpy as np
from os import path
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
from torchvision import transforms
class VOC07(Dataset):
    def __init__(self,data_dir,split=None,transform=None):
        self.class_name = ['person','bird','cat','cow','dog','horse','sheep','aeroplane','bicycle','boat','bus','car','motorbike','train','bottle','chair','diningtable','pottedplant','sofa','tvmonitor']
        self.data_dir = data_dir
        if split=='train':
            p = path.join(data_dir,'ImageSets','Main','train.txt')
        elif split=='val':
            p = path.join(data_dir,'ImageSets','Main','val.txt')
        else:
            p = path.join(data_dir,'ImageSets','Main','trainval.txt')
        with open(p) as f:
            self.imgs = [img.strip() for img in f.readlines()]
        self.transform = transform
    @property
    def class_num(self):
        return len(self.class_name)
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self,idx):
        name = self.imgs[idx]
        img = path.join(self.data_dir,'JPEGImages',name+'.jpg')
        img = cv2.imread(img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        anno = path.join(self.data_dir,'Annotations',name+'.xml')
        bbox = []
        label = []
        tree = ET.parse(anno)
        for i in tree.findall('object'):
            box = i.find('bndbox')
            bbox.append([
                int(box.find('ymin').text),
                int(box.find('xmin').text),
                int(box.find('ymax').text),
                int(box.find('xmax').text)
            ])
            label.append(self.class_name.index(i.find('name').text))
        bbox = np.array(bbox)
        label = np.array(label)
        sample = {'image':img,'box':bbox,'label':label}
        if self.transform:
            sample = self.transform(sample)
        return sample
class Rescale:
    def __init__(self,output_size):
        self.output_size = output_size
    def __call__(self,sample):
        image = sample['image']
        box = sample['box']
        h,w = image.shape[:2]
        if isinstance(self.output_size,int):
            if h>w:
                new_h, new_w = self.output_size*h//w,self.output_size
            else:
                new_h,new_w = self.output_size,self.output_size*w//h
        else:
            new_w,new_h = self.output_size
        image = cv2.resize(image, (new_w,new_h))
        box = box*[new_h/h,new_w/w,new_h/h,new_w/w]
        return {
            'image':image,
            'box':box,
            'label':sample['label']
        }
class RandomCrop:
    def __init__(self,output_size):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size
    def __call__(self,sample):
        new_w,new_h = self.output_size
        while True:
            image,box,label = sample['image'],sample['box'],sample['label']
            h,w = image.shape[:2]
            top = np.random.randint(0,h-new_h)
            left = np.random.randint(0,w-new_w)
            image = image[top:top+new_h,left:left+new_w]
            box = box-[top,left,top,left]
            box[:,0::2] = np.clip(box[:,0::2],0,new_h)
            box[:,1::2] = np.clip(box[:,1::2],0,new_w)
            box_h = box[:,2]-box[:,0]
            box_w = box[:,3]-box[:,1]
            keep = np.where((box_h>20) & (box_w>20))[0]
            box = box[keep]
            label = label[keep]
            if box.shape[0]!=0:
                break
        return {
            'image':image,
            'box':box,
            'label':label
        }
class ToTensor:
    def __call__(self,sample):
        image = np.array(sample['image'])
        image = transforms.ToTensor()(image)
        return {
            'image':image,
            'box':torch.tensor(sample['box'],dtype=torch.float),
            'label':torch.tensor(sample['label'],dtype=torch.int64)
        }
class Normlize:
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    def __call__(self,sample):

        image = sample['image']
        image = self.normalize(image)
        return {
            'image':image,
            'box':sample['box'],
            'label':sample['label']
        }

                                
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    train_data = VOC07('VOCdevkit/VOC2007',split='train',transform=transforms.Compose([
        Rescale(801),
        RandomCrop((800,600)),
        ToTensor(),
        Normlize(),
    ]))
    data_loader = DataLoader(train_data,batch_size=1)
    for i,sample in enumerate(data_loader):
        print(i)
        # image, boxes = sample['image'][0],sample['box'][0]
        # image = torch.tensor([0.485, 0.456, 0.406]).reshape(-1,1,1) + torch.tensor([0.229, 0.224, 0.225]).reshape(-1,1,1)*image
        # plt.imshow(image.numpy().transpose(1,2,0))
        # plt.show()
        # for i in range(boxes.shape[0]):
        #     box = boxes[i]
        #     x,y=box[1],box[0]
        #     w = box[3]-box[1]
        #     h = box[2]-box[0]
        #     plt.gca().add_patch(Rectangle((x,y),w,h,fill=False,color='r'))
        #plt.show()