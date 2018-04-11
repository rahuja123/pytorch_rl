import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import PIL
import cv2
import random

class Prid_Dataset(Dataset):
    """Loading of PRID-Dataset"""
    
    def __init__(self, name, data_dir,transform = None):
        self.data = np.load(name)
        self.data_dir = data_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        img_path = self.data[index][0:-1]
        images = []
        label = self.data[index][-1]
        for path_list in img_path:
            for j in path_list:
                images.append(io.imread(j))
        #im = cv2.imread(img_path)
        #print im.shape
        #print len(images[0])
        """
        crop_h, crop_w = 224,112
        ori_h, ori_w = im.shape[:2]
        resize = 128
        if ori_h < ori_w:
            h, w = resize, int(float(resize) / ori_h * ori_w)
        else:
            h, w = int(float(resize) / ori_w * ori_h), resize

        if h != ori_h:
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)

        x, y = (w - crop_w) / 2, (h - crop_h) / 2
        x = random.randint(0, w - crop_w)
        y = random.randint(0, h - crop_h)
        im = im[y:y + crop_h, x:x + crop_w, :]
        """
        
        if self.transform:
            image_new = []
            for i in images:
                image_new.append(self.transform(i))
            images = image_new
        print len(images)    
        sample = {'image':images,'label':label}
        return 



dataset = Prid_Dataset(name='pairs.npy',
                       data_dir = '../../prid_2011/regions',
                       transform=transforms.Compose([
        									#transforms.RandomReSizedCrop(128),
        									transforms.ToPILImage(),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor()
        									#transforms.Normalize(mean=[128.0, 128.0, 128.0],
                             				#std=[1, 1, 1])
    					]))

train_loader = DataLoader(dataset=dataset,shuffle=True)


sample_batch = next(iter(train_loader))
#print sample_batch['image'],sample_batch['label']

#for i_batch, sample_batched in enumerate(train_loader):
#    print(i_batch, sample_batched['image'],
#          sample_batched['label'])