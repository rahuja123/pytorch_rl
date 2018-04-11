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
        self.all_images_file = self.data[:,0]
        self.image_labels = self.data[:,1]
        self.data_dir = data_dir
        self.transform = transform
        self.batch_size = 8
        
    def __len__(self):
        return len(self.all_images_file)
    
    def __getitem__(self,index):
        img_path = self.all_images_file[index]
        label = self.image_labels[index]
        im = io.imread(img_path)
        #print im.shape
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

        
        if self.transform:
            im = self.transform(im)

        sample = {'image':im,'label':label}

        return sample



dataset_1 = Prid_Dataset(name='images.npy',
                       data_dir = '../../prid_2011',
                       transform=transforms.Compose([
        									#transforms.RandomReSizedCrop(128),
        									transforms.ToPILImage(),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor()
        									#transforms.Normalize(mean=[128.0, 128.0, 128.0],
                             				#std=[1, 1, 1])
    					]))

#train_loader_1 = DataLoader(dataset=dataset_1,batch_size=8,shuffle=True)


#sample_batch = next(iter(train_loader))
#print sample_batch['image'],sample_batch['label']

#for i_batch, sample_batched in enumerate(train_loader):
#    print(i_batch, sample_batched['image'],
#          sample_batched['label'])