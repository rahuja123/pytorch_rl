import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import PIL
import cv2
import random
from pytorch_even_data_iterator import train_loader_2
from pytorch_data_iterator import train_loader_1

def concat_array(data1, data2):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    n = data2.shape[0]
    print n
    k = 4
    data_lst = []
    for i in range(0, n,4):
        data_lst.append(data1[i:i + k])
        data_lst.append(data2[i:i + k])

            # print data_lst[0].shape, data_lst[1].shape
    data = np.concatenate(data_lst)

    return data

def next_item(train_loader_1,train_loader_2):

	sample_batch_1 = next(iter(train_loader_1))
	sample_batch_2 = next(iter(train_loader_2))
	#print sample_batch_1,sample_batch_2
	label = concat_array(sample_batch_1['label'], sample_batch_2['label'])
	#print data
	#print data.shape
	data = concat_array(sample_batch_1['image'], sample_batch_2['image'])
	labels = [label]
	#labels.append(label)
	#labels.append(label)
	#labels.append(label)
	sample_batch_3 = {'image':data,'label':labels}
	return sample_batch_3


sample_batch_4 = next_item(train_loader_1,train_loader_2)
#sample_batch = next(iter(train_loader))
print sample_batch_4['image'],sample_batch_4['label']

