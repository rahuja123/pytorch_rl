from baseline_network import AlexNet
import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torch.autograd import Variable
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import PIL
import cv2
import random
from pytorch_data_iterator import Prid_Dataset
from pytorch_test_data_iterator import Prid_Test_Dataset
import argparse
import torch.utils.model_zoo as model_zoo


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prid_Dataset_Person Reidentification')

    parser.add_argument('--data-dir', type=str,
                        default="../../prid_2011/multi_shot/",
                        help='data directory')
    parser.add_argument('--num-examples', type=int, default=20000,
                        help='the number of training examples')
    parser.add_argument('--num-id', type=int, default=100,
                        help='the number of training ids')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='the batch size')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='the initial learning rate')
    parser.add_argument('--num-epoches', type=int, default=1,
                        help='the number of training epochs')
    parser.add_argument('--mode', type=str, default='ilds_baseline_b4',
                        help='save names of model and log')
    parser.add_argument('--lsoftmax', action='store_true', default=False,
                        help='if use large margin softmax')
    parser.add_argument('--verifi-label', action='store_true', default=False,
                        help='if add verifi label')
    parser.add_argument('--verifi', action='store_true', default=False,
                        help='if use verifi loss')
    parser.add_argument('--triplet', action='store_true', default=False,
                        help='if use triplet loss')
    parser.add_argument('--lmnn', action='store_true', default=True,
                        help='if use LMNN loss')    
    parser.add_argument('--center', action='store_true', default=False,
                        help='if use center loss')
    parser.add_argument('--verifi-threshd', type=float, default=0.9,
                        help='verification threshold')
    parser.add_argument('--triplet-threshd', type=float, default=0.9,
                        help='triplet threshold')
    return parser.parse_args()


args = parse_args()

print args
batch_size = args.batch_size #-----
num_epoch = 1 #-----
lr = args.lr#-----

''' INSTANTIATING THE MODEL'''
model = AlexNet()
model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}
#model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))


'''ITERABLE DATASET'''

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

train_loader_1 = DataLoader(dataset=dataset_1,batch_size=8,shuffle=True)

test_dataset = Prid_Test_Dataset(name='test_images.npy',
                       data_dir = '../../prid_2011',
                       transform=transforms.Compose([
                                            #transforms.RandomReSizedCrop(128),
                                            transforms.ToPILImage(),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor()
                                            #transforms.Normalize(mean=[128.0, 128.0, 128.0],
                                            #std=[1, 1, 1])
                        ]))
test_loader = DataLoader(dataset=test_dataset,batch_size=8,shuffle=True)



#######################
#  USE GPU FOR MODEL  #
#######################

if torch.cuda.is_available():
    model.cuda()

'''INSTANTIATE LOSS CLASS'''
criterion = nn.CrossEntropyLoss()


''' INSTANTIATE OPTIMIZER CLASS '''
learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

''' TRAIN THE MODEL '''
iter = 0
for epoch in range(3):
    for i, (sample_batch) in enumerate(train_loader_1):
        #print "Batch: ",i
        #######################
        #  USE GPU FOR MODEL  #
        #######################

        images = sample_batch['image']
        labels = torch.from_numpy(np.asarray(sample_batch['label'],dtype='int64'))
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)
        
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
        
        # Forward pass to get output/logits
        outputs = model(images)

        #print outputs.size(),labels.size()
        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)
        print "Loss: ",loss.data[0]
        # Getting gradients w.r.t. parameters
        loss.backward()
        
        # Updating parameters
        optimizer.step()
        
        
        iter += 1
        """
        if iter % 1== 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for (sample_test_batch) in test_loader:
                #######################
                #  USE GPU FOR MODEL  #
                #######################

                test_images = sample_test_batch['image']
                test_labels = torch.from_numpy(np.asarray(sample_test_batch['label'],dtype='int64'))
                if torch.cuda.is_available():
                    images = Variable(test_images.cuda())
                else:
                    images = Variable(test_images)
                
                # Forward pass only to get logits/output
                outputs = model(images)
                
                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)
                
                # Total number of labels
                total += test_labels.size(0)
                
                #######################
                #  USE GPU FOR MODEL  #
                #######################
                # Total correct predictions
                if torch.cuda.is_available():
                    correct += (predicted.cpu() == test_labels.cpu()).sum()
                else:
                    correct += (predicted == test_labels).sum()
            
            accuracy = 100 * correct / total
            
            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.data[0], accuracy))
        """    



