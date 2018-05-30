
import torch.utils.model_zoo as model_zoo
#from utils import *
#from viper_test import *
from viper_train import *
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}



def save_checkpoint(state, filename='checkpoint'):
    torch.save(state, filename)
    print("model_saved")


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        #label= label.data
        #print (type(output1.data))
        #output1 = Variable(output1.data.long())
        #output2 = Variable(output2.data.long())
        #print (type(labels_.data))
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive



class AlexNet(nn.Module):

    def __init__(self, num_classes=632):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 100),
        )

    def forward_once(self, x):
        x = self.features(x)
        temp = x
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def forward(self, input1, input2):
        output1= self.forward_once(input1)
        output2= self.forward_once(input2)
        return output1, output2


#def alexnet(pretrained=False, **kwargs):
if __name__ == '__main__':
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet()
    #model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))

    # pre_trained_model=torch.load('pretrained_model_weights/alexnet-owt-4df8aa71.pth')
    # pre_trained_weights=list(pre_trained_model.items())
    #
    # custom_alexnet=model.state_dict()
    #
    # count=0
    # for key,value in custom_alexnet.items():
    #
    #     layer_name,weights=pre_trained_weights[count]
    #     #if count == 0:
    #     #    print custom_alexnet[key]
    #     print(layer_name,": Updation done.")
    #     custom_alexnet[key]=weights
    #     #if count == 0:
    #     #    print custom_alexnet[key][0]
    #     count = count + 1
    #     if count==14:
    #         break
    #             #model.state_dict()[key] = custom_alexnet[key]
    #             #print custom_alexnet[key]
    #             #weights
    #             #print model.state_dict()[key]
    #     print(count)
    #
    # model.load_state_dict(custom_alexnet)
    # #print custom_alexnet['classifier.4.bias']
    # #print model.state_dict()['classifier.4.bias']
    # for name,param in model.named_parameters():
    #     param.requires_grad = True


    test = True
    if test :

        data_file = 'pairs_train_viper.npy'
    # else:
    #     im = 'VIPeR/cam_a/000_45.bmp'
    #     index = 0
    #     image = []
    #     images = []
    #     image.append(im)
    #     image.append(index)
    #     image = np.array(image)
    #     images.append(image)
    #     print(images)
    #     np.save('test_image',images)
        #data_file = np.load('test_image.npy')

    train_dataset = Viper_Train(name=data_file,
                       data_dir = 'data',
                       transform=transforms.Compose([
                                            #transforms.RandomReSizedCrop(128),
                                            transforms.ToPILImage(),
                                            #transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor()
                                            #transforms.Normalize(mean=[128.0, 128.0, 128.0],
                                            #std=[1, 1, 1])
                        ]))

    train_loader = DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)


    test_dataset = Viper_Train(name=data_file,
                       data_dir = 'data',
                       transform=transforms.Compose([
                                            #transforms.RandomReSizedCrop(128),
                                            transforms.ToPILImage(),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor()
                                            #transforms.Normalize(mean=[128.0, 128.0, 128.0],
                                            #std=[1, 1, 1])
                        ]))

    test_loader = DataLoader(dataset=test_dataset,batch_size=1,shuffle=True)



    if test:
        #model.load_state_dict(torch.load('classifier/model_run_complete_dqn_viper.pth'))
        model = torch.load('classifier/model_run_complete_dqn_prid.pth')
        for i, (sample_batch) in enumerate(test_loader):
            image0 = sample_batch['image0']
            image1 = sample_batch['image1']

            #print images,"imageeeeeeeeeeeeeeeeees"
            labels = torch.from_numpy(np.asarray(sample_batch['label'],dtype='int64'))
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            else:
                image0 = Variable(image0)
                image1 = Variable(image1)

                #labels = Variable(labels)
            output1, output2 = model(image0, image1)
            # _, predicted = torch.max(outputs.data, 1)
            #print("Predicted:",predicted, "Actual:", labels)
            euclidean_distance = F.pairwise_distance(output1, output2)
            print(euclidean_distance, labels)




    '''INSTANTIATE LOSS CLASS'''
    criterion = ContrastiveLoss()


    ''' INSTANTIATE OPTIMIZER CLASS '''
    learning_rate = 0.001

    optimizer = torch.optim.Adam(model.parameters(),lr = 0.0005)
    iter = 0
    for epoch in range(10):
        for i, (sample_batch) in enumerate(train_loader):
            correct = 0
            image0 = sample_batch['image0']
            image1=  sample_batch['image1']
            labels = torch.from_numpy(np.asarray(sample_batch['label'],dtype='int64'))
            if torch.cuda.is_available():
                image0 = Variable(image0.cuda())
                image1 = Variable(image1.cuda())

                labels = Variable(labels.cuda())
            else:
                image0 = Variable(image0)
                image1 = Variable(image1)

                labels = Variable(labels)


        # Clear gradients w.r.t. named_parameterss
            optimizer.zero_grad()

        # Forward pass to get output/logits
            output1, output2 = model(image0, image1)
            #_, predicted = torch.max(outputs.data, 1)
            #   print predicted
        #print outputs.size(),labels.size()
        # Calculate Loss: softmax --> cross entropy loss
            print(type(labels.data))
            print(labels)
            labels = Variable(labels.data.float())
            #print (tyVariablepe(labels_))
            #labels_ = Variable(labels_)
            # print(labels)
            #correct += (predicted == labels_).sum()
            #print("CORRECT", correct)
            loss = criterion(output1, output2, labels)
            print("Loss: ",loss.data[0])
        # Getting gradients w.r.t. parameters
            loss.backward()
        #    for param in model.parameters():
        #        param.grad.data.clamp_(-1, 1)

        # Updating parameters
            optimizer.step()


            iter += 1

        #if iter % 100== 0:

        save_checkpoint(model, 'classifier/model_run_complete_dqn_prid.pth')
    """
    #Calculate Accuracy

    model.load_state_dict(torch.load('classifier/model_run_complete_dqn.pth'))
    correct = 0
    total = 0
            # Iterate through test dataset
    for i, (sample_test_batch)  in enumerate(train_loader):
                #######################
                #  USE GPU FOR MODEL  #
                #######################

        test_images = sample_test_batch['image']
        test_labels = torch.from_numpy(np.asarray(sample_test_batch['label'],dtype='int64'))
        #print test_labels
        if torch.cuda.is_available():
            images = Variable(test_images.cuda())
        else:
            images = Variable(test_images)

                # Forward pass only to get logits/output
        outputs = model(images)
        #print outputs
                # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)
        print predicted
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
        #print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.data[0], accuracy))
        #
    print "ACCURACY:",accuracy
    """
