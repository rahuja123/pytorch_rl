import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image
import cv2 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from pytorch_dqn import DQN
from torchviz import make_dot
import shutil
from alexnet import AlexNet
from viper_train import *



use_cuda = torch.cuda.is_available()
resume= False
test = True
new = 1
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

'''
LOADING DATASET
'''
data_file = 'testing_pairs.npy'

data = np.load(data_file)
data = np.array(data)
#test_data = data[900:-1,:]
test_data = data
#train_data = data[0:1000,:]
print(len(data))
print("dataset loaded")


'''
PREPROCESSING DATASET
'''
normalize_ = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5])

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

'''
PROVIDING DATA FOR 1 EPISODE
'''
def data_iter_(data,index):
    triplet = data[index][0:-1]
    #print(len(triplet))

    label = data[index][-1]
    images = torch.FloatTensor()
    for lists in triplet:
        for im_path in lists:
            #print im_path

            images = torch.cat((images, preprocess(np.array(cv2.imread(im_path))).unsqueeze(0)))
            #print images.size


    sample = {'images': images,'label':label}
    return sample


BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

##MODEL
policy_net = DQN()
target_net = DQN()
#alexnet = AlexNet()
#alexnet.load_state_dict(torch.load('classifier/model_run_complete_dqn_viper.pt'))
#alexnet.eval()
#alexnet = torch.load('classifier/model_run_complete_dqn_viper.pth')
alexnet = torch.load('classifier/model_run_complete_dqn_viper.pth')
alexnet.eval()


print "Alexnet weights loaded"

##LOADING PRETRAINED WEIGHTS
if resume:
    policy_net.load_state_dict(torch.load('alexnet_model/model_run_complete_dqn_viper.pth'))
    print "Already Trained Model load_state_dict" #give_path

target_net.load_state_dict(policy_net.state_dict())
#target_net.eval()
#policy_net.eval()

print("DQN inititated")



if torch.cuda.is_available():
    model = model.cuda()

count = 0
for name,param in policy_net.named_parameters():
        param.requires_grad = True


optimizer = optim.Adam(policy_net.parameters(),lr = 0.00001)

memory = ReplayMemory(10000)

num_episodes = 10
index = 0
steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold or test:
        return state.max(0)[1].view(1,1).float()
    else:
        return torch.FloatTensor([[random.randrange(3)]])


def pool_avg_func(state, new_state):
    output= torch.div(state+new_state,2)
    return output


if test:
    policy_net = torch.load('alexnet_model/model_run_complete_dqn_viper_final.pth')
    alexnet = torch.load('classifier/model_run_complete_dqn_viper.pth')
    policy_net.eval()
    alexnet.eval()
    
    """
    test_dataset = Viper_Train(name='test_image.npy',
                       data_dir = 'data',
                       transform=transforms.Compose([
                                            #transforms.RandomReSizedCrop(128),
                                            transforms.ToPILImage(),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor()
                                            #transforms.Normalize(mean=[128.0, 128.0, 128.0],
                                            #std=[1, 1, 1])
                        ]))

    test_loader = DataLoader(dataset=test_dataset,batch_size=2,shuffle=False)



    for i, (sample_batch) in enumerate(test_loader):
        images = sample_batch['image']

        labels = torch.from_numpy(np.asarray(sample_batch['label'],dtype='int64'))
        images = Variable(images)
        _,features = alexnet(images)

    #print "Predicted:",predicted, "Actual:", labels
    
    """
    print "Already Trained Model load_state_dict" #give_path
    for name,param in policy_net.named_parameters():
        param.requires_grad = False

    for name,param in alexnet.named_parameters():
        param.requires_grad = False
    correct = 0

    for i_episode in range(len(test_data)):
        batch_iter_sample= data_iter_(test_data,i_episode) 
        #print batch_iter_sample['images']
         ## size - 6*size of an image #done
        data_iter = Variable(batch_iter_sample['images'])
        #label = Variable(torch.from_numpy(np.array(batch_iter_sample['label'])))

        data_iter = data_iter.type(FloatTensor)
        #print data_iter
        outputs, features  = alexnet(data_iter)

        if new:
            features = features.view(features.size(0),-1)

            x1 = Variable(torch.randn(3, features.size(1)).type(torch.FloatTensor))
            for i in range(0,3):
                    x1[i] = (features[2*i] - features[2*i+1]).abs()

            states = x1.data
            states[1,:] = pool_avg_func(states[0,:],states[1,:])
            states[2,:] = pool_avg_func(states[1,:],states[2,:])
            features_ = Variable(states)
            action_values = policy_net(features_)
            #print states.size()
            #print action_values.size()
        else:
            states = features.data
            action_values = policy_net(features)


        _, predicted_= torch.max(outputs.data, 1)
        #print predicted
        #print features[0]

        #action_values = policy_net(features)
        action_values = action_values.data
        lab = batch_iter_sample['label']
        
        """
        images = sample_batch['image']
        lab = torch.from_numpy(np.asarray(sample_batch['label'],dtype='int64'))
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            #labels = Variable(labels.cuda())
        else:
            images = Variable(images)
            #labels = Variable(labels)

        _, features  = alexnet(images)
        action_values = policy_net(features)
        action_values = action_values.data

        """
        #print i_episode
        t = 0
        while True:

            #print("t=", t)
            if t>2:
                break

            done = 0
            #correct = 4
            #action = select_action(action_values[t])
            action = action_values[t].max(0)[1].view(1,1)
            print "Action:",action,"Label:",lab
            
            if action.numpy()[0] == 0:
                #print "Action:",action,"Label:",lab
                if lab == 1:
                    correct = correct +1
                print "OVER"    
                done = 1   
                break

            elif action.numpy()[0] == 1:
                #print "Action:",action,"Label:",lab
                if lab == 0:
                        correct = correct +1
                print "OVER"            
                done = 1   
                break
            
            if t>=2:
                #print "Action:",action,"Label:",lab
                print "OVER"
                done = 1
                break

            #print done
            if done:
                break
            t = t+1

   

    #print "ACCURACY ON TESTING: ", correct 
    exit()





def save_checkpoint(state, filename='checkpoint'):
    torch.save(state, filename)
    print("model_saved")


def optimize_model():


    if len(memory) < BATCH_SIZE:
        return
    while True:
        transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
        batch = Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))
    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
        x = [s for s in batch.next_state if s is not None]
        if len(x)>0:
            non_final_next_states = Variable(torch.cat(x),volatile= True) ##size = (_,3x64x50)
            break
    if new:
        state_batch = Variable(torch.cat(batch.state))
    else:
        state_batch = Variable(torch.cat(batch.state))

    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    
    action_values = policy_net(state_batch) ##predicted
    action_batch_ = torch.LongTensor((action_batch.size(0),1))
    action_batch_ = action_batch.type_as(action_batch_)
    state_action_values = action_values.gather(1,action_batch_)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_values[non_final_mask]= target_net(non_final_next_states).max(1)[0].detach().data
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    #next_state_values.requires_grad = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA).view(BATCH_SIZE,-1)  + reward_batch
    #expected_state_action_values = (next_state_values * GAMMA)+ reward_batch
    expected_state_action_values = Variable(expected_state_action_values.data)
    #print expected_state_action_values
    #expected_state_action_values = torch.add((next_state_values * GAMMA).data.view(8,-1),reward_batch.data)
    #print reward_batch
    #print expected_state_action_values
    # Compute Huber loss
    #loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    #print(loss.data, "loss")
    # print(state_batch.volatile)
    # print(loss.volatile)
    #print "loss: ",loss
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    #a = make_dot(loss, params=dict(model.named_parameters()))
    #plt.show(a)
    for param in policy_net.parameters():
        #print(name,param.requires_grad)
        param.grad.data.clamp_(-1, 1)
        #$i+=1
    optimizer.step()




for i_episode in range(len(train_data)):
    print("episode", i_episode)
    
    batch_iter_sample= data_iter_(train_data,i_episode)
    #print batch_iter_sample
    #im = batch_iter_sample['images'].numpy()[0]
    #im = np.transpose(im,(1,2,0))
    #print im
    #plt.figure()
    #plt.imshow(im)
    #plt.show() 
    #for i in range(6):
    #    im = batch_iter_sample['images'].numpy()[i]
    #    im = np.transpose(im,(1,2,0))
    #    print im
    #   plt.figure()
    #    plt.imshow(im)
    #    plt.show() 

         ## size - 6*size of an image #done
    
    data_iter = Variable(batch_iter_sample['images'])
    label = Variable(torch.from_numpy(np.array(batch_iter_sample['label'])))
    #print data_iter.data.numpy()[0].shape

    data_iter = data_iter.type(FloatTensor)
    #states = data_iter.data
    #prev_state= data_iter[0:2]
    lab = batch_iter_sample['label']

    _,features = alexnet(data_iter)


    if new:
        features = features.view(features.size(0),-1)

        x1 = Variable(torch.randn(3, features.size(1)).type(torch.FloatTensor))
        for i in range(0,3):
                x1[i] = (features[2*i] - features[2*i+1]).abs()

        states = x1.data
        states[1,:] = pool_avg_func(states[0,:],states[1,:])
        states[2,:] = pool_avg_func(states[1,:],states[2,:])
        features_ = Variable(states)
        action_values = policy_net(features_)
        #print states.size()
        #print action_values.size()
    else:
        states = features.data
        action_values = policy_net(features)

    action_values = action_values.data

    #print action_values
    t = 0
    while True:
        print("t=", t)
   
        done = 0

        action = select_action(action_values[t])
        # print(action_values[t])
        # print(action)
        if new:
            state = states[t,:].unsqueeze(0)
        else:
            state = torch.cat([states[2*t].unsqueeze(0),states[2*t+1].unsqueeze(0)])

        if action.numpy()[0] == 0:
            done = 1
            next_state = None
            if lab == 1:
                #print "CORRECTTTTTTTTTTTTTTTTTTTTTTT"
                reward = 1
            else:
                reward = -20
        elif action.numpy()[0] == 1:
            next_state = None
            done = 1
            if lab == 0:
                #print "CORECTTTTTOOOOOOOOOOOOOOOOOOOOOO"
                reward = 1
            else:
                reward = -20

        elif action.numpy()[0] == 2:

            if t==2:
                done = 1
                reward = -20
                next_state = None
            else:
                reward = 0.2
                if new:
                    next_state = states[t+1,:].unsqueeze(0)
                else:

                    next_state = torch.cat([states[2*t+2].unsqueeze(0),states[2*t+3].unsqueeze(0)])

        reward = torch.FloatTensor([reward]).unsqueeze(0)
        memory.push(state, action, next_state, reward)

        # Move to the next state

        #state = next_state
        optimize_model()
    
        t = t+1

        if done:
            break
        # Perform one step of the optimization (on the target network)
        #optimize_model()
    if(i_episode%100==0):
            save_checkpoint(policy_net.state_dict(), 'alexnet_model/model_run_epi%d_dqn.pt' %(i_episode))

            # Update the target network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
count = 0

for i in range(0,1000):
    optimize_model()
    if i % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    if(i%100==0):
        save_checkpoint(policy_net.state_dict(), 'alexnet_model/model_run_epi%d_dqn.pt' %(i+300))



    
save_checkpoint(policy_net, 'alexnet_model/model_run_complete_dqn_viper.pth')
