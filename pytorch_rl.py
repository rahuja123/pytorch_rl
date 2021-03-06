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


use_cuda = torch.cuda.is_available()
resume= True
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
STEP 1: LOADING DATASET
'''
data_file = 'pairs.npy'
data = np.load(data_file)
data = np.array(data)
print(len(data))
print("dataset loaded")

preprocess = transforms.Compose([
   transforms.ToTensor()
])

def data_iter_(data,index):
    triplet = data[index][0:-1]
    #print(len(triplet))

    label = data[index][-1]
    images = torch.FloatTensor()
    for lists in triplet:
        for im_path in lists:
            images = torch.cat((images, preprocess(np.array(cv2.imread(im_path))).unsqueeze(0)))


    sample = {'images': images,'label':label}
    return sample


BATCH_SIZE = 8
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

##MODEL
policy_net = DQN()
target_net = DQN()

##LOADING PRETRAINED WEIGHTS
if resume:
    policy_net.load_state_dict(torch.load('model_run_epi300_dqn.pt')) #give_path

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

print("DQN inititated")

if torch.cuda.is_available():
    model = model.cuda()


optimizer = optim.RMSprop(policy_net.parameters())
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
    if sample > eps_threshold:
        # print(state.max(0)[1].view(1, 1).float().type())
        return state.max(0)[1].view(1,1).float()
        # return state.max(0)[1].view(1,1).float() #original_varibale

    else:
        return torch.FloatTensor([[random.randrange(3)]])



def pool_avg_func(state, new_state):
    print("hey")
    print state,new_state
    output= torch.div(state+new_state,2)
    print output
    return output

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
    #print batch
    # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))
    #print non_final_mask
    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
        x = [s for s in batch.next_state if s is not None]
        if len(x)>0:
            non_final_next_states = Variable(torch.cat(x)) ##size = (_,3x64x50)
            break

    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    action_values = policy_net(state_batch) ##predicted
    action_batch_ = torch.LongTensor((action_batch.size(0),1))
    #action_batch_ = actionon_batch.clone()
    action_batch_ = action_batch.type_as(action_batch_)
    state_action_values = action_values.gather(1,action_batch_)
    #print action_values,state_action_values

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_values[non_final_mask]= target_net(non_final_next_states).data.max(1)[0]
    #temp_action_value= model(non_final_next_states)
    #next_state_values[non_final_mask] = temp_action_value.max(1)[0]
    #print next_state_values,non_final_mask
    #print next_state_values
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    #next_state_values.requires_grad = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA).view(BATCH_SIZE,-1)  + reward_batch
    #expected_state_action_values = torch.add((next_state_values * GAMMA).data.view(8,-1),reward_batch.data)
    #print reward_batch
    #print expected_state_action_values
    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    print(loss.data, "loss")
    # print(state_batch.volatile)
    # print(loss.volatile)
    #print "loss: ",loss
    # Optimize the model
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    #a = make_dot(loss, params=dict(model.named_parameters()))
    #plt.show(a)
    for name,param in policy_net.named_parameters():
        #print(name,param.requires_grad)
        param.grad.data.clamp_(-1, 1)
        #$i+=1
    optimizer.step()

#for name,param in model.named_parameters():
#    param.requires_grad = True

for i_episode in range(len(data)):
    print("episode", i_episode)
    
    batch_iter_sample= data_iter_(data,i_episode)  ## size - 6*size of an image #done
    data_iter = Variable(batch_iter_sample['images'])
    label = Variable(torch.from_numpy(np.array(batch_iter_sample['label'])))

    data_iter = data_iter.type(FloatTensor)
    states = data_iter.data
    prev_state= data_iter[0:2]

    action_values = policy_net(data_iter)
    action_values = action_values.data
    #print action_values

    for t in count():
        # print(type(prev_state,))
        # print(type(data_iter[2*t : 2*t+1]))
        print("t=", t)
        # print(prev_state)
        # print(data_iter[2*t : 2*t+2])
        #pool_avg=pool_avg_func(prev_state, data_iter[2*t : 2*t+2])
        #prev_state = pool_avg
        #print(pool_avg)


        #)
        # print(action_values[0])
        
        done = 0

        action = select_action(action_values[t])
        # print(action_values[t])
        # print(action)

        state = torch.cat([states[2*t].unsqueeze(0),states[2*t+1].unsqueeze(0)])
        # print(state)
        #next_state = torch.cat([states[2*t+2].unsqueeze(0),states[2*t+3].unsqueeze(0)])
        # print(next_state)

        # print(pool_avg(state, next_state), "pooled")

        if action.numpy()[0] == 0:
            done = 1
            next_state = None
            if label == 1:
                reward = 1
            else:
                reward = -2
        elif action.numpy()[0] == 1:
            next_state = None
            done = 1
            if label == 0:
                reward = 1
            else:
                reward = -1

        elif action.numpy()[0] == 2:

            if t==2:
                done = 1
                reward = -1
                next_state = None
            else:
                reward = 0.2
                next_state = torch.cat([states[2*t+2].unsqueeze(0),states[2*t+3].unsqueeze(0)])

        reward = torch.FloatTensor([reward]).unsqueeze(0)
        memory.push(state, action, next_state, reward)

        # Move to the next state

        #state = next_state
        optimize_model()
    


        if done:
            break
        # Perform one step of the optimization (on the target network)
        #optimize_model()
    if(i_episode%100==0):
            save_checkpoint(policy_net.state_dict(), 'model_run_epi%d_dqn.pt' %(i_episode))

            # Update the target network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
count = 0

for i in range(0,500):
    optimize_model()
    if i % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())


        
save_checkpoint(policy_net.state_dict(), 'model_run_complete_dqn.pt')
