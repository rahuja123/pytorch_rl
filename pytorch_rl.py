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

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
use_cuda = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

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

preprocess = transforms.Compose([
   transforms.ToTensor()
])
def data_iter_(data,index):
    triplet = data[index][0:-1]
    print(triplet)
    label = data[index][-1]
    images = torch.FloatTensor()
    for lists in triplet:
        for im_path in lists:
            images = torch.cat((images, preprocess(np.array(cv2.imread(im_path))).unsqueeze(0)))

    sample = {'images': images,'label':label}
    return sample


#print len(data)
#sample = data_iter_(data,0)
#print len(sample['images']),sample['label']
#print sample['images'].size()

BATCH_SIZE = 8
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

model = DQN()


optimizer = optim.RMSprop(model.parameters())
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
        #print state.max(0)[1].view(1, 1).float().type()
        return state.max(0)[1].view(1, 1).float()
    else:
        return torch.FloatTensor([[random.randrange(3)]])

last_sync = 0


def optimize_model():
    global last_sync
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
    #print batch.next_state.type()
        x = [s for s in batch.next_state if s is not None]
        if len(x)>0:
            #print x
            non_final_next_states = Variable(torch.cat(x)) ##size = (_,3x64x50)
            break
    print(non_final_next_states.size())
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))
    #print state_batch.size()
    action_values = model(state_batch) ##predicted
    print(action_values.volatile)
    #print action_values.size()
    action_batch_ = torch.LongTensor((BATCH_SIZE,1))
    #action_batch_ = actionon_batch.clone()
    action_batch_ = action_batch.type_as(action_batch_)
    state_action_values = action_values.gather(1,action_batch_)
    #print action_values,state_action_values

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_values[non_final_mask]= model(non_final_next_states).data.max(1)[0]
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
    print(loss)
    print(state_batch.volatile)
    print(loss.volatile)
    #print "loss: ",loss
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    a = make_dot(loss, params=dict(model.named_parameters()))
    plt.show(a)
    for name,param in model.named_parameters():
        print(name,param.requires_grad)
        param.grad.data.clamp_(-1, 1)
        #$i+=1
    optimizer.step()

for name,param in model.named_parameters():
    param.requires_grad = True

num_episodes = 10
for i_episode in range(len(data)):
    optimizer.zero_grad()
    batch_iter_sample= data_iter_(data,i_episode)  ## size - 6*size of an image
    data_iter = Variable(batch_iter_sample['images'])
    label = Variable(torch.from_numpy(np.array(batch_iter_sample['label'])))
    data_iter = data_iter.type(FloatTensor)
    action_values = model(data_iter)
    #print data_iter.size()
    action_values = action_values.data
    #print action_values.size()
    #print data_iter.size()
    done = 0

    states = data_iter.data
    #print states.size()

    for t in count():
        action = select_action(action_values[t])

        state = torch.cat([states[t].unsqueeze(0),states[t+1].unsqueeze(0)])

        if action.numpy()[0] == 0:
            done = 1
            next_state = None
            if label == 1:
                reward = 1
            else:
                reward = -1
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
                next_state = torch.cat([states[t+1].unsqueeze(0),states[t+2].unsqueeze(0)])

        reward = torch.FloatTensor([reward]).unsqueeze(0)
        memory.push(state, action, next_state, reward)

        # Move to the next state

        state = next_state
        optimize_model()

        if done:
            break
        # Perform one step of the optimization (on the target network)
        #optimize_model()
