import torch
from baseline_network import AlexNet
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.init as init

# def weight_init(m):
#     if isinstance(m, nn.Linear):
#         size = m.weight.size()
#         fan_out = size[0] # number of rows
#         fan_in = size[1] # number of columns
#         variance = np.sqrt(2.0/(fan_in + fan_out))
#         m.weight.data.normal_(0.0, variance)
#         return m


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.model = AlexNet()

        self.fc1 = nn.Linear(1536,256)
        init.xavier_uniform(self.fc1.weight, gain=np.sqrt(2))
        init.constant(self.fc1.bias, 0.1)
        self.fc2 = nn.Linear(256,128)
        init.xavier_uniform(self.fc2.weight, gain=np.sqrt(2))
        init.constant(self.fc2.bias, 0.1)
        self.fc3 = nn.Linear(128,3)
        init.xavier_uniform(self.fc3.weight, gain=np.sqrt(2))
        init.constant(self.fc3.bias, 0.1)

    def features(self,x):
        x = self.model(x)
        return x

    def forward(self,x,flag=0):
    	#x = self.model(x)
    	#a = x[0:x.size(0)/2]
    	#b = x[x.size(0)/2:x.size(0)]
    	#y = a+b
        x = self.features(x)
        new_batch_size = int(x.size(0)/2)
        # print(x.size(0), "x.size(0)")
        # print(new_batch_size, "new batch size")
        # print(x.size, "x.size")


        x1 = Variable(torch.randn(new_batch_size, x.size(1)).type(torch.FloatTensor))
        for i in range(0,new_batch_size):
            x1[i] = x[2*i]+x[2*i+1]

        x = self.fc1(x1)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
if __name__ == '__main__':
    model = DQN()
    #print model.paramteres()
    #print model.paramteres()
    for name,param in model.named_parameters():
        param.requires_grad = True
        # print(param, "param")
        # print(name, "name")
    pre_trained_model=torch.load('alexnet-owt-4df8aa71.pth')
    pre_trained_weights=list(pre_trained_model.items())
   
    custom_alexnet=model.state_dict()

    count=0
    for key,value in custom_alexnet.items():

        layer_name,weights=pre_trained_weights[count]
        #if count == 0:
        #    print custom_alexnet[key]
        #print layer_name,": Updation done."
        custom_alexnet[key]=weights
        #if count == 0:
        #    print custom_alexnet[key][0]
        count = count + 1
        if count>=10:
            break
    print "PreTrained Weights Updated"
