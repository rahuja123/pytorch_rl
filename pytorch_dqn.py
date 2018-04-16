import torch
from baseline_network_ import AlexNet
import torch.nn as nn
from torch.autograd import Variable

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.model = AlexNet()
        self.fc1 = nn.Linear(1536,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,3)
        print("ho gaya finally initialize")

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
