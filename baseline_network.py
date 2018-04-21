import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_id=100):
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
       
        #self.global_avg_pool = nn.AvgPool2d(kernel_size=(5,2))
        #self.fc1 = nn.Linear(256,num_id)
        ##self.sf1 = nn.Softmax() --> included in criterion
        ##self.cross_entropy = nn.CrossEntropyLoss() --> inculded in criterion
    
    def forward(self, x):
        x = self.features(x)
        #x = self.global_avg_pool(x)
        # x = x.view(x.size(0), 256)
        ##L2 Normalisation
        """
        for i in xrange(0,x.size(0)):
            y = torch.sum(x[i].pow(2))
            y += 1e-10
            print y
            for j in xrange(0,x.size(1)):
                x[i,j] = x[i,j]/np.sqrt(y.numpy())
        """
        #x = self.fc1(x)
        ##x = self.sf1(x)
        
        return x

def alexnet(pretrained=True):
    model = AlexNet()
    #print model	
    
    print "pretrained weights updated"
    return model


if __name__ == '__main__':
    model = AlexNet(num_id=100)
    i = 1
    for name,param in model.named_parameters():
    #    param.requires_grad = True
        if i==1:

            print param
            print name
            i+=1
    """
    pre_trained_model=torch.load('alexnet-owt-4df8aa71.pth')
    pre_trained_weights=list(pre_trained_model.items())
   
    custom_alexnet=model.state_dict()

    count=0
    for key,value in custom_alexnet.items():

        layer_name,weights=pre_trained_weights[count]
        if count == 0:
            print custom_alexnet[key][0]
        #print layer_name,": Updation done."
        custom_alexnet[key]=weights
        if count == 0:
            print custom_alexnet[key][0]
        count = count + 1
        if count>=10:
            break
    print "PreTrained Weights Updated"
    """