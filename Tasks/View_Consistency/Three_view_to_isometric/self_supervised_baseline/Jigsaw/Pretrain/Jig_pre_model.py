from __future__ import print_function, division

from torchvision import models
import torch.nn as nn
import torch
import numpy as np
import bagnets.pytorchnet


class Jig(nn.Module):

    def __init__(self,model_type=None):
        super(Jig, self).__init__()
        
        if model_type=="Bagnet33":
            self.model = bagnets.pytorchnet.bagnet33(pretrained=False)
            num_ftrs = self.model.fc.in_features
            self.model.fc=torch.nn.Linear(num_ftrs, 1000)

        if model_type=="vgg16":
            self.model = models.vgg16(pretrained=False)
            # num_ftrs = self.model.classifier[6].in_features
            # self.model.classifier[6]=torch.nn.Linear(num_ftrs, 1000)
            self.fc0 = nn.Sequential()
            self.fc0.add_module('fc0_s1',nn.Linear(25088, 4096, bias=True))
            self.fc0.add_module('relu0_s1',nn.ReLU(inplace=True))
            self.fc0.add_module('drop0_s1',nn.Dropout(p=0.5, inplace=False))

            self.fc1 = nn.Sequential()
            self.fc1.add_module('fc1_s1',nn.Linear(9*4096, 4096, bias=True))
            self.fc1.add_module('relu1_s1',nn.ReLU(inplace=True))
            self.fc1.add_module('drop1_s1',nn.Dropout(p=0.5, inplace=False))

            self.classifier = nn.Sequential()
            self.classifier.add_module('fc8',nn.Linear(4096, 1000))

        if model_type=="resnet50":
            self.model = models.resnet50(pretrained=False)
            num_ftrs = self.model.fc.in_features
            self.model.fc=torch.nn.Linear(num_ftrs, 1000)


    def save(self,checkpoint):
        torch.save(self.state_dict(), checkpoint)
        
   
    def forward(self, x):
            z = self.model.features(x)
            z = self.model.avgpool(z) # here go through all the network except fc layer
            z = self.fc0(z)
            z = self.fc1(z)
            z = self.classifier(z)

            return z

       
