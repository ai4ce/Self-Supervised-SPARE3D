from __future__ import print_function, division

from torchvision import models
import torch.nn as nn
import torch
import numpy as np
import bagnets.pytorchnet
from Jig_pre_model import Jig



class ThreeV2I_BC(nn.Module):

    def __init__(self,model_type=None, saved_model=None):
        super(ThreeV2I_BC, self).__init__()

        if model_type=="vgg16":
            self.model_J = Jig("vgg16")
            self.model_J.load_state_dict(torch.load(saved_model))
            self.model_J.model.features[0]=torch.nn.Conv2d(in_channels=12, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))
            num_ftrs = self.model_J.model.classifier[6].in_features
            self.model_J.model.classifier[6] = torch.nn.Linear(num_ftrs, 1)
            self.model_J.fc1.fc1_s1 = nn.Linear(4096, 4096, bias=True)
            self.model_J.classifier.fc8 = nn.Linear(4096, 1)

        if model_type=="resnet50":
            self.model = models.resnet50(pretrained=False)
            self.model.conv1=torch.nn.Conv2d(in_channels=12, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))
            num_ftrs = self.model.fc.in_features
            self.model.fc=torch.nn.Linear(num_ftrs, 1)
  

    def forward(self, x):
        y = self.model_J.model(x)
        return y 


class ThreeV2I_ML(nn.Module):

    def __init__(self,model_type=None):
        super(ThreeV2I_ML, self).__init__()

        if model_type=="vgg11":
            self.model_3V = models.vgg11(pretrained=False)
            self.model_3V.features[0]=torch.nn.Conv2d(in_channels=9, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))
            num_ftrs = self.model_3V.classifier[6].in_features
            self.model_3V.classifier[6]=torch.nn.Linear(num_ftrs, 128)
            ### model for extracting orthogonal view features
            self.model_ortho = models.vgg11(pretrained=False)
            self.model_ortho.features[0]=torch.nn.Conv2d(in_channels=3, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))
            self.model_ortho.classifier[6]=torch.nn.Linear(num_ftrs, 128)
        if model_type=="vgg13":
            self.model_3V = models.vgg13(pretrained=False)
            self.model_3V.features[0]=torch.nn.Conv2d(in_channels=9, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))
            num_ftrs = self.model_3V.classifier[6].in_features
            self.model_3V.classifier[6]=torch.nn.Linear(num_ftrs, 128)
            ### model for extracting orthogonal view features
            self.model_ortho = models.vgg13(pretrained=False)
            self.model_ortho.features[0]=torch.nn.Conv2d(in_channels=3, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))
            self.model_ortho.classifier[6]=torch.nn.Linear(num_ftrs, 128)
        if model_type=="vgg16":
            self.model_3V = models.vgg16(pretrained=False)
            self.model_3V.features[0]=torch.nn.Conv2d(in_channels=9, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))
            num_ftrs = self.model_3V.classifier[6].in_features
            self.model_3V.classifier[6]=torch.nn.Linear(num_ftrs, 128)
            ### model for extracting orthogonal view features
            self.model_ortho = models.vgg16(pretrained=False)
            self.model_ortho.features[0]=torch.nn.Conv2d(in_channels=3, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))
            self.model_ortho.classifier[6]=torch.nn.Linear(num_ftrs, 128)
        if model_type=="vgg19":
            self.model_3V = models.vgg19(pretrained=False)
            self.model_3V.features[0]=torch.nn.Conv2d(in_channels=9, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))
            num_ftrs = self.model_3V.classifier[6].in_features
            self.model_3V.classifier[6]=torch.nn.Linear(num_ftrs, 128)
            ### model for extracting orthogonal view features
            self.model_ortho = models.vgg19(pretrained=False)
            self.model_ortho.features[0]=torch.nn.Conv2d(in_channels=3, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))
            self.model_ortho.classifier[6]=torch.nn.Linear(num_ftrs, 128)

        if model_type=="resnet18":
            self.model_3V = models.resnet18(pretrained=False)
            self.model_3V.conv1=torch.nn.Conv2d(in_channels=9, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))
            num_ftrs = self.model_3V.fc.in_features
            self.model_3V.fc=torch.nn.Linear(num_ftrs, 128)

            self.model_ortho = models.resnet18(pretrained=False)
            self.model_ortho.conv1=torch.nn.Conv2d(in_channels=3, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))     
            self.model_ortho.fc=torch.nn.Linear(num_ftrs, 128)
        if model_type=="resnet34":
            self.model_3V = models.resnet34(pretrained=False)
            self.model_3V.conv1=torch.nn.Conv2d(in_channels=9, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))
            num_ftrs = self.model_3V.fc.in_features
            self.model_3V.fc=torch.nn.Linear(num_ftrs, 128)

            self.model_ortho = models.resnet34(pretrained=False)
            self.model_ortho.conv1=torch.nn.Conv2d(in_channels=3, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))     
            self.model_ortho.fc=torch.nn.Linear(num_ftrs, 128)
        if model_type=="resnet50":
            self.model_3V = models.resnet50(pretrained=False)
            self.model_3V.conv1=torch.nn.Conv2d(in_channels=9, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))
            num_ftrs = self.model_3V.fc.in_features
            self.model_3V.fc=torch.nn.Linear(num_ftrs, 128)

            self.model_ortho = models.resnet50(pretrained=False)
            self.model_ortho.conv1=torch.nn.Conv2d(in_channels=3, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))     
            self.model_ortho.fc=torch.nn.Linear(num_ftrs, 128)
        if model_type=="resnet101":
            self.model_3V = models.resnet101(pretrained=False)
            self.model_3V.conv1=torch.nn.Conv2d(in_channels=9, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))
            num_ftrs = self.model_3V.fc.in_features
            self.model_3V.fc=torch.nn.Linear(num_ftrs, 128)

            self.model_ortho = models.resnet101(pretrained=False)
            self.model_ortho.conv1=torch.nn.Conv2d(in_channels=3, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))     
            self.model_ortho.fc=torch.nn.Linear(num_ftrs, 128)

    def forward(self, x_3V,x_ortho):
        feature_3v = self.model_3V(x_3V)
        feature_ortho = self.model_ortho(x_ortho)
        distance=nn.functional.pairwise_distance(feature_3v,feature_ortho,2,keepdim=True)
        return distance
