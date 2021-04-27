from __future__ import print_function, division

from torchvision import models
import torch.nn as nn
import torch
import numpy as np
import bagnets.pytorchnet
import math


class Three2I_SimCLR(nn.Module):

    def __init__(self,model_type=None,pretrained=True):
        super(Three2I_SimCLR, self).__init__()
        if model_type=="vgg16":
            self.model_f=models.vgg16(pretrained=pretrained).features
            self.model_r = models.vgg16(pretrained=pretrained).features
            self.model_t = models.vgg16(pretrained=pretrained).features
            self.model_i = models.vgg16(pretrained=pretrained).features
            self.mlp1=nn.Sequential(nn.Linear(512*3*6*6,4096),nn.ReLU(inplace=True),
                                    nn.Linear(4096,256))
            self.mlp2 =nn.Sequential(nn.Linear(512*6*6,4096),nn.ReLU(inplace=True),
                                    nn.Linear(4096,256))
            nn.init.normal_(self.mlp1[0].weight, 0, 0.01)
            nn.init.constant_(self.mlp1[0].bias, 0)
            nn.init.normal_(self.mlp1[2].weight, 0, 0.01)
            nn.init.constant_(self.mlp1[2].bias, 0)
            nn.init.normal_(self.mlp2[0].weight, 0, 0.01)
            nn.init.constant_(self.mlp2[0].bias, 0)
            nn.init.normal_(self.mlp2[2].weight, 0, 0.01)
            nn.init.constant_(self.mlp2[2].bias, 0)

        if model_type=="resnet50":
            pass

    def forward(self, f,r,t,i):
        yf = self.model_f(f)
        yf = torch.flatten(yf, start_dim=1)
        yr = self.model_r(r)
        yr = torch.flatten(yr, start_dim=1)
        yt = self.model_t(t)
        yt = torch.flatten(yt, start_dim=1)
        yi = self.model_i(i)
        yi = torch.flatten(yi, start_dim=1)

        yv = torch.cat((yf, yr, yt), dim=1)

        y_v = self.mlp1(yv)
        y_i = self.mlp2(yi)
        return y_v,y_i
