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
            self.mlp1=nn.Sequential(nn.Linear(512*3*6*6,4096),nn.ReLU(inplace=True))
            self.mlp2 =nn.Sequential(nn.Linear(512*6*6,4096),nn.ReLU(inplace=True))
            self.mlp3 = nn.Sequential(nn.Linear(2 * 4096, 4096), nn.ReLU(inplace=True),
                                      nn.Linear(4096, 1))
            nn.init.normal_(self.mlp1[0].weight, 0, 0.01)
            nn.init.constant_(self.mlp1[0].bias, 0)
            nn.init.normal_(self.mlp2[0].weight, 0, 0.01)
            nn.init.constant_(self.mlp2[0].bias, 0)
            nn.init.normal_(self.mlp3[0].weight, 0, 0.01)
            nn.init.constant_(self.mlp3[0].bias, 0)
            nn.init.normal_(self.mlp3[2].weight, 0, 0.01)
            nn.init.constant_(self.mlp3[2].bias, 0)

        if model_type=="resnet50":
            pass

    def forward(self, xf,xr,xt,xa1,xa2,xa3,xa4):
        yf=self.model_f(xf)
        yf= torch.flatten(yf, start_dim=1)
        yr = self.model_r(xr)
        yr = torch.flatten(yr, start_dim=1)
        yt = self.model_t(xt)
        yt = torch.flatten(yt, start_dim=1)
        ya1 = self.model_i(xa1)
        ya1 = torch.flatten(ya1, start_dim=1)
        ya2 = self.model_i(xa2)
        ya2 = torch.flatten(ya2, start_dim=1)
        ya3 = self.model_i(xa3)
        ya3 = torch.flatten(ya3, start_dim=1)
        ya4 = self.model_i(xa4)
        ya4 = torch.flatten(ya4, start_dim=1)

        yv = torch.cat((yf, yr, yt), dim=1)

        yv = self.mlp1(yv)
        ya1 = self.mlp2(ya1)
        ya2 = self.mlp2(ya2)
        ya3 = self.mlp2(ya3)
        ya4 = self.mlp2(ya4)

        y1=self.mlp3(torch.cat((yv, ya1), dim=1))
        y2 = self.mlp3(torch.cat((yv, ya2), dim=1))
        y3 = self.mlp3(torch.cat((yv, ya3), dim=1))
        y4 = self.mlp3(torch.cat((yv, ya4), dim=1))
        return y1,y2,y3,y4

