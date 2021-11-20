from __future__ import print_function, division

from torchvision import models
import torch.nn as nn
import torch
import numpy as np
#import bagnets.pytorchnet
import math
import glob


class Three2I_opt2(nn.Module):

    def __init__(self,model=None,pretrained=False):
        super(Three2I_opt2, self).__init__()
        self.model=model
        if model=='vgg16':
            self.model_f=models.vgg16(pretrained=pretrained).features
            self.model_r = models.vgg16(pretrained=pretrained).features
            self.model_t = models.vgg16(pretrained=pretrained).features
            self.model_i = models.vgg16(pretrained=pretrained).features
            self.mlp1=nn.Sequential(nn.Linear(512*3*6*6,4096),nn.ReLU(inplace=True))
            self.mlp2 =nn.Sequential(nn.Linear(512*6*6,4096),nn.ReLU(inplace=True))
            self.mlp3 = nn.Sequential(nn.Linear(2*4096,4096),nn.ReLU(inplace=True),
                                      nn.Linear(4096,1))
            nn.init.normal_(self.mlp1[0].weight, 0, 0.01)
            nn.init.constant_(self.mlp1[0].bias, 0)
            nn.init.normal_(self.mlp2[0].weight, 0, 0.01)
            nn.init.constant_(self.mlp2[0].bias, 0)
            nn.init.normal_(self.mlp3[0].weight, 0, 0.01)
            nn.init.constant_(self.mlp3[0].bias, 0)
            nn.init.normal_(self.mlp3[2].weight, 0, 0.01)
            nn.init.constant_(self.mlp3[2].bias, 0)
        if model=='resnet50':
            self.model_f=nn.Sequential(models.resnet50(pretrained=pretrained).conv1,
                                   models.resnet50(pretrained=pretrained).bn1,
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                   models.resnet50(pretrained=pretrained).layer1,
                                   models.resnet50(pretrained=pretrained).layer2,
                                   models.resnet50(pretrained=pretrained).layer3,
                                   models.resnet50(pretrained=pretrained).layer4,
                               nn.AdaptiveAvgPool2d((1, 1)))
            self.model_r = nn.Sequential(models.resnet50(pretrained=pretrained).conv1,
                                         models.resnet50(pretrained=pretrained).bn1,
                                         nn.ReLU(inplace=True),
                                         nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                         models.resnet50(pretrained=pretrained).layer1,
                                         models.resnet50(pretrained=pretrained).layer2,
                                         models.resnet50(pretrained=pretrained).layer3,
                                         models.resnet50(pretrained=pretrained).layer4,
                                         nn.AdaptiveAvgPool2d((1, 1)))
            self.model_t = nn.Sequential(models.resnet50(pretrained=pretrained).conv1,
                                         models.resnet50(pretrained=pretrained).bn1,
                                         nn.ReLU(inplace=True),
                                         nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                         models.resnet50(pretrained=pretrained).layer1,
                                         models.resnet50(pretrained=pretrained).layer2,
                                         models.resnet50(pretrained=pretrained).layer3,
                                         models.resnet50(pretrained=pretrained).layer4,
                                         nn.AdaptiveAvgPool2d((1, 1)))
            self.model_i = nn.Sequential(models.resnet50(pretrained=pretrained).conv1,
                                         models.resnet50(pretrained=pretrained).bn1,
                                         nn.ReLU(inplace=True),
                                         nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                         models.resnet50(pretrained=pretrained).layer1,
                                         models.resnet50(pretrained=pretrained).layer2,
                                         models.resnet50(pretrained=pretrained).layer3,
                                         models.resnet50(pretrained=pretrained).layer4,
                                         nn.AdaptiveAvgPool2d((1, 1)))
            self.mlp1 = nn.Sequential(nn.Linear(512 * 3 * 4, 1024), nn.ReLU(inplace=True))
            self.mlp2 = nn.Sequential(nn.Linear(512 * 4, 1024), nn.ReLU(inplace=True))
            self.mlp3 = nn.Linear(2*1024, 1)
            nn.init.normal_(self.mlp3.weight, 0, 0.01)
            nn.init.constant_(self.mlp3.bias, 0)
        nn.init.normal_(self.mlp1[0].weight, 0, 0.01)
        nn.init.constant_(self.mlp1[0].bias, 0)
        nn.init.normal_(self.mlp2[0].weight, 0, 0.01)
        nn.init.constant_(self.mlp2[0].bias, 0)

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
