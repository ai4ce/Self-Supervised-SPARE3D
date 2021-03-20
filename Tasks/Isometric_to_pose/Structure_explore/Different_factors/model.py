from __future__ import print_function, division

from torchvision import models
import torch.nn as nn
import torch
import numpy as np
#import bagnets.pytorchnet
import math
import glob

class I2P(nn.Module):

    def __init__(self,model_type=None,new_model=False,nd=False,na=False,share_weights=False,pretrained=False,fc='1'):
        super(I2P, self).__init__()
        self.model_type=model_type
        self.new_model = new_model
        self.nd=nd
        self.na=na
        self.share_weights=share_weights
        self.fc=fc
        if model_type=="vgg16":
            if new_model:
                print(2)
                if share_weights:
                    self.model = models.vgg16(pretrained=pretrained).features
                else:
                    self.model_f = models.vgg16(pretrained=pretrained).features
                    self.model_r = models.vgg16(pretrained=pretrained).features
                    self.model_t = models.vgg16(pretrained=pretrained).features
                    self.model_a = models.vgg16(pretrained=pretrained).features
                if not na:
                    self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
                if nd:
                    self.classifier = nn.Sequential(
                        nn.Linear(4 * 512*7*7, 4096),
                        nn.ReLU(inplace=True),
                        nn.Linear(4096, 4096),
                        nn.ReLU(inplace=True),
                        nn.Linear(4096, 4),
                    )
                else:
                    self.classifier = nn.Sequential(
                        nn.Linear(4 * 512 * 7 * 7, 4096),
                        nn.ReLU(inplace=True),nn.Dropout(),
                        nn.Linear(4096, 4096),
                        nn.ReLU(inplace=True),nn.Dropout(),
                        nn.Linear(4096, 4),
                    )
                if fc=='3':
                    self.classifier[0] = nn.Linear(2 * 4096, 4096)
                    if not na:
                        self.fcv = nn.Linear(3 * 512 * 7 * 7, 4096)
                        self.fca = nn.Linear(512 * 7 * 7, 4096)
                    else:
                        self.fcv = nn.Linear(3 * 512 * 6 * 6, 4096)
                        self.fca = nn.Linear(512 * 6 * 6, 4096)
                else:
                    if not na:
                        self.classifier[0]=nn.Linear(4*512*7*7, 4096)
                    else:
                        self.classifier[0] = nn.Linear(4 * 512 * 6 * 6, 4096)
                nn.init.normal_(self.classifier[0].weight, 0, 0.01)
                nn.init.constant_(self.classifier[0].bias, 0)
                if nd:
                    nn.init.normal_(self.classifier[2].weight, 0, 0.01)
                    nn.init.constant_(self.classifier[2].bias, 0)
                    nn.init.normal_(self.classifier[4].weight, 0, 0.01)
                    nn.init.constant_(self.classifier[4].bias, 0)
                else:
                    nn.init.normal_(self.classifier[3].weight, 0, 0.01)
                    nn.init.constant_(self.classifier[3].bias, 0)
                    nn.init.normal_(self.classifier[6].weight, 0, 0.01)
                    nn.init.constant_(self.classifier[6].bias, 0)
            else:
                print(1)
                self.model = models.vgg16(pretrained=pretrained).features
                self.model[0] = torch.nn.Conv2d(in_channels=12, out_channels=64,
                                                kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                nn.init.kaiming_normal_(self.model[0].weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(self.model[0].bias, 0)
                if not na:
                    self.avgpool=nn.AdaptiveAvgPool2d((7, 7))
                if nd:
                    self.classifier = nn.Sequential(
                        nn.Linear(512*7*7, 4096),
                        nn.ReLU(inplace=True),
                        nn.Linear(4096, 4096),
                        nn.ReLU(inplace=True),
                        nn.Linear(4096, 4),
                    )
                else:
                    self.classifier = nn.Sequential(
                        nn.Linear(512 * 7 * 7, 4096),
                        nn.ReLU(inplace=True),nn.Dropout(),
                        nn.Linear(4096, 4096),
                        nn.ReLU(inplace=True),nn.Dropout(),
                        nn.Linear(4096, 4),
                    )
                if not na:
                    self.classifier[0]=nn.Linear(512*7*7, 4096)
                else:
                    self.classifier[0] = nn.Linear(512 * 6 * 6, 4096)
                nn.init.normal_(self.classifier[0].weight, 0, 0.01)
                nn.init.constant_(self.classifier[0].bias, 0)
                if nd:
                    nn.init.normal_(self.classifier[2].weight, 0, 0.01)
                    nn.init.constant_(self.classifier[2].bias, 0)
                    nn.init.normal_(self.classifier[4].weight, 0, 0.01)
                    nn.init.constant_(self.classifier[4].bias, 0)
                else:
                    nn.init.normal_(self.classifier[3].weight, 0, 0.01)
                    nn.init.constant_(self.classifier[3].bias, 0)
                    nn.init.normal_(self.classifier[6].weight, 0, 0.01)
                    nn.init.constant_(self.classifier[6].bias, 0)

    def forward(self, xf,xr,xt,xa):
        if self.new_model:
            if not self.share_weights:
                yf = self.model_f(xf)
                if not self.na:
                    yf=self.avgpool(yf)
                yf = torch.flatten(yf, start_dim=1)
                yr = self.model_r(xr)
                if not self.na:
                    yr = self.avgpool(yr)
                yr = torch.flatten(yr, start_dim=1)
                yt = self.model_t(xt)
                if not self.na:
                    yt = self.avgpool(yt)
                yt = torch.flatten(yt, start_dim=1)
                ya = self.model_a(xa)
                if not self.na:
                    ya = self.avgpool(ya)
                ya = torch.flatten(ya, start_dim=1)
            else:
                yf = self.model(xf)
                if not self.na:
                    yf=self.avgpool(yf)
                yf = torch.flatten(yf, start_dim=1)
                yr = self.model(xr)
                if not self.na:
                    yr = self.avgpool(yr)
                yr = torch.flatten(yr, start_dim=1)
                yt = self.model(xt)
                if not self.na:
                    yt = self.avgpool(yt)
                yt = torch.flatten(yt, start_dim=1)
                ya = self.model(xa)
                if not self.na:
                    ya = self.avgpool(ya)
                ya = torch.flatten(ya, start_dim=1)
            if self.fc=='3':
                yv=torch.cat((yf,yr,yt),dim=1)
                yv=self.fcv(yv)
                ya=self.fca(ya)
                y = torch.cat((yv, ya), dim=1)
                y=self.classifier(y)
            else:
                y = torch.cat((yf, yr, yt,ya), dim=1)
                y = self.classifier(y)
        else:
            x = torch.cat((xf, xr, xt, xa), dim=1)
            if not self.na:
                y = self.model(x)
                y=self.avgpool(y)
                y = torch.flatten(y, 1)
                y = self.classifier(y)
            else:
                y = self.model(x)
                y = torch.flatten(y, 1)
                y = self.classifier(y)
        return y
