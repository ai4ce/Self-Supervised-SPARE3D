from __future__ import print_function, division

from torchvision import models
import torch.nn as nn
import torch
#import bagnets.pytorchnet

class I2P(nn.Module):

    def __init__(self,model_type=None,pretrained=False):
        super(I2P, self).__init__()
        if model_type=='vgg16':
            self.model_f = models.vgg16(pretrained=pretrained).features
            self.model_r = models.vgg16(pretrained=pretrained).features
            self.model_t = models.vgg16(pretrained=pretrained).features
            self.model_a = models.vgg16(pretrained=pretrained).features
            self.fcv = nn.Sequential(nn.Linear(3 * 512 * 6 * 6, 4096),nn.ReLU(inplace=True))
            self.fca = nn.Sequential(nn.Linear(512 * 6 * 6, 4096),nn.ReLU(inplace=True))
            self.classifier = nn.Sequential(
                nn.Linear(2*4096, 4096),
                nn.ReLU(inplace=True), nn.Dropout(),
                nn.Linear(4096, 4)
            )
            nn.init.normal_(self.fcv[0].weight, 0, 0.01)
            nn.init.constant_(self.fcv[0].bias, 0)
            nn.init.normal_(self.fca[0].weight, 0, 0.01)
            nn.init.constant_(self.fca[0].bias, 0)
            nn.init.normal_(self.classifier[0].weight, 0, 0.01)
            nn.init.constant_(self.classifier[0].bias, 0)
            nn.init.normal_(self.classifier[3].weight, 0, 0.01)
            nn.init.constant_(self.classifier[3].bias, 0)

        if model_type=='resnet50':
            self.model_f = nn.Sequential(models.resnet50(pretrained=pretrained).conv1,
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
            self.model_a = nn.Sequential(models.resnet50(pretrained=pretrained).conv1,
                                         models.resnet50(pretrained=pretrained).bn1,
                                         nn.ReLU(inplace=True),
                                         nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                         models.resnet50(pretrained=pretrained).layer1,
                                         models.resnet50(pretrained=pretrained).layer2,
                                         models.resnet50(pretrained=pretrained).layer3,
                                         models.resnet50(pretrained=pretrained).layer4,
                                         nn.AdaptiveAvgPool2d((1, 1)))
            self.fcv = nn.Sequential(nn.Linear(512 * 3 * 4, 1024), nn.ReLU(inplace=True))
            self.fca = nn.Sequential(nn.Linear(512 * 4, 1024), nn.ReLU(inplace=True))
            self.classifier = nn.Linear(2048, 4)
            nn.init.normal_(self.classifier.weight, 0, 0.01)
            nn.init.constant_(self.classifier.bias, 0)
            nn.init.normal_(self.fcv[0].weight, 0, 0.01)
            nn.init.constant_(self.fcv[0].bias, 0)
            nn.init.normal_(self.fca[0].weight, 0, 0.01)
            nn.init.constant_(self.fca[0].bias, 0)

    def forward(self, xf,xr,xt,xa):
        yf = self.model_f(xf)
        yf = torch.flatten(yf, start_dim=1)
        yr = self.model_r(xr)
        yr = torch.flatten(yr, start_dim=1)
        yt = self.model_t(xt)
        yt = torch.flatten(yt, start_dim=1)
        ya = self.model_a(xa)
        ya = torch.flatten(ya, start_dim=1)
        yv = torch.cat((yf, yr, yt), dim=1)
        yv = self.fcv(yv)
        ya = self.fca(ya)
        y = torch.cat((yv, ya), dim=1)
        y = self.classifier(y)
        return y
