from __future__ import print_function, division

from torchvision import models
import torch.nn as nn
import torch


models.resnet50()


class P2I(nn.Module):

    def __init__(self,model_type=None,pretrained=False):
        super(P2I, self).__init__()
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
                nn.Linear(4096, 8)
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
            self.classifier = nn.Linear(2048, 8)
            nn.init.normal_(self.classifier.weight, 0, 0.01)
            nn.init.constant_(self.classifier.bias, 0)
            nn.init.normal_(self.fcv[0].weight, 0, 0.01)
            nn.init.constant_(self.fcv[0].bias, 0)
            nn.init.normal_(self.fca[0].weight, 0, 0.01)
            nn.init.constant_(self.fca[0].bias, 0)

    def forward(self, xf,xr,xt,xa1,xa2,xa3,xa4):
        yf = self.model_f(xf)
        yf = torch.flatten(yf, start_dim=1)
        yr = self.model_r(xr)
        yr = torch.flatten(yr, start_dim=1)
        yt = self.model_t(xt)
        yt = torch.flatten(yt, start_dim=1)
        ya1 = self.model_a(xa1)
        ya1 = torch.flatten(ya1, start_dim=1)
        ya2 = self.model_a(xa2)
        ya2 = torch.flatten(ya2, start_dim=1)
        ya3 = self.model_a(xa3)
        ya3 = torch.flatten(ya3, start_dim=1)
        ya4 = self.model_a(xa4)
        ya4 = torch.flatten(ya4, start_dim=1)
        yv = torch.cat((yf, yr, yt), dim=1)
        yv = self.fcv(yv)
        ya1 = self.fca(ya1)
        ya2 = self.fca(ya2)
        ya3 = self.fca(ya3)
        ya4 = self.fca(ya4)
        y1 = torch.cat((yv, ya1), dim=1)
        y1 = self.classifier(y1)
        y2 = torch.cat((yv, ya2), dim=1)
        y2 = self.classifier(y2)
        y3 = torch.cat((yv, ya3), dim=1)
        y3 = self.classifier(y3)
        y4 = torch.cat((yv, ya4), dim=1)
        y4 = self.classifier(y4)
        return y1,y2,y3,y4
