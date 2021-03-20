from __future__ import print_function, division

import torch.nn as nn
import torch


cfgs = {
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):

    def __init__(self, features, width, num_classes=4, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(int(width/8*512)* 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False,width=8):
    layers = []
    in_channels = 12
    n=width/8
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, int(v*n), kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(int(v*n)), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = int(v*n)
    return nn.Sequential(*layers)

def vgg(width=8,cfg='B',batch_norm=False):
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm,width=width),width)
    return model

class I2P(nn.Module):

    def __init__(self, width=8,depth='B',batch_norm=False):
        super(I2P, self).__init__()
        if depth=='vgg13':
            self.depth='B'
        elif depth=='vgg16':
            self.depth='D'
        elif depth=='vgg19':
            self.depth='E'
        if width==384:
            self.width=6
        elif width==448:
            self.width=7
        elif width==512:
            self.width=8
        elif width==1024:
            self.width=16
        self.model = vgg(self.width,self.depth,batch_norm)
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = torch.nn.Linear(num_ftrs, 4)

    def forward(self, x):
        y = self.model(x)
        return y