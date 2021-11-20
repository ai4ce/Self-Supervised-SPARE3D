from __future__ import print_function, division

from torchvision import models
import torch.nn as nn
import torch

class SR(nn.Module):
    def __init__(self,pretrained=False):
        super(SR, self).__init__()
        self.model_f = models.vgg16(pretrained=pretrained).features
        self.model_r = models.vgg16(pretrained=pretrained).features
        self.model_t = models.vgg16(pretrained=pretrained).features
        self.model_a = models.vgg16(pretrained=pretrained).features
        self.fcv = nn.Sequential(nn.Linear(3 * 512 * 6 * 6, 4096), nn.ReLU(inplace=True))
        self.fca = nn.Sequential(nn.Linear(512 * 6 * 6, 4096), nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(
            nn.Linear(2 * 4096, 4096),
            nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(4096, 20)
        )
        nn.init.normal_(self.fcv[0].weight, 0, 0.01)
        nn.init.constant_(self.fcv[0].bias, 0)
        nn.init.normal_(self.fca[0].weight, 0, 0.01)
        nn.init.constant_(self.fca[0].bias, 0)
        nn.init.normal_(self.classifier[0].weight, 0, 0.01)
        nn.init.constant_(self.classifier[0].bias, 0)
        nn.init.normal_(self.classifier[3].weight, 0, 0.01)
        nn.init.constant_(self.classifier[3].bias, 0)

    def forward(self, xf,xr,xt,xp):
        yf = self.model_f(xf)
        yf = torch.flatten(yf, start_dim=1)
        yr = self.model_r(xr)
        yr = torch.flatten(yr, start_dim=1)
        yt = self.model_t(xt)
        yt = torch.flatten(yt, start_dim=1)
        yp = self.model_a(xp)
        yp = torch.flatten(yp, start_dim=1)
        yv = torch.cat((yf, yr, yt), dim=1)
        yv = self.fcv(yv)
        yp = self.fca(yp)
        yp = torch.cat((yv, yp), dim=1)
        y = self.classifier(yp)
        return y
