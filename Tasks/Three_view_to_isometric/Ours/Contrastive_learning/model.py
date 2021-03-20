import torchvision.models as models
import torch.nn as nn
import torch

class Three2I_self(nn.Module):
    def __init__(self,model=None,pretrained=False):
        super(Three2I_self, self).__init__()
        self.model=model
        if model=='vgg13':
            self.model_f=models.vgg13(pretrained=pretrained).features
            self.model_r = models.vgg13(pretrained=pretrained).features
            self.model_t = models.vgg13(pretrained=pretrained).features
            self.model_i = models.vgg13(pretrained=pretrained).features
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
        if model=='vgg19':
            self.model_f=models.vgg19(pretrained=pretrained).features
            self.model_r = models.vgg19(pretrained=pretrained).features
            self.model_t = models.vgg19(pretrained=pretrained).features
            self.model_i = models.vgg19(pretrained=pretrained).features
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
            self.mlp1 = nn.Sequential(nn.Linear(512 * 3 * 4, 1024))
            self.mlp2 = nn.Sequential(nn.Linear(512 * 4, 1024))
            self.mlp3 = nn.Linear(2048, 1)
            nn.init.normal_(self.mlp3.weight, 0, 0.01)
            nn.init.constant_(self.mlp3.bias, 0)
        nn.init.normal_(self.mlp1[0].weight, 0, 0.01)
        nn.init.constant_(self.mlp1[0].bias, 0)
        nn.init.normal_(self.mlp2[0].weight, 0, 0.01)
        nn.init.constant_(self.mlp2[0].bias, 0)



    def forward(self,xfl,xrl,xtl,xil,xfr,xrr,xtr,xir,test=True):
        if test:
            yf = self.model_f(xfl)
            yf = torch.flatten(yf, start_dim=1)
            yr = self.model_r(xrl)
            yr = torch.flatten(yr, start_dim=1)
            yt = self.model_t(xtl)
            yt = torch.flatten(yt, start_dim=1)
            ya1 = self.model_i(xil)
            ya1 = torch.flatten(ya1, start_dim=1)
            ya2 = self.model_i(xfr)
            ya2 = torch.flatten(ya2, start_dim=1)
            ya3 = self.model_i(xrr)
            ya3 = torch.flatten(ya3, start_dim=1)
            ya4 = self.model_i(xtr)
            ya4 = torch.flatten(ya4, start_dim=1)

            yv = torch.cat((yf, yr, yt), dim=1)

            yv = self.mlp1(yv)
            ya1 = self.mlp2(ya1)
            ya2 = self.mlp2(ya2)
            ya3 = self.mlp2(ya3)
            ya4 = self.mlp2(ya4)

            y1 = self.mlp3(torch.cat((yv, ya1), dim=1))
            y2 = self.mlp3(torch.cat((yv, ya2), dim=1))
            y3 = self.mlp3(torch.cat((yv, ya3), dim=1))
            y4 = self.mlp3(torch.cat((yv, ya4), dim=1))
        else:
            yfl=self.model_f(xfl)
            yfl= torch.flatten(yfl, start_dim=1)
            yrl = self.model_r(xrl)
            yrl = torch.flatten(yrl, start_dim=1)
            ytl = self.model_t(xtl)
            ytl = torch.flatten(ytl, start_dim=1)
            yil = self.model_i(xil)
            yil = torch.flatten(yil, start_dim=1)
            yfr = self.model_f(xfr)
            yfr = torch.flatten(yfr, start_dim=1)
            yrr = self.model_r(xrr)
            yrr = torch.flatten(yrr, start_dim=1)
            ytr = self.model_t(xtr)
            ytr = torch.flatten(ytr, start_dim=1)
            yir = self.model_i(xir)
            yir = torch.flatten(yir, start_dim=1)

            yvl = torch.cat((yfl, yrl, ytl), dim=1)

            yvl = self.mlp1(yvl)
            yil = self.mlp2(yil)
            yvr = torch.cat((yfr, yrr, ytr), dim=1)
            yvr = self.mlp1(yvr)
            yir = self.mlp2(yir)

            y1=self.mlp3(torch.cat((yvl, yil), dim=1))
            y2 = self.mlp3(torch.cat((yvl, yir), dim=1))
            y3 = self.mlp3(torch.cat((yvr, yil), dim=1))
            y4 = self.mlp3(torch.cat((yvr, yir), dim=1))
        return y1,y2,y3,y4


