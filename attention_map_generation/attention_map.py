import glob
import random

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models
import bagnets.pytorchnet
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

Tasks = ['Three2I','I2P','P2I']
phases=['train','valid']
model_types=['resnet50','vgg16','Bagnet33']
samples_num=2
overlay=False
root='C:/Users/ay162\Desktop\slack/test' # use your own root path here
outf='C:/Users/ay162\Desktop\slack/test' # use your own outpur path here


class Bagnet(nn.Module):
    def __init__(self, Task=None):
        super(Bagnet, self).__init__()
        self.model = bagnets.pytorchnet.bagnet33(pretrained=False)
        self.model.conv1 = torch.nn.Conv2d(in_channels=12, out_channels=64,
                                           kernel_size=1, stride=1, padding=0, bias=False)
        num_ftrs = self.model.fc.in_features
        if Task=='Three2I':
            self.model.fc = torch.nn.Linear(num_ftrs, 1)
        elif Task=='I2P':
            self.model.fc = torch.nn.Linear(num_ftrs, 4)
        else:
            self.model.fc = torch.nn.Linear(num_ftrs, 8)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.conv2(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        g0 = self.model.layer1(x)
        g1 = self.model.layer2(g0)
        g2 = self.model.layer3(g1)
        g3 = self.model.layer4(g2)
        return [g.pow(2).mean(1) for g in (g0, g1, g2, g3)]

    
class Resnet(nn.Module):
    def __init__(self, Task=None):
        super(Resnet, self).__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.conv1 = torch.nn.Conv2d(in_channels=12, out_channels=64,
                                           kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        num_ftrs = self.model.fc.in_features
        if Task=='Three2I':
            self.model.fc = torch.nn.Linear(num_ftrs, 1)
        elif Task=='I2P':
            self.model.fc = torch.nn.Linear(num_ftrs, 4)
        else:
            self.model.fc = torch.nn.Linear(num_ftrs, 8)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        g0 = self.model.layer1(x)
        g1 = self.model.layer2(g0)
        g2 = self.model.layer3(g1)
        g3 = self.model.layer4(g2)
        return [g.pow(2).mean(1) for g in (g0, g1, g2, g3)]

    
class VGG(nn.Module):
    def __init__(self, Task=None):
        super(VGG, self).__init__()
        self.model = models.vgg16(pretrained=False)
        self.model.features[0] = torch.nn.Conv2d(in_channels=12, out_channels=64,
                                                 kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        num_ftrs = self.model.classifier[6].in_features
        if Task=='Three2I':
            self.model.classifier[6] = torch.nn.Linear(num_ftrs, 1)
        elif Task=='I2P':
            self.model.classifier[6] = torch.nn.Linear(num_ftrs, 4)
        else:
            self.model = models.vgg16_bn(pretrained=False)
            self.model.features[0] = torch.nn.Conv2d(in_channels=12, out_channels=64,
                                                     kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.model.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True), nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True), nn.Dropout(),
                nn.Linear(4096, 8),
            )
            
    def forward(self, x):
        x=self.model.features[0](x)
        x = self.model.features[1](x)
        x = self.model.features[2](x)
        x = self.model.features[3](x)
        g0 = self.model.features[4](x)
        x=self.model.features[5](g0)
        x = self.model.features[6](x)
        x = self.model.features[7](x)
        x = self.model.features[8](x)
        g1 = self.model.features[9](x)
        x=self.model.features[10](g1)
        x = self.model.features[11](x)
        x = self.model.features[12](x)
        x = self.model.features[13](x)
        x = self.model.features[14](x)
        x = self.model.features[15](x)
        g2 = self.model.features[16](x)
        x=self.model.features[17](g2)
        x = self.model.features[18](x)
        x = self.model.features[19](x)
        x = self.model.features[20](x)
        x = self.model.features[21](x)
        x = self.model.features[22](x)
        g3 = self.model.features[23](x)
        x=self.model.features[24](g3)
        x = self.model.features[25](x)
        x = self.model.features[26](x)
        x = self.model.features[27](x)
        x = self.model.features[28](x)
        x = self.model.features[29](x)
        g4 = self.model.features[30](x)
        return [g.pow(2).mean(1) for g in (g0, g1, g2, g3, g4)]

def convert_input(Dic, name):
    file_name = glob.glob(Dic + name)
    img = cv2.imread(file_name[0])

    return img / 255

for Task in Tasks:
    for phase in phases:
        if Task=='I2P':
            input_dir = os.path.join('D:\spare3d_plus\Final_data\I2P_data\Line_data',phase)
        elif Task=='Three2I':
            input_dir = os.path.join('D:\spare3d_plus\Three2I\Three2I_Line_Test',phase)
        else:
            input_dir = os.path.join('D:\spare3d_plus\Final_data\P2I_data\Line_data',phase)

        dics=sorted(os.listdir(input_dir))
        dics.remove('answer.json')
        shapes=random.sample(dics,samples_num)
        for shape in shapes:
            if Task=='I2P':
                input_dic=os.path.join(input_dir,shape)
                Front_img = convert_input(input_dic, "/*_f.png")
                Right_img = convert_input(input_dic, "/*_r.png")
                Top_img = convert_input(input_dic, "/*_t.png")
                Ans=convert_input(input_dic, "/answer.png")
                View = np.concatenate((Front_img, Right_img, Top_img), axis=2)
                inputs = [torch.tensor(np.moveaxis(np.concatenate((Ans, View), axis=2), -1, 0)).reshape((1, 12, 200, 200))]
            else:
                input_dic=os.path.join(input_dir,shape)
                Front_img = convert_input(input_dic, "/*f.png")
                Right_img = convert_input(input_dic, "/*r.png")
                Top_img = convert_input(input_dic, "/*t.png")
                Ans=[convert_input(input_dic, "/%d.png"%i) for i in range(4)]
                View = np.concatenate((Front_img, Right_img, Top_img), axis=2)
                inputs=[torch.tensor(np.moveaxis(np.concatenate((Ans[i], View), axis=2), -1, 0)).reshape((1, 12, 200, 200)) for i in range(4)]

            model_paths=[glob.glob(root+'/'+Task+'_model'+'/'+m+'*.pth')[0] for m in model_types]
            types=[Resnet(Task),VGG(Task),Bagnet(Task)]
            out_dir=os.path.join(outf,'Attention_Map')
            out_dir=os.path.join(out_dir,Task)
            out_dir = os.path.join(out_dir, phase)
            out_dir = os.path.join(out_dir, shape)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            for i,type in enumerate(model_types):
                model=types[i]
                model.load_state_dict(torch.load(model_paths[i], map_location='cpu'))
                model.eval()
                img = []
                for j, input in enumerate(inputs):
                    list = []
                    with torch.no_grad():
                        gs = model(input.float())
                    if not overlay:
                        if Task=='I2P':
                            original_img = cv2.imread(input_dic + '/answer.png')
                        else:
                            original_img=cv2.imread(input_dic+'/%d.png'%j)
                        list.append(original_img)
                        x,y,_=original_img.shape
                        for k, g in enumerate(gs):
                            plt.imsave(os.path.join(out_dir,'%d.png'%k),g[0])
                            im=cv2.imread(os.path.join(out_dir,'%d.png'%k))
                            os.remove(os.path.join(out_dir,'%d.png'%k))
                            list.append(cv2.resize(im, (x, y)))
                        img.append(np.concatenate(([t for t in list]),axis=0))
                    else:
                        if Task=='I2P':
                            ans = cv2.imread(input_dic + '/answer.png')
                            imf=cv2.imread(glob.glob(input_dic + '/*_f.png')[0])
                            imr = cv2.imread(glob.glob(input_dic + '/*_r.png')[0])
                            imt = cv2.imread(glob.glob(input_dic + '/*_t.png')[0])
                            original_img=np.minimum(ans,np.minimum(imf,np.minimum(imr,imt)))
                        else:
                            ans = cv2.imread(input_dic+'/%d.png'%j)
                            imf=cv2.imread(glob.glob(input_dic + '/*_f.png')[0])
                            imr = cv2.imread(glob.glob(input_dic + '/*_r.png')[0])
                            imt = cv2.imread(glob.glob(input_dic + '/*_t.png')[0])
                            original_img=np.minimum(ans,np.minimum(imf,np.minimum(imr,imt)))
                        list.append(original_img)
                        x,y,_=original_img.shape
                        for k, g in enumerate(gs):
                            plt.imsave(os.path.join(out_dir,'%d.png'%k),g[0])
                            im=cv2.imread(os.path.join(out_dir,'%d.png'%k))
                            os.remove(os.path.join(out_dir,'%d.png'%k))
                            list.append(cv2.addWeighted(cv2.resize(im, (x, y)),0.6,original_img,0.4,0))
                        img.append(np.concatenate(([t for t in list]),axis=0))
                cv2.imwrite(os.path.join(out_dir,'{}_{}_{}.png'.format(Task,type,shape)),np.concatenate(([t for t in img]),axis=1))
