from __future__ import print_function, division
import torch
import glob
from torch.utils.data import Dataset
import os
import numpy as np
import cv2

class P2I_data(Dataset):
    def __init__(self, root_dir,dataset):
        self.root_dir = root_dir
        self.dataset = dataset

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        dic = list(self.dataset.keys())[idx]
        input_dic = os.path.join(self.root_dir, dic)
        answer=list(self.dataset[dic]['correct'].keys())[0]
        pose=self.dataset[dic]['correct'][answer]
        f = {}
        for q in list(self.dataset[dic].keys()):
            f.update(self.dataset[dic][q])
        Front_img = np.moveaxis(self.convert_input(input_dic, "*_f.png"), -1, 0)
        Right_img = np.moveaxis(self.convert_input(input_dic, "*_r.png"), -1, 0)
        Top_img = np.moveaxis(self.convert_input(input_dic, "*_t.png"), -1, 0)
        Ans_1 = np.moveaxis(self.convert_input(input_dic, f['0']), -1, 0)
        Ans_2 = np.moveaxis(self.convert_input(input_dic, f['1']), -1, 0)
        Ans_3 = np.moveaxis(self.convert_input(input_dic, f['2']), -1, 0)
        Ans_4 = np.moveaxis(self.convert_input(input_dic, f['3']), -1, 0)
        Label = self.convert_answer(answer)
        View_vector = self.view_vector(pose)
        return Front_img, Right_img, Top_img, Ans_1, Ans_2,Ans_3,Ans_4,Label, View_vector

    def __len__(self):
        return len(self.dataset)

    def convert_input(self, Dic, name):
        file_name = glob.glob(Dic +'/'+ name)
        print(file_name)
        img = cv2.imread(file_name[0])
        return img / 255

    def view_vector(self, index):
        i=int(index.replace('.png',''))-1
        output = torch.tensor([False for i in range(20)])
        output[i]=True
        return output

    def convert_answer(self, index):
        if index == '0':
            output = torch.tensor([1,0,0,0])
        if index == '1':
            output = torch.tensor([0,1,0,0])
        if index == '2':
            output = torch.tensor([0,0,1,0])
        if index == '3':
            output = torch.tensor([0,0,0,1])
        return output


class I2P_data(Dataset):
    def __init__(self, root_dir, dataset):
        self.root_dir = root_dir
        self.dataset = dataset

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        dic = list(self.dataset.keys())[idx]
        input_dic = os.path.join(self.root_dir, dic)

        answer = list(self.dataset[dic].keys())[0]

        Front_img = self.convert_input(input_dic, "/*f.png")
        Right_img = self.convert_input(input_dic, "/*r.png")
        Top_img = self.convert_input(input_dic, "/*t.png")
        Front_img = np.moveaxis(Front_img, -1, 0)
        Right_img = np.moveaxis(Right_img, -1, 0)
        Top_img = np.moveaxis(Top_img, -1, 0)
        Ans = self.convert_input(input_dic, self.dataset[dic][answer])
        Ans = np.moveaxis(Ans, -1, 0)
        Label = self.convert_answer(answer)
        return Front_img, Right_img, Top_img, Ans, Label

    def __len__(self):
        return len(self.dataset)

    def convert_input(self, Dic, name):
        file_name = glob.glob(Dic + '/' + name)
        img = cv2.imread(file_name[0])
        return img / 255

    def convert_answer(self, index):
        return torch.tensor([int(index)])