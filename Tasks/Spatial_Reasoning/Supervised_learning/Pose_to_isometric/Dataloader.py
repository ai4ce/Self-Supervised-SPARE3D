from __future__ import print_function, division
import torch
import glob
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import json


class P2I_data(Dataset):
    def __init__(self, root_dir):
        self.dic = sorted(os.listdir(root_dir))
        self.dic.remove('answer.json')
        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, 'answer.json'), 'r') as f:
            self.answer = json.load(f)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        input_dic = os.path.join(self.root_dir,self.dic[idx])
        answer = self.answer[self.dic[idx]]
        Front_img = np.moveaxis(self.convert_input(input_dic, "/*_f.png"), -1, 0)
        Right_img = np.moveaxis(self.convert_input(input_dic, "/*_r.png"), -1, 0)
        Top_img = np.moveaxis(self.convert_input(input_dic, "/*_t.png"), -1, 0)
        Ans_1 = np.moveaxis(self.convert_input(input_dic, "/0.png"), -1, 0)
        Ans_2 = np.moveaxis(self.convert_input(input_dic, "/1.png"), -1, 0)
        Ans_3 = np.moveaxis(self.convert_input(input_dic, "/2.png"), -1, 0)
        Ans_4 = np.moveaxis(self.convert_input(input_dic, "/3.png"), -1, 0)
        Label = self.convert_answer(answer)
        View_vector = self.view_vector(input_dic)
        return Front_img, Right_img, Top_img, Ans_1, Ans_2,Ans_3,Ans_4,Label, View_vector

    def __len__(self):
        return len(self.dic)

    def convert_input(self, Dic, name):
        file_name = glob.glob(Dic + name)
        img = cv2.imread(file_name[0])
        return img / 255

    def view_vector(self, Dic):
        a = glob.glob(Dic + "/*.txt")
        base = os.path.basename(a[0])
        index = base.replace(".txt", "")
        if index == "pose_1":
            output = torch.tensor([True,False,False,False,False,False,False,False])
        if index == "pose_2":
            output = torch.tensor([False,True,False,False,False,False,False,False])
        if index == "pose_3":
            output = torch.tensor([False,False,True,False,False,False,False,False])
        if index == "pose_4":
            output = torch.tensor([False,False,False,True,False,False,False,False])
        if index == "pose_5":
            output = torch.tensor([False,False,False,False,True,False,False,False])
        if index == "pose_6":
            output = torch.tensor([False,False,False,False,False,True,False,False])
        if index == "pose_7":
            output = torch.tensor([False,False,False,False,False,False,True,False])
        if index == "pose_8":
            output = torch.tensor([False,False,False,False,False,False,False,True])
        return output

    def convert_answer(self, index):
        if index == 0:
            output = torch.tensor([1,0,0,0])
        if index == 1:
            output = torch.tensor([0,1,0,0])
        if index == 2:
            output = torch.tensor([0,0,1,0])
        if index == 3:
            output = torch.tensor([0,0,0,1])
        return output
