from __future__ import print_function, division

import torch
import glob
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import json


class ThreeV2I_BC_data(Dataset):
    def __init__(self, root_dir):
        self.dic=sorted(os.listdir(root_dir))
        self.dic.remove('answer.json')
       
        self.root_dir=root_dir
        with open(os.path.join(self.root_dir, 'answer.json'), 'r') as f:
            self.answer = json.load(f)
    def __getitem__(self, idx):   
        if torch.is_tensor(idx):
            idx = idx.tolist()        
        input_dic = os.path.join(self.root_dir,
                                self.dic[idx])
       
        answer = self.answer[self.dic[idx]]
  
        Front_img=self.convert_input(input_dic,"/*f.png")
        Right_img=self.convert_input(input_dic,"/*r.png")
        Top_img=self.convert_input(input_dic,"/*t.png")
        Front_img=np.moveaxis(Front_img, -1, 0)
        Right_img = np.moveaxis(Right_img, -1, 0)
        Top_img = np.moveaxis(Top_img, -1, 0)
        Ans_1=self.convert_input(input_dic,"/0.png")
        Ans_2=self.convert_input(input_dic,"/1.png")
        Ans_3=self.convert_input(input_dic,"/2.png")
        Ans_4=self.convert_input(input_dic,"/3.png")
        Ans_1=np.moveaxis(Ans_1, -1, 0)
        Ans_2=np.moveaxis(Ans_2, -1, 0)
        Ans_3 = np.moveaxis(Ans_3, -1, 0)
        Ans_4 = np.moveaxis(Ans_4, -1, 0)
        Label=self.convert_answer(answer)
        
        return Front_img,Right_img,Top_img,Ans_1,Ans_2,Ans_3,Ans_4,Label
    
    def __len__(self):
        return len(self.dic)
    
    def convert_input(self,Dic,name):
        file_name=glob.glob(Dic+name)
        img=cv2.imread(file_name[0])
     
        return img/255
    
    def convert_answer(self,index):
    
        if index==0:
            output=torch.tensor([1,0,0,0])
        if index==1:
            output=torch.tensor([0,1,0,0])
        if index==2:
            output=torch.tensor([0,0,1,0])
        if index==3:
            output=torch.tensor([0,0,0,1])
        return output


class ThreeV2I_ML_data(Dataset):
    def __init__(self, root_dir):
        self.dic=sorted(os.listdir(root_dir))
        self.dic.remove('answer.json')
        
        self.root_dir=root_dir
        with open(os.path.join(self.root_dir, 'answer.json'), 'r') as f:
            self.answer = json.load(f)
    def __getitem__(self, idx):   
        if torch.is_tensor(idx):
            idx = idx.tolist()        
        input_dic = os.path.join(self.root_dir,
                                self.dic[idx])
    
        answer = self.answer[self.dic[idx]]
        
        Front_img=self.convert_input(input_dic,"/*f.png")
        Right_img=self.convert_input(input_dic,"/*r.png")
        Top_img=self.convert_input(input_dic,"/*t.png")
        Ans_1=self.convert_input(input_dic,"/0.png")
        Ans_2=self.convert_input(input_dic,"/1.png")
        Ans_3=self.convert_input(input_dic,"/2.png")
        Ans_4=self.convert_input(input_dic,"/3.png")
        Label=self.convert_answer(answer)
        View=np.moveaxis((np.concatenate((Front_img,Right_img,Top_img),axis=2)),-1,0)
        input_1=np.moveaxis(Ans_1,-1,0)
        input_2=np.moveaxis(Ans_2,-1,0)
        input_3=np.moveaxis(Ans_3,-1,0)
        input_4=np.moveaxis(Ans_4,-1,0)
        
        return View,input_1,input_2,input_3,input_4,Label
    
    def __len__(self):
        return len(self.dic)
    
    def convert_input(self,Dic,name):
        file_name=glob.glob(Dic+name)
        img=cv2.imread(file_name[0])
      
        return img/255
    
    def convert_answer(self,index):
       
        if index==0:
            output=torch.tensor([0])
        if index==1:
            output=torch.tensor([1])
        if index==2:
            output=torch.tensor([2])
        if index==3:
            output=torch.tensor([3])
        return output
