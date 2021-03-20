from __future__ import print_function, division

import torch
import glob
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import json

class Three2I_SimCLR_data(Dataset):
    def __init__(self, root_dir):
        self.dic=sorted(os.listdir(root_dir))
        self.root_dir=root_dir

    def __getitem__(self, idx):   
        if torch.is_tensor(idx):
            idx = idx.tolist()        
        input_dic = os.path.join(self.root_dir,
                                self.dic[idx])
  
        Front_imgl=self.convert_input(input_dic,"/*_fleft.png")
        Right_imgl=self.convert_input(input_dic,"/*_rleft.png")
        Top_imgl=self.convert_input(input_dic,"/*_tleft.png")
        Isometricl=self.convert_input(input_dic,"/isometric_left.png")

        return Front_imgl,Right_imgl,Top_imgl,Isometricl
    def __len__(self):
        return len(self.dic)
    def convert_input(self,Dic,name):
        file_name=glob.glob(Dic+name)
        img=cv2.imread(file_name[0])
        img=np.moveaxis(img,-1,0)
        return img/255

