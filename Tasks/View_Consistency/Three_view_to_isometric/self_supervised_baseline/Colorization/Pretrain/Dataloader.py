from __future__ import print_function, division

import torch
import glob
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import json


class ThreeV2I_data(Dataset):
    def __init__(self, root_dir):
        self.dic = sorted(os.listdir(root_dir))
        self.root_dir = root_dir

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        input_dic = os.path.join(self.root_dir,
                                 self.dic[idx])
        original_img = cv2.imread(input_dic)
        gray_img=(np.moveaxis(cv2.cvtColor(original_img,cv2.COLOR_RGB2GRAY),-1,0)/255).reshape(1,200,200)
        original_img=np.moveaxis(original_img, -1, 0)/255
        return gray_img, original_img

    def __len__(self):
        return len(self.dic)


