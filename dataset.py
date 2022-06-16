import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision.io import read_image

class MyDataSet(Dataset):
    train_image_path = os.path.join('CamVid','train')
    train_label_path = os.path.join('CamVid','train_labels')
    train_image_name = []
    train_label_name = []
    for filename in os.listdir(train_image_path):
            if filename.endswith('png'):
                train_image_name.append(os.path.join(train_image_path,filename))
                train_label_name.append(os.path.join(train_label_path,filename.replace('.png','_L.png')))
                
    val_image_path = os.path.join('CamVid','train')
    val_label_path = os.path.join('CamVid','train_labels')
    val_image_name = []
    val_label_name = []
    for filename in os.listdir(val_image_path):
            if filename.endswith('png'):
                val_image_name.append(os.path.join(val_image_path,filename))
                val_label_name.append(os.path.join(val_label_path,filename.replace('.png','_L.png')))
                
                
    test_image_path = os.path.join('CamVid','train')
    test_label_path = os.path.join('CamVid','train_labels')
    test_image_name = []
    test_label_name = []
    for filename in os.listdir(test_image_path):
            if filename.endswith('png'):
                test_image_name.append(os.path.join(test_image_path,filename))
                test_label_name.append(os.path.join(test_label_path,filename.replace('.png','_L.png')))                    
    def __init__(self,data_type,transform = None):
        self.data_type = data_type
        self.transform = transform
    def __getitem__(self,idx):
        if self.data_type == 'train':
            return read_image(self.train_image_name[idx]),read_image(self.train_label_name[idx])
        elif self.data_type == 'val':
            return read_image(self.val_image_name[idx]),read_image(self.val_label_name[idx])
        else:
            return read_image(self.test_image_name[idx]),read_image(self.test_label_name[idx])
    def __len__(self):
        if self.data_type == 'train':
            return len(self.train_image_name)
        elif self.data_type == 'val':
            return len(self.val_image_name)
        else:
            return len(self.test_image_name)