#import necessary library
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data.dataloader import Dataset, DataLoader

import cv2
import os
import numpy as np
import glob
import random
import matplotlib.pyplot as plt

#out_channel numbers
n1, n2, n3 = 128, 64, 3

#filters(kernels) size
f1, f2, f3 = 9, 3, 5

upscale_factor = 3

input_size = 33
output_size = input_size - f1 - f2 - f3 + 3

stride = 14

#train hyperparam
batch_size = 128
epochs = 200

path = "C:/Users/User/Pictures/Image_dt/T91"

#use cpu for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset) :
    def __init__(self, img_paths, input_size, output_size, stride = 14, upscale_factor = 3) :
        super(CustomDataset, self).__init__()

        self.img_paths = glob.glob(img_paths + '/' + '*.png')
        self.stride = stride
        self.upscale_factor = upscale_factor
        self.sub_lr_imgs = []
        self.sub_hr_imgs = []
        self.input_size = input_size
        self.output_size = output_size
        self.pad = abs(self.input_size - self.output_size) // 2

        print("Start {} Images Pre-Processing".format(len(self.img_paths)))
        for img_path in self.img_paths :
            img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        
        #mod crop
        h = img.shape[0] - np.mod(img.shape[0], self.upscale_factor)
        w = img.shape[1] - np.mod(img.shape[0], self)

print(torch.__version__)
