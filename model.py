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

class SRCNN(nn.Module) :
    def __init__(self, kernel_list, filters_list, num_channels = 3) :
        super(SRCNN,self).__init__()
        
        f1, f2, f3 = kernel_list
        n1, n2, n3 = filters_list
        
        self.conv1 = nn.Conv2d(num_channels, n1, kernel_size=f1)
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=f2)
        self.conv3 = nn.Conv2d(n2, num_channels, kernel_size=f3)
        self.relu = nn.ReLU(inplace=True)
        
        torch.nn.init.xavier_normal_(self.conv1.weight)
        torch.nn.init.xavier_normal_(self.conv2.weight)
        torch.nn.init.xavier_normal_(self.conv3.weight)
        
        torch.nn.init.zeros_(self.conv1.bias)
        torch.nn.init.zeros_(self.conv2.bias)
        torch.nn.init.zeros_(self.conv3.bias)
    
    def forward(self, x) :
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
