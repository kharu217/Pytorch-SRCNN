#%%

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

path = "Data_SRCNN\\T91"

#use cpu for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset) :
    def __init__(self, img_paths, input_size, output_size, stride = 14, upscale_factor = 3) :
        super(CustomDataset, self).__init__()

        self.img_paths = glob.glob(img_paths + '\\*.png')
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
            w = img.shape[1] - np.mod(img.shape[0], self.upscale_factor)
            
            img = img[:h, :w]
            
            #zoom img
            label = img.astype(np.float32)/255.0
            temp_input = cv2.resize(label, dsize=(0, 0), fx = 1/self.upscale_factor, fy = 1/self.upscale_factor, interpolation = cv2.INTER_AREA)
            input = cv2.resize(temp_input, dsize = (0, 0), fx = self.upscale_factor, fy = self.upscale_factor, interpolation = cv2.INTER_CUBIC)
        
            for h in range(0, input.shape[0] - self.input_size + 1, self.stride) :
                for w in range(0, input.shape[1] - self.input_size + 1) :
                    sub_lr_img = input[h:h+self.input_size, w:w+self.input_size, :]
                    sub_hr_img = label[h+self.pad:self.pad+self.output_size, w+self.pad:w+self.pad+self.output_size, :]
                    
                    sub_lr_img = sub_lr_img.transpose((2, 0, 1))
                    sub_hr_img = sub_hr_img.transpose((2, 0, 1))
                    
                    self.sub_lr_imgs.append(sub_lr_img)
                    self.sub_hr_imgs.append(sub_hr_img)
                
        print("Finish, Created {} Sub-Images".format(len(self.sub_lr_imgs)))     
        self.sub_lr_imgs = np.asarray(self.sub_lr_imgs[0])
        self.sub_hr_imgs = np.asarray(self.sub_hr_imgs[0])

    def __len__(self) :
        return len(self.sub_lr_imgs)
    
    def __getitem__(self, idx) :
        lr_img = self.sub_lr_imgs[idx]
        hr_img = self.sub_hr_imgs[idx]
        return lr_img, hr_img    

train_dataset = CustomDataset(path, input_size, output_size)
img = cv2.imread(train_dataset.img_paths[12])
print(img.shape)
plt.imshow(img)

#img lr, hr comparison(has error)
# fig, axes = plt.subplots(1,2, figsize = (5, 5))
# idx = random.randint(0, len(train_dataset.sub_lr_imgs))

# axes[0].set_title('lr')
# axes[1].set_title('hr')
# axes[0].imshow(train_dataset.sub_lr_imgs[idx].transpose(1, 2, 0))
# axes[1].imshow(train_dataset.sub_hr_imgs[idx].transpose(1, 2, 0))

# print(idx)
# %%