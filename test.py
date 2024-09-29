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

from model import SRCNN

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
epochs = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = "Data_SRCNN\\T91"
save_path = "saved_test_model\\test_model.h5"

model_s = SRCNN()
model_s.load_state_dict(torch.load(save_path))
hr_img_path = "Data_SRCNN\\T91\\t1.png"

hr_img = cv2.imread(hr_img_path)
hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
print("img shape : {}".format(hr_img.shape))

plt.imshow(hr_img)

hr_img = hr_img.astype(np.float32)/255.0
temp_img = cv2.resize(hr_img, dsize=(0, 0), fx = 1/upscale_factor, fy = 1/upscale_factor, interpolation = cv2.INTER_AREA)
bicubic_img = cv2.resize(temp_img, dsize = (0, 0), fx = upscale_factor, fy = upscale_factor, interpolation = cv2.INTER_CUBIC)

model_s.eval()
input_img = bicubic_img.transpose((2, 0, 1))
input_img = torch.tensor(input_img).unsqueeze(0).to(device)

with torch.no_grad() :
    srcnn_img = model_s(input_img)
    
srcnn_img = srcnn_img.squeeze().cpu().numpy().transpose((1,2,0))

fig, axes = plt.subplots(1, 3, figsize=(10, 5))

axes[0].imshow(hr_img)
axes[0].set_title('hr')

axes[1].imshow(bicubic_img)
axes[1].set_title('bicubic')

axes[2].imshow(srcnn_img)
axes[2].set_title('srcnn')  

#real test
hr_img_path_r = "Data_SRCNN\\T91\\169900813.png"

temp_img_r = cv2.imread(hr_img_path_r)
temp_img_r = cv2.cvtColor(temp_img_r, cv2.COLOR_BGR2RGB)
input_img_r = temp_img_r.astype(np.float32)/255.0
input_img_r = input_img_r.transpose((2, 0, 1))
input_img_r = torch.tensor(input_img_r).unsqueeze(0).to(device)

with torch.no_grad() :
    srcnn_img_r = model_s(input_img_r)
    
srcnn_img_r = srcnn_img_r.squeeze().cpu().numpy().transpose((1,2,0))

fig_r, axes_r = plt.subplots(1, 2, figsize=(10, 5))
axes_r[0].imshow(temp_img_r)
axes_r[1].imshow(srcnn_img_r)

# %%