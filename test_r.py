#%%
import matplotlib.image
import matplotlib.pyplot
import torch
import matplotlib
import model
import cv2
import numpy as np

device = 'cuda'

model_s = model.SRCNN().to(device)

model_s.load_state_dict(torch.load(r"saved_test_model\test_model.h5"))

model_s.eval()
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

matplotlib.pyplot.imshow(srcnn_img_r)
# %%