import os
import PIL.Image as Image
import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as transforms
from tvnet import *

# load image
img1 = Image.open('frame/img1.png').convert('RGB')
img2 = Image.open('frame/img2.png').convert('RGB')
h, w= img1.size
plt.imshow(img1)

# calculate optical flow
net = TVNet(max_iterations=50)
#u1, u2, rho = net(img1,img2)
loss, u1, u2 = net.get_loss(img1,img2)

# Save flow map to file, for visualization in matlab
u1_np = np.squeeze(u1.detach().numpy())
u2_np = np.squeeze(u2.detach().numpy())
flow_mat = np.zeros([h, w, 2])
flow_mat[:, :, 0] = u1_np
flow_mat[:, :, 1] = u2_np

if not os.path.exists('result'):
    os.mkdir('result')
res_path = os.path.join('result', 'result-pytorch.mat')
sio.savemat(res_path, {'flow': flow_mat})

## note that the flow map can also be visulized using library cv2
## # Sample code using cv2:
## import cv2
## img1cv = cv2.imread('frame/img1.png')
## plt.imshow(img1cv)
## #Use Hue, Saturation, Value colour model 
## hsv = np.zeros(img1cv.shape, dtype=np.uint8)
## hsv[..., 1] = 255
## mag, ang = cv2.cartToPolar(u1.detach().squeeze().numpy(), u2.detach().squeeze().numpy())
## hsv[..., 0] = ang * 180 / np.pi / 2
## hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
## bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
## cv2.imshow("colored flow", bgr)
## cv2.waitKey(0)
## cv2.destroyAllWindows()
