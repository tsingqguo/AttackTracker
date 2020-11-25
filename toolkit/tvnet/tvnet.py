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
from pysot.core.config import cfg

class Centered_Grad(nn.Module):
    # layer to calculate center gradient
    def __init__(self):
        super(Centered_Grad, self).__init__()
        # initialize center kernels
        self.x_ker_init = torch.tensor([[[[-0.5, 0, 0.5]]]],dtype=torch.float, requires_grad=True)
        self.y_ker_init = torch.tensor([[[[-0.5], [0], [0.5]]]],dtype=torch.float, requires_grad=True)
        self.center_conv_x = nn.Conv2d(1, 1, (1,3),  padding=(0,1), bias=False)
        self.center_conv_x.weight.data = self.x_ker_init
        self.center_conv_y = nn.Conv2d(1, 1, (3,1), padding=(1,0), bias=False)
        self.center_conv_y.weight.data = self.y_ker_init
    def forward(self, x):
        # Input x shape: N*C*H*W
        assert len(x.shape) == 4
        diff_x = self.center_conv_x(x)
        diff_y = self.center_conv_y(x)
        # refine the boundary
        ## ! check if the selection of tensor reduce the dimension!
        first_col = 0.5 * (x[:,:,:,1:2]-x[:,:,:,0:1])
        last_col = 0.5 * (x[:,:,:,-1:]-x[:,:,:,-2:-1])
        diff_x_valid = diff_x[:,:,:,1:-1]
        diff_x = torch.cat((first_col, diff_x_valid, last_col), 3)
        
        first_row = 0.5 * (x[:,:,1:2,:]-x[:,:,0:1,:])
        last_row = 0.5 * (x[:,:,-1:,:]-x[:,:,-2:-1,:])
        diff_y_valid = diff_y[:,:,1:-1,:]
        diff_y = torch.cat((first_row, diff_y_valid, last_row), 2)
        return diff_x, diff_y

class Forward_Grad(nn.Module):
    # layer to calculate forward gradient
    def __init__(self):
        super(Forward_Grad, self).__init__()
        # initialize center kernels
        self.x_ker_init = torch.tensor([[[[-1, 1]]]],dtype=torch.float, requires_grad=True)
        self.y_ker_init = torch.tensor([[[[-1], [1]]]],dtype=torch.float, requires_grad=True)
        self.forward_conv_x = nn.Conv2d(1, 1, (1,2), bias=False)
        self.forward_conv_x.weight.data = self.x_ker_init
        self.forward_conv_y =nn.Conv2d(1, 1, (2,1), bias=False)
        self.forward_conv_y.weight.data = self.y_ker_init
    def forward(self, x):
        # Input x shape: N*C*H*W
        assert len(x.shape) == 4
        x.padright = F.pad(x, (0,1,0,0))
        x.padbottom = F.pad(x, (0,0,0,1))
        diff_x = self.forward_conv_x(x.padright)
        diff_y = self.forward_conv_y(x.padbottom)

        # refine the boundary
        diff_x[:,:,:,(x.shape[3]-1)] = 0
        diff_y[:,:,(x.shape[2]-1),:] = 0
        return diff_x, diff_y

class Div(nn.Module):
    # layer to calculate divergence
    def __init__(self):
        super(Div, self).__init__()
        # initialize div kernels
        self.x_ker_init = torch.tensor([[[[-1, 1]]]],dtype=torch.float, requires_grad=True)
        self.y_ker_init = torch.tensor([[[[-1], [1]]]],dtype=torch.float, requires_grad=True)
        self.div_conv_x = nn.Conv2d(1, 1, (1,2), bias = False)
        self.div_conv_x.weight.data = self.x_ker_init
        self.div_conv_y =nn.Conv2d(1, 1, (2,1), bias = False)
        self.div_conv_y.weight.data = self.y_ker_init
    def forward(self, x, y):
        assert len(x.shape) == 4
        assert len(y.shape) == 4
        # refine the boundary
        first_col = torch.zeros([x.shape[0], x.shape[1], x.shape[2], 1])
        x_pad = torch.cat((first_col, x[:,:,:,0:-1]), dim = 3)
        first_row = torch.zeros([x.shape[0], x.shape[1], 1, x.shape[3]])
        y_pad = torch.cat((first_row, y[:,:,0:-1,:]), dim = 2)
        
        x.padright = F.pad(x_pad, (0,1,0,0))
        y.padbottom = F.pad(y_pad, (0,0,0,1))
        
        diff_x = self.div_conv_x(x.padright)
        diff_y = self.div_conv_y(y.padbottom)
        
        div = diff_x + diff_y
        return div

    
    
class IterBlock(nn.Module):
    GRAD_IS_ZERO = 1e-12
    def __init__(self, 
                 l_t=0.045,  # lbda*theta
                 taut=2.5, # tau/theta
                 theta=0.3  # weight parameter for (u - v)^2
                ):
        super(IterBlock, self).__init__()
        self.l_t = l_t
        self.taut = taut
        self.theta = theta
        self.divergence_x = Div()
        self.divergence_y = Div()
        self.forward_gradient_x = Forward_Grad()
        self.forward_gradient_y = Forward_Grad()

    def forward(self, diff2_x_warp, diff2_y_warp, u1, u2, grad, p11 , p12, p21, p22,
                rho_c=0):
    
        rho = rho_c + diff2_x_warp * u1 + diff2_y_warp * u2 + self.GRAD_IS_ZERO;
        # calculate v^k+1 with thresholding operation
        masks1 = rho < -self.l_t * grad
        d1_1 = torch.where(masks1, self.l_t * diff2_x_warp, torch.zeros_like(diff2_x_warp))
        d2_1 = torch.where(masks1, self.l_t * diff2_y_warp, torch.zeros_like(diff2_y_warp))

        masks2 = rho > self.l_t * grad
        d1_2 = torch.where(masks2, -self.l_t * diff2_x_warp, torch.zeros_like(diff2_x_warp))
        d2_2 = torch.where(masks2, -self.l_t * diff2_y_warp, torch.zeros_like(diff2_y_warp))

        masks3 = (~masks1) & (~masks2) & (grad > self.GRAD_IS_ZERO)
        d1_3 = torch.where(masks3, -rho / grad * diff2_x_warp, torch.zeros_like(diff2_x_warp))
        d2_3 = torch.where(masks3, -rho / grad * diff2_y_warp, torch.zeros_like(diff2_y_warp))

        v1 = d1_1 + d1_2 + d1_3 + u1
        v2 = d2_1 + d2_2 + d2_3 + u2

        u1 = (v1 + self.theta * self.divergence_x(p11, p12)).squeeze(1)
        u2 = (v2 + self.theta * self.divergence_y(p21, p22)).squeeze(1)

        u1x, u1y = self.forward_gradient_x(u1.unsqueeze(1))
        u2x, u2y = self.forward_gradient_y(u2.unsqueeze(1))

        p11 = (p11 + self.taut * u1x) / (
            1.0 + self.taut * torch.sqrt(u1x.pow(2) + u1y.pow(2) + self.GRAD_IS_ZERO));
        p12 = (p12 + self.taut * u1y) / (
            1.0 + self.taut * torch.sqrt(u1x.pow(2) + u1y.pow(2) + self.GRAD_IS_ZERO));
        p21 = (p21 + self.taut * u2x) / (
            1.0 + self.taut * torch.sqrt(u2x.pow(2) + u2y.pow(2) + self.GRAD_IS_ZERO));
        p22 = (p22 + self.taut * u2y) / (
            1.0 + self.taut * torch.sqrt(u2x.pow(2) + u2y.pow(2) + self.GRAD_IS_ZERO));
        return p11, p12, p21, p22, rho, u1, u2
    
    
class WarpBlock(nn.Module):
    # Block for each Warp operation, contain many IterBlock
    GRAD_IS_ZERO = 1e-12

    def __init__(self, 
                 tau=0.25,  # time step
                 lbda=0.15,  # weight parameter for the data term
                 theta=0.3,  # weight parameter for (u - v)^2
                 max_iterations=10  # maximum number of iterations for optimization
                ):
        super(WarpBlock, self).__init__()
        self.l_t = lbda * theta
        self.theta = theta
        self.taut = tau / theta
        self.max_iterations = max_iterations
        self.centered_gradient = Centered_Grad()

        for i in range(max_iterations):
            iterblock = IterBlock(self.l_t, self.taut, self.theta)
            self.add_module('iterblock%d' % (i+1), iterblock)
            
    
    def forward(self, x1, x2, u1, u2):
        
        diff2_x, diff2_y = self.centered_gradient(x2)
        p11 = p12 = p21 = p22 = torch.zeros_like(x1)
        
        x2_warp = self.warp_image(x2, u1, u2)
        x2_warp = x2_warp.view(x2.shape)

        diff2_x_warp = self.warp_image(diff2_x, u1, u2)
        diff2_x_warp = diff2_x_warp.view(diff2_x.shape)

        diff2_y_warp = self.warp_image(diff2_y, u1, u2)
        diff2_y_warp = diff2_y_warp.view(diff2_y.shape)

        diff2_x_sq = diff2_x_warp.pow(2)
        diff2_y_sq = diff2_y_warp.pow(2)

        grad = diff2_x_sq + diff2_y_sq + self.GRAD_IS_ZERO

        rho_c = x2_warp - diff2_x_warp * u1 - diff2_y_warp * u2 - x1

        for ii in range(self.max_iterations):
            [p11, p12, p21, p22, rho, u1, u2] = self._modules['iterblock%d' % (ii+1)](diff2_x_warp, 
                                                                                      diff2_y_warp, 
                                                                                      u1, u2, grad, 
                                                                                      p11 , p12, p21, p22,
                                                                                      rho_c)
        return u1, u2, rho
        
    def warp_image(self, x, u, v):
        # warp image according to displacement u and v
        # Still need to think padding_mode in grid_sample (zero-padding or border-padding)
        assert len(x.shape) == 4
        assert len(u.shape) == 3
        assert len(v.shape) == 3
        N,C,iH,iW = x.shape
        tensorHorizontal = torch.linspace(-1.0, 1.0, iW).view(1, 1, iW,1).expand(N, iH, -1, -1)
        tensorVertical = torch.linspace(-1.0, 1.0, iH).view(1, iH, 1,1).expand(N, -1, iW, -1)

        tensorGrid = torch.cat([tensorHorizontal, tensorVertical], 3)
        tensorFlow = torch.cat((u.unsqueeze(3), v.unsqueeze(3)), dim=3)
        # normalize the flow to grid scale
        tensorFlow = torch.cat([ tensorFlow[:, :, :, 0:1] / ((iW - 1.0) / 2.0), tensorFlow[:, :, :, 1:2] / ((iH - 1.0) / 2.0) ], 3)
        return torch.nn.functional.grid_sample(input=x, grid=(tensorGrid + tensorFlow), mode='bilinear', padding_mode='border')


    
class TVNet(torch.nn.Module):
    # Default Shape:
    # Input image shape: N*C*H*W (C=1 for gray image)
    # Flow shape (flow_x): N*H*W 
    GRAD_IS_ZERO = 1e-12

    def __init__(self, 
                 tau=0.25,  # time step
                 lbda=0.15,  # weight parameter for the data term
                 theta=0.3,  # weight parameter for (u - v)^2
                 max_iterations=10,  # maximum number of iterations for optimization
                ):
        super(TVNet, self).__init__()
        ## self.n_scales = 1
        self.tau = tau
        self.lbda = lbda
        self.theta = theta
        self.max_iterations = max_iterations
        # 1. to grey scale & normalize
        self.trans_grayscale = transforms.Compose([transforms.Grayscale(),transforms.ToTensor()])
        #self.gaussian_smooth = self.gaussian_smooth()
        # 2. gaussiancode smooth
        self.gaussiankernel = torch.FloatTensor([[[[0.000874, 0.006976, 0.01386, 0.006976, 0.000874],
                                            [0.006976, 0.0557, 0.110656, 0.0557, 0.006976],
                                            [0.01386, 0.110656, 0.219833, 0.110656, 0.01386],
                                            [0.006976, 0.0557, 0.110656, 0.0557, 0.006976],
                                            [0.000874, 0.006976, 0.01386, 0.006976, 0.000874]]]])
        
        
        # 3. Warp Block
        self.warpblock = WarpBlock(self.tau, self.lbda, self.theta, 
                              self.max_iterations)

        self.register_parameter('u1', None)
        self.register_parameter('u2', None)


    def forward(self, x1, x2):
        
        ## n_scales = 1
        x1 = self.trans_grayscale(x1)*255
        x2 = self.trans_grayscale(x2)*255
        x1 = torch.unsqueeze(x1, 0)
        x2 = torch.unsqueeze(x2, 0)

        x1 = F.conv2d(x1, self.gaussiankernel, padding=2)
        x2 = F.conv2d(x2, self.gaussiankernel, padding=2)

        # checked
        for i in range(len(x1.shape)):
            assert x1.shape[i] == x2.shape[i]

        height = x1.shape[-2]
        width = x2.shape[-1]
        
        # initialize u0 at the begining of training
        if self.u1 is None:
            self.u1 = nn.Parameter(x1.new_full((1,height, width), 0))  
            self.u2 = nn.Parameter(x2.new_full((1,height, width), 0))
        
        u1 = self.u1.expand(x1.shape[0], -1, -1)
        u2 = self.u2.expand(x2.shape[0], -1, -1)
        u1, u2, rho = self.warpblock(x1, x2, u1, u2)
            
        return u1, u2, rho
    
    def warp_image(self, x, u, v):
        # warp image according to displacement u and v
        # Still need to think padding_mode in grid_sample (zero-padding or border-padding)
        assert len(x.shape) == 4
        assert len(u.shape) == 3
        assert len(v.shape) == 3
        N,C,iH,iW = x.shape
        tensorHorizontal = torch.linspace(-1.0, 1.0, iW).view(1, 1, iW,1).expand(N, iH, -1, -1)
        tensorVertical = torch.linspace(-1.0, 1.0, iH).view(1, iH, 1,1).expand(N, -1, iW, -1)

        tensorGrid = torch.cat([tensorHorizontal, tensorVertical], 3)
        tensorFlow = torch.cat((u.unsqueeze(3), v.unsqueeze(3)), dim=3)
        # normalize the flow to grid scale
        tensorFlow = torch.cat([ tensorFlow[:, :, :, 0:1] / ((iW - 1.0) / 2.0), tensorFlow[:, :, :, 1:2] / ((iH - 1.0) / 2.0) ], 3)
        return torch.nn.functional.grid_sample(input=x, grid=(tensorGrid + tensorFlow), mode='bilinear', padding_mode='border')

    def zoom_size(self, height, width, factor):
        new_height = int(float(height) * factor + 0.5)
        new_width = int(float(width) * factor + 0.5)                  
        return new_height, new_width# Input x shape: N*C*H*W
        assert len(x.shape) == 4
        x_ker_init = torch.tensor([[[[-0.5, 0, 0.5]]]],dtype=torch.float, requires_grad=True)
        diff_x = F.conv2d(x, x_ker_init, padding=(0,1))
        y_ker_init = torch.tensor([[[[-0.5], [0], [0.5]]]],dtype=torch.float, requires_grad=True)
        diff_y = F.conv2d(x, y_ker_init, padding=(1,0))
        # refine the boundary
        ## ! check if the selection of tensor reduce the dimension!
        first_col = 0.5 * (x[:,:,:,1:2]-x[:,:,:,0:1])
        last_col = 0.5 * (x[:,:,:,-1:]-x[:,:,:,-2:-1])
        diff_x_valid = diff_x[:,:,:,1:-1]
        diff_x = torch.cat((first_col, diff_x_valid, last_col), 3)
        
        first_row = 0.5 * (x[:,:,1:2,:]-x[:,:,0:1,:])
        last_row = 0.5 * (x[:,:,-1:,:]-x[:,:,-2:-1,:])
        diff_y_valid = diff_y[:,:,1:-1,:]
        diff_y = torch.cat((first_row, diff_y_valid, last_row), 2)
        return diff_x, diff_y

    def zoom_image(self, x, new_height, new_width):
        assert len(x.shape) == 4   
        return F.interpolate(x, size=(new_height, new_width), mode='bilinear', align_corners=True)
    
    def forward_gradient(self, x):
        assert len(x.shape) == 4

        x.padright = F.pad(x, (0,1,0,0))
        x.padbottom = F.pad(x, (0,0,0,1))

        x_ker_init = torch.tensor([[[[-1, 1]]]],dtype=torch.float, requires_grad=True)
        diff_x = F.conv2d(x.padright, x_ker_init, padding=0)
        y_ker_init = torch.tensor([[[[-1], [1]]]],dtype=torch.float, requires_grad=True)
        diff_y = F.conv2d(x.padbottom, y_ker_init, padding=0)

        # refine the boundary
        diff_x[:,:,:,(x.shape[3]-1)] = 0
        diff_y[:,:,(x.shape[2]-1),:] = 0
        return diff_x, diff_y

    
    def get_loss(self, x1, x2,
             tau=0.25,  # time step
             lbda=0.15,  # weight parameter for the data term
             theta=0.3,  # weight parameter for (u - v)^2
             warps=5,  # number of warpings per scale
             zfactor=0.5,  # factor for building the image piramid
             max_scales=5,  # maximum number of scales for image piramid
             max_iterations=20  # maximum number of iterations for optimization
             ):

        u1, u2, rho = self.forward(x1, x2)
        
        x1 = self.trans_grayscale(x1)*255
        x2 = self.trans_grayscale(x2)*255
        x1 = torch.unsqueeze(x1, 0)
        x2 = torch.unsqueeze(x2, 0)
        
        # computing loss
        u1x, u1y = self.forward_gradient(u1.unsqueeze(1))
        u2x, u2y = self.forward_gradient(u2.unsqueeze(1))

        x2_warp = self.warp_image(x2, u1, u2)
        x2_warp = x2_warp.view(x2.shape)
        loss = lbda * torch.mean(torch.abs(x2_warp - x1)) + torch.mean(
            torch.abs(u1x) + torch.abs(u1y) + torch.abs(u2x) + torch.abs(u2y))
        return loss, u1, u2

