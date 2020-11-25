# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.xcorr import xcorr_fast, xcorr_depthwise
from pysot.utils.siamf import Trans,CalTrans
from pysot.core.config import cfg

class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

class UPChannelRPN(RPN):
    def __init__(self, anchor_num=5, feature_in=256):
        super(UPChannelRPN, self).__init__()

        cls_output = 2 * anchor_num
        loc_output = 4 * anchor_num

        self.template_cls_conv = nn.Conv2d(feature_in,
                feature_in * cls_output, kernel_size=3)
        self.template_loc_conv = nn.Conv2d(feature_in,
                feature_in * loc_output, kernel_size=3)

        self.search_cls_conv = nn.Conv2d(feature_in,
                feature_in, kernel_size=3)
        self.search_loc_conv = nn.Conv2d(feature_in,
                feature_in, kernel_size=3)

        self.loc_adjust = nn.Conv2d(loc_output, loc_output, kernel_size=1)

    def forward(self, z_f, x_f):
        cls_kernel = self.template_cls_conv(z_f)
        loc_kernel = self.template_loc_conv(z_f)

        cls_feature = self.search_cls_conv(x_f)
        loc_feature = self.search_loc_conv(x_f)

        cls = xcorr_fast(cls_feature, cls_kernel)
        loc = self.loc_adjust(xcorr_fast(loc_feature, loc_kernel))
        return cls, loc

class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels,
                 kernel_size=3, hidden_kernel_size=5, TRANS=False):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.head = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=1)
                )
        self.V=[]
        self.U=[]
        self.lambda_v = cfg.SIAMF.LAMBDA_V
        self.lambda_u = cfg.SIAMF.LAMBDA_U
        self.lr_u = cfg.SIAMF.LR_U
        self.lr_v = cfg.SIAMF.LR_V
        self.TRANS= TRANS

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        # add the DSiam Trans here
        if self.TRANS:
            kernel= Trans(kernel, self.V, self.lr_v)
            search = Trans(search, self.U, self.lr_u)
        feature = xcorr_depthwise(search, kernel)
        out = self.head(feature)
        return out

    def update_v(self,kernel,kernel_new):
        kernel_new = self.conv_kernel(kernel_new)
        kernel = self.conv_kernel(kernel)
        self.V = CalTrans(kernel,kernel_new,self.lambda_v)

    def update_u(self,search,search_new):
        search_new = self.conv_search(search_new)
        search = self.conv_search(search)
        self.U = CalTrans(search,search_new,self.lambda_u)

class DepthwiseRPN(RPN):
    def __init__(self, anchor_num=5, in_channels=256, out_channels=256):
        super(DepthwiseRPN, self).__init__()

        self.cls = DepthwiseXCorr(in_channels, out_channels, 2 * anchor_num,TRANS=cfg.SIAMF.TRANS)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4 * anchor_num)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return cls, loc

    def update(self,xf,xbf,zf,ztf):
        self.cls.update_v(zf,ztf)
        self.cls.update_u(xf,xbf)

    def set_params(self,lambda_v = cfg.SIAMF.LAMBDA_V,lambda_u = cfg.SIAMF.LAMBDA_U,
                 lr_v = cfg.SIAMF.LR_V,lr_u = cfg.SIAMF.LR_U):
        self.cls.lr_v=lr_v
        self.cls.lr_u=lr_u
        self.cls.lambda_v=lambda_v
        self.cls.lambda_u=lambda_u

class MultiRPN(RPN):
    def __init__(self, anchor_num, in_channels, weighted=False):
        super(MultiRPN, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('rpn' + str(i + 2),
                    DepthwiseRPN(anchor_num, in_channels[i], in_channels[i]))
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))

    def forward(self, z_fs, x_fs):
        cls = []
        loc = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            rpn = getattr(self, 'rpn'+str(idx))
            c, l = rpn(z_f, x_f)
            cls.append(c)
            loc.append(l)

        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            loc_weight = F.softmax(self.loc_weight, 0)

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            return weighted_avg(cls, cls_weight), weighted_avg(loc, loc_weight)
        else:
            return avg(cls), avg(loc)

    def update(self,x_fs,xb_fs,z_fs,zt_fs):
        for idx, (z_f, zt_f,x_f,xb_f) in enumerate(zip(z_fs, zt_fs, x_fs, xb_fs), start=2):
            rpn = getattr(self, 'rpn'+str(idx))
            rpn.update(x_f,xb_f,z_f,zt_f)

    def set_params(self,lambda_v=cfg.SIAMF.LAMBDA_V, lambda_u=cfg.SIAMF.LAMBDA_U,
                 lr_v=cfg.SIAMF.LR_V, lr_u=cfg.SIAMF.LR_U):
        for idx in range(len(cfg.SIAMF.LAYERS)):
            rpn = getattr(self, 'rpn'+str(idx+2))
            rpn.set_params(lambda_v,lambda_u,lr_v,lr_u)