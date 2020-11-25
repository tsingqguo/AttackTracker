# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.neck import get_neck


class AttModelBuilder(nn.Module):
    def __init__(self):
        super(AttModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

        # build mask head
        if cfg.MASK.MASK:
            self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                           **cfg.MASK.KWARGS)

            if cfg.REFINE.REFINE:
                self.refine_head = get_refine_head(cfg.REFINE.TYPE)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def validate(self,x_crop,zf):
        # get feature
        xf = self.backbone(x_crop)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.rpn_head(zf, xf)
        # get loss
        cls = self.log_softmax(cls)

        return cls

    def forward(self, data):
        """ only used in training
        """
        if cfg.CUDA:
            zf = data['template_zf'].cuda()
            search = data['search'].cuda()
            label_cls = data['label_cls'].cuda()
            adv_cls = data['adv_cls'].cuda()
        else:
            zf = data['template_zf']
            search = data['search']
            label_cls = data['label_cls']
            adv_cls = data['adv_cls']

        # get feature
        xf = self.backbone(search)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.rpn_head(zf, xf)

        # get loss
        cls = self.log_softmax(cls)

        cls_loss = select_cross_entropy_loss(cls, adv_cls)

        # calculate validate candidates
        # diff_cls = cls[:,:,:,:,1]-cls[:,:,:,:,0]
        # valid_cls = label_cls-adv_cls
        # valid_cls = diff_cls.mul(valid_cls)
        # valid_cls = valid_cls.sum()

        outputs = {}
        outputs['total_loss'] = cls_loss
        outputs['cls'] = cls

        if cfg.MASK.MASK:
            # TODO
            mask, self.mask_corr_feature = self.mask_head(zf, xf)
            mask_loss = None
            outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
            outputs['mask_loss'] = mask_loss

        return outputs