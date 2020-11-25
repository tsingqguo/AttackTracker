# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F

from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.neck import get_neck


class CROSSModelBuilder(nn.Module):
    def __init__(self,cfg):
        super(CROSSModelBuilder, self).__init__()

        self.cfg = cfg

        # build backbone
        self.backbone = get_backbone(self.cfg.BACKBONE.TYPE,
                                     **self.cfg.BACKBONE.KWARGS)

        # build adjust layer
        if self.cfg.ADJUST.ADJUST:
            self.neck = get_neck(self.cfg.ADJUST.TYPE,
                                 **self.cfg.ADJUST.KWARGS)

        # build rpn head
        self.rpn_head = get_rpn_head(self.cfg.RPN.TYPE,
                                     **self.cfg.RPN.KWARGS)

        # build mask head
        if self.cfg.MASK.MASK:
            self.mask_head = get_mask_head(self.cfg.MASK.TYPE,
                                           **self.cfg.MASK.KWARGS)

            if self.cfg.REFINE.REFINE:
                self.refine_head = get_refine_head(self.cfg.REFINE.TYPE)

    def template(self, z):
        zf = self.backbone(z)
        if self.cfg.MASK.MASK:
            zf = zf[-1]
        if self.cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if self.cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if self.cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.rpn_head(self.zf, xf)
        if self.cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc,
                'mask': mask if self.cfg.MASK.MASK else None
               }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if self.cfg.MASK.MASK:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if self.cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        cls, loc = self.rpn_head(zf, xf)

        # get loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

        outputs = {}
        outputs['total_loss'] = self.cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            self.cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        if self.cfg.MASK.MASK:
            # TODO
            mask, self.mask_corr_feature = self.mask_head(zf, xf)
            mask_loss = None
            outputs['total_loss'] += self.cfg.TRAIN.MASK_WEIGHT * mask_loss
            outputs['mask_loss'] = mask_loss
        return outputs