# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import torch

from pysot.core.config import cfg
import torch.nn.functional as F


class BaseTracker(object):
    """ Base tracker of single objec tracking
    """
    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox(list): [x, y, width, height]
                        x, y need to be 0-based
        """
        raise NotImplementedError

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        raise NotImplementedError


class SiameseTracker(BaseTracker):

    # def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
    #     """
    #     args:
    #         im: bgr based image
    #         pos: center position
    #         model_sz: exemplar size
    #         s_z: original size
    #         avg_chans: channel average
    #     """
    #     if isinstance(pos, float):
    #         pos = [pos, pos]
    #     sz = original_sz
    #     im_sz = im.shape
    #     c = (original_sz + 1) / 2
    #     # context_xmin = round(pos[0] - c) # py2 and py3 round
    #     context_xmin = np.floor(pos[0] - c + 0.5)
    #     context_xmax = context_xmin + sz - 1
    #     # context_ymin = round(pos[1] - c)
    #     context_ymin = np.floor(pos[1] - c + 0.5)
    #     context_ymax = context_ymin + sz - 1
    #     left_pad = int(max(0., -context_xmin))
    #     top_pad = int(max(0., -context_ymin))
    #     right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    #     bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))
    #
    #     context_xmin = context_xmin + left_pad
    #     context_xmax = context_xmax + left_pad
    #     context_ymin = context_ymin + top_pad
    #     context_ymax = context_ymax + top_pad
    #
    #     r, c, k = im.shape
    #     if any([top_pad, bottom_pad, left_pad, right_pad]):
    #         size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
    #         te_im = np.zeros(size, np.uint8)
    #         te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
    #         if top_pad:
    #             te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
    #         if bottom_pad:
    #             te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
    #         if left_pad:
    #             te_im[:, 0:left_pad, :] = avg_chans
    #         if right_pad:
    #             te_im[:, c + left_pad:, :] = avg_chans
    #         im_patch = te_im[int(context_ymin):int(context_ymax + 1),
    #                          int(context_xmin):int(context_xmax + 1), :]
    #     else:
    #         im_patch = im[int(context_ymin):int(context_ymax + 1),
    #                       int(context_xmin):int(context_xmax + 1), :]
    #
    #     if not np.array_equal(model_sz, original_sz):
    #         im_patch = cv2.resize(im_patch, (model_sz, model_sz))
    #     im_patch = im_patch.transpose(2, 0, 1)
    #     im_patch = im_patch[np.newaxis, :, :, :]
    #     im_patch = im_patch.astype(np.float32)
    #     im_patch = torch.from_numpy(im_patch)
    #     if cfg.CUDA:
    #         im_patch = im_patch.cuda()
    #     return im_patch

    def get_subwindow(self,im_tensor, pos, model_sz, original_sz, avg_chans):

            if isinstance(im_tensor,np.ndarray):
                im_tensor = torch.from_numpy(im_tensor).float()
            else:
                im_tensor = im_tensor.squeeze(0).permute(1, 2, 0)

            if isinstance(pos, float):
                pos = [pos, pos]
            sz = original_sz

            im = im_tensor
            im_sz = im.size()

            c = (original_sz + 1) / 2
            # context_xmin = round(pos[0] - c) # py2 and py3 round
            context_xmin = np.floor(pos[0] - c + 0.5)
            context_xmax = context_xmin + sz - 1
            # context_ymin = round(pos[1] - c)
            context_ymin = np.floor(pos[1] - c + 0.5)
            context_ymax = context_ymin + sz - 1
            left_pad = int(max(0., -context_xmin))
            top_pad = int(max(0., -context_ymin))
            right_pad = int(max(0., context_xmax - im_sz[1] + 1))
            bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

            context_xmin = context_xmin + left_pad
            context_xmax = context_xmax + left_pad
            context_ymin = context_ymin + top_pad
            context_ymax = context_ymax + top_pad

            r, c, k = im.size()[0],im.size()[1],im.size()[2]
            avg_chans = torch.from_numpy(avg_chans)

            if any([top_pad, bottom_pad, left_pad, right_pad]):
                size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
                te_im = torch.zeros(size)

                te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im

                if top_pad:
                    te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
                if bottom_pad:
                    te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
                if left_pad:
                    te_im[:, 0:left_pad, :] = avg_chans
                if right_pad:
                    te_im[:, c + left_pad:, :] = avg_chans

                im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                           int(context_xmin):int(context_xmax + 1), :]
            else:
                im_patch = im[int(context_ymin):int(context_ymax + 1),
                           int(context_xmin):int(context_xmax + 1), :]

            im_patch = im_patch.permute(2, 0, 1)
            im_patch = im_patch.unsqueeze(0)

            if not np.array_equal(model_sz, original_sz):
                im_patch = F.upsample(im_patch, size=(model_sz,model_sz), mode='bilinear')

            if cfg.CUDA:
                im_patch = im_patch.cuda()

            return im_patch


    def get_orgimg(self,im,im_patch,pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape

        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            tim_patch = te_im[int(context_ymin):int(context_ymax + 1),
                             int(context_xmin):int(context_xmax + 1), :]
        else:
            tim_patch = im[int(context_ymin):int(context_ymax + 1),
                          int(context_xmin):int(context_xmax + 1), :]
        if cfg.CUDA:
            im_patch = im_patch.cpu()
        im_patch = torch.squeeze(im_patch)
        im_patch = im_patch.detach().numpy()
        im_patch = im_patch.astype(np.uint8)
        im_patch = im_patch.transpose(1,2,0)
        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (tim_patch.shape[0],tim_patch.shape[1]))

        if any([top_pad, bottom_pad, left_pad, right_pad]):
            te_im[int(context_ymin):int(context_ymax + 1),
                int(context_xmin):int(context_xmax + 1), :]=im_patch
            im = te_im[top_pad:top_pad + r, left_pad:left_pad + c, :]
        else:
            im[int(context_ymin):int(context_ymax + 1),
                int(context_xmin):int(context_xmax + 1), :]=im_patch


        return im

class pyBaseTracker:

    def visdom_ui_handler(self, data):
        if data['event_type'] == 'KeyPress':
            if data['key'] == ' ':
                self.pause_mode = not self.pause_mode

            elif data['key'] == 'ArrowRight' and self.pause_mode:
                self.step = True

    def __init__(self, params):
        self.params = params
        self.pause_mode = False
        self.step = False


    def init(self, image, info: dict) -> dict:
        """Overload this function in your tracker. This should initialize the model."""
        raise NotImplementedError


    def track(self, image) -> dict:
        """Overload this function in your tracker. This should track in the frame and update the model."""
        raise NotImplementedError
