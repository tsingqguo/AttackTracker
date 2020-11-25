# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import visdom,torch

from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker
from pysot.core.config import cfg
#from tools.saliency.networks.poolnet import build_model, weights_init
import cv2

class CrossSiamRPNTracker(SiameseTracker):
    def __init__(self, model,cfg):
        super(CrossSiamRPNTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.cfg = cfg
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()


    def generate_anchor(self, score_size):
        cfg=self.cfg
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox,ifdetect = False,using_detect=False,checkpoint=None):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        cfg = self.cfg

        if ifdetect:

            # Transforms
            self.center_pos = np.array([img.shape[1]//2,
                                        img.shape[0]//2])
            self.size = np.array([img.shape[1]*0.2,img.shape[0]*0.2])

            if using_detect:
                try:
                    from extern.ssd_detect.ssd_detect import detect

                    bboxes=detect(img,min_score=0.1,max_overlap=0.1,top_k=10,checkpoint=checkpoint)
                    center_pos = np.array([img.shape[1] // 2,img.shape[0] // 2])

                    bbox_center_dist = np.sqrt(np.square(bboxes[:,0]-center_pos[0])+np.square(bboxes[:,1]-center_pos[1]))

                    bbox = bboxes[bbox_center_dist==bbox_center_dist.min(),:].squeeze(0)

                    self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                                bbox[1]+(bbox[3]-1)/2])
                    self.size = np.array([bbox[2], bbox[3]])
                except:
                    self.center_pos = np.array([img.shape[1] // 2,
                                                img.shape[0] // 2])
                    self.size = np.array([img.shape[1] * 0.2, img.shape[0] * 0.2])

            # # calculate z crop size
            # w_x = size[0] + 1.0 * np.sum(size)
            # h_x = size[1] + 1.0 * np.sum(size)
            # s_x = round(np.sqrt(w_x * h_x))
            #
            # # calculate channle average
            # self.channel_average = np.mean(img, axis=(0, 1))
            #
            # # get crop
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            # tmp_img = self.get_subwindow(img, center_pos,
            #                             cfg.TRACK.INSTANCE_SIZE,
            #                             s_x, self.channel_average)
            #
            # net_saliecny = build_model("resnet").cuda()
            # net_saliecny.eval()  # use_global_stats = True
            # net_saliecny.apply(weights_init)
            # saliency_path = "/home/guoqing/projects/AttackTrack/extern/"
            # net_saliecny.base.load_pretrained_model(
            #     torch.load(saliency_path+"saliency/dataset/pretrained/resnet50_caffe.pth"))
            # net_saliecny.load_state_dict(torch.load(saliency_path+"saliency/results/run-0/models/final.pth"))
            # net_saliecny.eval()  # use_global_stats = True
            # pred_ = net_saliecny(tmp_img)

        else:

            self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                        bbox[1]+(bbox[3]-1)/2])
            self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.model.template(z_crop)

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

    def track(self,img,x_crop=[],ispert=False):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        if isinstance(img,np.ndarray):
            img_h,img_w = img.shape[0],img.shape[1]
        else:
            img_h,img_w = img.size()[2],img.size()[3]

        cfg = self.cfg
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)


        if ispert==False: #or cfg.ATTACKER.GEN_ADV_IMG:
            x_crop = self.get_subwindow(img, self.center_pos,
                                        cfg.TRACK.INSTANCE_SIZE,
                                        round(s_x), self.channel_average)

        outputs = self.model.track(x_crop)

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        if cfg.ATTACKER.SHOW:
            vis = visdom.Visdom(env='Adversarial Example Showing')
            vis.images(x_crop, win='X_track')
            disp_score =self.model.log_softmax(outputs['cls'])
            disp_score = disp_score[:,:,:,:,1].permute(1,0,2,3)
            disp_score_min = torch.min(torch.min(disp_score,2)[0],2)[0]
            disp_score_max = torch.max(torch.max(disp_score, 2)[0],2)[0]
            dispmin = torch.zeros(disp_score.size())
            dispmax = torch.zeros(disp_score.size())
            for i in range(4):
                dispmin[i,:,:,:] = disp_score_min[i].repeat(1,1,21,21)
                dispmax[i,:,:,:] = disp_score_max[i].repeat(1,1,21,21)
            disp_score = (disp_score-dispmin)/(dispmax-dispmin)*255
            vis.images(disp_score,win='Response_track')

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR


        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, [img_h,img_w])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
                'bbox': bbox,
                'best_score': best_score
               }

