# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import random
import torch.nn.functional as F
import visdom
from pysot.core.config import cfg
import matplotlib.pyplot as plt
from pysot.models.loss import select_cross_entropy_loss
from pysot.datasets.anchor_target import AnchorTarget
from pysot.utils.bbox import center2corner, Center, get_axis_aligned_bbox
from math import cos, sin, pi
import os
import cv2
from pysot.attacker.oim_attacker import OIMAttacker

class CURLAttacker(OIMAttacker):
    def __init__(self, type, max_num=10, eplison=1, inta=10, lamb=0.00001, norm_type='L_inf', apts_num=2, reg_type='weighted',accframes=30):
        
        self.type = type
        # parameters for bim
        self.eplison = eplison
        self.inta = inta
        self.norm_type = norm_type
        self.reg_type = reg_type
        self.max_num = max_num
        self.lamb = lamb
        self.v_id = 0
        # self.st = st()
        self.apts_num = apts_num
        self.target_traj = []
        self.prev_delta = None
        self.tacc = True
        self.meta_path = []
        self.acc_iters = 0
        self.accframes = accframes

        self.down_hill = False
        self.advs_init = True

    def attack(self, tracker, img, prev_perts=None, weights=None,APTS=False, OPTICAL_FLOW=False, ADAPT=False, Enable_same_prev=True):
        """
        args:
            tracker, img(np.ndarray): BGR image
        return:
            adversirial image
        """
        w_z = tracker.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
        h_z = tracker.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = tracker.get_subwindow(img, tracker.center_pos,
                                       cfg.TRACK.INSTANCE_SIZE,
                                       round(s_x), tracker.channel_average)
        outputs = tracker.model.track(x_crop)
        cls = tracker.model.log_softmax(outputs['cls'])
        diff_cls = cls[:, :, :, :, 1] - cls[:, :, :, :, 0]
        label_cls = diff_cls.ge(0).float()

        if self.type == 'UA':
            adv_cls, same_prev = self.ua_label(tracker, scale_z, outputs)
            adv_cls = adv_cls.long()
        elif self.type == 'TA':
            adv_cls, same_prev = self.ta_label(tracker, scale_z, outputs)
            adv_cls = adv_cls.long()

        max_iteration = self.max_num
        if cfg.CUDA:
            pertA = torch.randn(x_crop.size()).cuda()
            pertB = torch.randn(x_crop.size()).cuda()
            advs = torch.randn(x_crop.size()).cuda()
        else:
            pertA = torch.randn(x_crop.size())
            pertB = torch.randn(x_crop.size())
            advs = torch.randn(x_crop.size())

        if prev_perts is None or (same_prev == False and Enable_same_prev == True):
            if cfg.CUDA:
                prev_perts = torch.zeros(x_crop.size()).cuda()
                weights = torch.ones(x_crop.size()).cuda()#torch.ones(1).cuda()
            else:
                prev_perts = torch.zeros(x_crop.size())
                weights = torch.ones(x_crop.size())#torch.ones(1)
        else:
            if APTS == False:
                if self.reg_type == 'weighted':
                    pert_sum = torch.mul(weights, prev_perts).sum(0)
                else:
                    pert_sum = prev_perts.sum(0)
                adv_x_crop = x_crop + pert_sum
                adv_x_crop = torch.clamp(adv_x_crop, 0, 255)
                pert_true = adv_x_crop - x_crop

                if cfg.ATTACKER.GEN_ADV_IMG:
                    adv_img = tracker.get_orgimg(img, x_crop, tracker.center_pos,
                                                 cfg.TRACK.INSTANCE_SIZE,
                                                 round(s_x), tracker.channel_average)
                else:
                    adv_img = None

                return x_crop, pert_true, prev_perts, weights,adv_img
            else:
                max_iteration = self.apts_num
                if self.tacc == False:
                    if cfg.CUDA:
                        prev_perts = torch.zeros(x_crop.size()).cuda()
                        weights = torch.ones(x_crop.size()).cuda()  # torch.ones(1).cuda()
                    else:
                        prev_perts = torch.zeros(x_crop.size())
                        weights = torch.ones(x_crop.size())  # torch.ones(1)

        if cfg.ATTACKER.SHOW:
            vis = visdom.Visdom(
                env='Adversarial Example Showing')  # (server='172.28.144.132',port=8022,env='Adversarial Example Showing')
            vis.images(x_crop, win='X_org')
            vis.images(adv_cls.permute(1, 0, 2, 3), win='adv_cls')

        # start attack
        losses = []
        m = 0

        self.acc_iters += max_iteration
        cls_lossA_prev = -9999.
        self.down_hill = True
        self.advs_init = True
        if isinstance(tracker.model.zf, list):
            zf = torch.cat(tracker.model.zf, 0)
        else:
            zf = tracker.model.zf

        while m < max_iteration:
            data = {
                'template_zf': zf.detach(),
                'search': x_crop.detach(),
                'pertA': pertA.detach(),
                'pertB': pertB.detach(),
                'advs': advs.detach(),
                'adv_cls': adv_cls.detach(),
                'cls_lossA_prev':cls_lossA_prev,
                }
            data['pertA'].requires_grad = True
            data['pertB'].requires_grad = True
            pert, pertA, pertB, advs, loss, cls, cls_lossA_prev = self.oim_once(tracker, data)
            losses.append(loss)
            m += 1
            # disp x_crop for each iteration and the pert
            if cfg.ATTACKER.SHOW:
                if self.reg_type == 'weighted':
                    pert_sum = torch.mul(weights, prev_perts).sum(0)
                else:
                    pert_sum = prev_perts.sum(0)
                x_crop_t = x_crop + pert_sum + pert
                x_crop_t = torch.clamp(x_crop_t, 0, 255)
                vis.images(x_crop_t, win='X_attack')
                vis.images(pert, win='Pert_incremental')
                vis.images(pert_sum + pert, win='Pert_attack')
                plt.plot(losses)
                plt.ylabel('Loss')
                plt.xlabel('Iteration')
                vis.matplot(plt, win='Loss_attack')
                # validate the score
                outputs = tracker.model.track(x_crop_t)
                disp_score = self.log_softmax(outputs['cls'].cpu())
                disp_score = disp_score[:, :, :, :, 1].permute(1, 0, 2, 3)
                disp_score_min = torch.min(torch.min(disp_score, 2)[0], 2)[0]
                disp_score_max = torch.max(torch.max(disp_score, 2)[0], 2)[0]
                dispmin = torch.zeros(disp_score.size())
                dispmax = torch.zeros(disp_score.size())
                for i in range(4):
                    dispmin[i, :, :, :] = disp_score_min[i].repeat(1, 1, disp_score.shape[2], disp_score.shape[3])
                    dispmax[i, :, :, :] = disp_score_max[i].repeat(1, 1, disp_score.shape[2], disp_score.shape[3])
                disp_score = (disp_score - dispmin) / (dispmax - dispmin) * 255
                vis.images(disp_score, win='Response_attack')

        self.opt_flow_prev_xcrop = x_crop
        if cfg.CUDA:
            prev_perts = torch.cat((prev_perts, pert), 0).cuda()
            weights = torch.cat((weights, torch.ones(x_crop.size()).cuda()),0).cuda()
        else:
            prev_perts = torch.cat((prev_perts, pert), 0)
            weights = torch.cat((weights, torch.ones(x_crop.size())), 0)

        if self.reg_type=='weighted':
            pert_sum = torch.mul(weights,prev_perts).sum(0)
        else:
            pert_sum = prev_perts.sum(0)

        adv_x_crop = x_crop + pert_sum
        adv_x_crop = torch.clamp(adv_x_crop, 0, 255)
        pert_true = adv_x_crop - x_crop
        if cfg.ATTACKER.GEN_ADV_IMG:
            adv_img = tracker.get_orgimg(img, x_crop, tracker.center_pos,
                                         cfg.TRACK.INSTANCE_SIZE,
                                         round(s_x), tracker.channel_average)
        else:
            adv_img = None

        return adv_x_crop, pert_true, prev_perts, weights,adv_img


    def oim_once(self, tracker, data):
        if cfg.CUDA:
            zf = data['template_zf'].cuda()
            search = data['search'].cuda()
            pertA = data['pertA'].cuda()
            pertB = data['pertB'].cuda()
            advs = data['advs'].cuda()
            adv_cls = data['adv_cls'].cuda()
            cls_lossA_prev = data['cls_lossA_prev']
        else:
            zf = data['template_zf']
            search = data['search']
            pertA = data['pertA']
            pertB = data['pertB']
            advs = data['advs']
            adv_cls = data['adv_cls']
            cls_lossA_prev = data['cls_lossA_prev']

        track_model = tracker.model

        zf_list = []
        if zf.shape[0] > 1:
            for i in range(0, zf.shape[0]):
                zf_list.append(zf[i, :, :, :].resize_(1, zf.shape[1], zf.shape[2], zf.shape[3]))
        else:
            zf_list = zf

        #generate gaussian noise
        if cfg.CUDA:
            noiseA =torch.randn(search.size()).cuda()
            noiseB =torch.randn(search.size()).cuda()
        else:
            noiseA =torch.randn(search.size())
            noiseB =torch.randn(search.size())

        xfA = track_model.backbone(
            search+pertA+noiseA+advs.mean(0))
        if cfg.ADJUST.ADJUST:
            xfA = track_model.neck(xfA)
        clsA_, locA = track_model.rpn_head(zf_list, xfA)

        xfB = track_model.backbone(
            search+pertB+noiseB+advs.mean(0))
        if cfg.ADJUST.ADJUST:
            xfB = track_model.neck(xfB)
        clsB_, locB = track_model.rpn_head(zf_list, xfB)

        # get loss1
        clsA = track_model.log_softmax(clsA_)
        cls_lossA = select_cross_entropy_loss(clsA, adv_cls)

        clsB = track_model.log_softmax(clsB_)
        cls_lossB = select_cross_entropy_loss(clsB, adv_cls)

        cls_lossA.backward()
        x_gradA = torch.sign(data['pertA'].grad)

        cls_lossB.backward()
        x_gradB = torch.sign(data['pertB'].grad)

        if self.down_hill:
            pertA = pertA-self.eplison * x_gradA
            pertA = torch.clamp(pertA, -self.inta, self.inta)
        else:
            pertA = pertA+self.eplison * x_gradA
            pertA = torch.clamp(pertA, -self.inta, self.inta)

        pertB = pertB-self.eplison * x_gradB
        pertB = torch.clamp(pertB, -self.inta, self.inta)

        if self.down_hill==False and cls_lossA_prev>cls_lossA:
            self.down_hill=True

        p_searchA = search + pertA
        p_searchA = torch.clamp(p_searchA, 0, 255)
        pertA = p_searchA - search

        p_searchB = search + pertB
        p_searchB = torch.clamp(p_searchB, 0, 255)
        pertB = p_searchB - search

        scoreA = tracker._convert_score(clsA_)
        pred_bboxA = tracker._convert_bbox(locA, tracker.anchors)
        bboxA = self.pred_bbox(tracker,pred_bboxA,scoreA)
        posA = np.array([bboxA[0], bboxA[1]])+tracker.center_pos
        scoreB = tracker._convert_score(clsB_)
        pred_bboxB = tracker._convert_bbox(locB, tracker.anchors)
        bboxB = self.pred_bbox(tracker, pred_bboxB, scoreB)
        posB = np.array([bboxB[0], bboxB[1]])+tracker.center_pos
        distA = np.linalg.norm(self.target_pos-posA)
        distB = np.linalg.norm(self.target_pos-posB)

        # update
        if self.advs_init:
            advs = p_searchA
            self.advs_init=False
        elif distA<20 or distB<20:
            if distA<20:
                advs = torch.cat((advs, p_searchA), 0)
            if distB<20:
                advs = torch.cat((advs, p_searchB), 0)

        cls_lossA_prev = cls_lossA

        if torch.norm(pertA, 2)>torch.norm(pertB, 2):
            pert = pertB
        else:
            pert = pertA

        return pert, pertA, pertB, advs, cls_lossB, clsB, cls_lossA_prev

    def pred_bbox(self,tracker,pred_bbox,score):

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        w_z = tracker.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
        h_z = tracker.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(tracker.size[0] * scale_z, tracker.size[1] * scale_z)))

        # aspect ratio penalty
        r_c = change((tracker.size[0] / tracker.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                 tracker.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        obj_bbox = pred_bbox[:, best_idx]
        obj_pos = np.array([obj_bbox[0], obj_bbox[1]])

        return obj_pos