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
from extern.pytracking.features.preprocessing import numpy_to_torch
from extern.pytracking.libs import complex, fourier, TensorList, dcf
from extern.pytracking.features.preprocessing import sample_patch
from math import cos, sin, pi
import os
import cv2


class OIMAttackerECO():
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
        self.weights = []
        self.weight_eplison = 1
        self.lamb_momen = 1
        self.target_pos = []
        self.accframes = accframes

    def save_tensorasimg(self, path, var):
        np_var = torch.squeeze(var).detach().cpu()
        if len(np_var.shape) == 2:
            np_var = np_var.view(1, np_var.shape[0], np_var.shape[1])
        np_var = np_var.numpy()
        np_var = np.transpose(np_var, (1, 2, 0))
        cv2.imwrite(path, np_var, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    def attack(self, tracker, img, prev_perts=None, weights=None,APTS=False, OPTICAL_FLOW=False, ADAPT=False, Enable_same_prev=True):
        """
        args:
            tracker, img(np.ndarray): BGR image
        return:
            adversirial image
        """
        # Convert image
        img = numpy_to_torch(img)
        
        # Get image patches
        x_crop = img

        if self.type == 'TA':
            adv_cls, same_prev = self.ta_label(tracker, scale_z, outputs)
            adv_cls = adv_cls.long()
        elif self.type == 'UA':
            adv_cls, same_prev = self.ua_label(tracker,img)
            adv_cls = adv_cls.long()

        max_iteration = self.max_num

        if cfg.CUDA:
            pert = torch.zeros(x_crop.size()).cuda()
        else:
            pert = torch.zeros(x_crop.size())

        if prev_perts is None or (same_prev == False and Enable_same_prev == True):
            if cfg.CUDA:
                prev_perts = torch.zeros(x_crop.size()).cuda()
                weights = torch.ones(x_crop.size()).cuda()
            else:
                prev_perts = torch.zeros(x_crop.size())
                weights = torch.ones(x_crop.size())
        else:
            if APTS == False:
                if self.reg_type == 'weighted':
                    pert_sum = torch.mul(weights, prev_perts).sum(0)
                else:
                    pert_sum = prev_perts.sum(0)
                adv_x_crop = x_crop + pert_sum
                adv_x_crop = torch.clamp(adv_x_crop, 0, 255)
                pert_true = adv_x_crop - x_crop

                adv_img = adv_x_crop

                if cfg.ATTACKER.SAVE_META:
                    # validate the score
                    outputs = tracker.model.track(adv_x_crop)
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
                    #
                    res_path = os.path.join(self.meta_path, str(self.v_id) + '_xcrop.jpg')
                    self.save_tensorasimg(res_path, x_crop)
                    res_path = os.path.join(self.meta_path, str(self.v_id) + '_adv_xcrop.jpg')
                    self.save_tensorasimg(res_path, adv_x_crop)
                    res_path = os.path.join(self.meta_path, str(self.v_id) + '_pert.jpg')
                    self.save_tensorasimg(res_path, pert_true)
                    res_path = os.path.join(self.meta_path, str(self.v_id) + '_pert_sum.jpg')
                    self.save_tensorasimg(res_path, pert_true)
                    for i in range(disp_score.shape[0]):
                        res_path = os.path.join(self.meta_path, str(self.v_id) + '_score' + str(i) + '.jpg')
                        self.save_tensorasimg(res_path, disp_score[i, :, :, :])

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

        if cfg.CUDA:
            momen = torch.zeros(x_crop.size()).cuda()
        else:
            momen = torch.zeros(x_crop.size())

        self.acc_iters += max_iteration
        while m < max_iteration:

            data = {
                'search': x_crop.detach(),
                'pert': pert.detach(),
                'prev_perts': prev_perts.detach(),
                'adv_cls': adv_cls.detach(),
                'weights':weights.detach(),
                'momen':momen.detach()
            }

            data['pert'].requires_grad = True
            #data['weights'].requires_grad = True
            pert, loss, update_cls, momen, weights = self.oim_once(tracker, data)
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

        if cfg.ATTACKER.SAVE_META:
            if self.reg_type == 'weighted':
                pert_sum = torch.mul(weights, prev_perts).sum(0)
            else:
                pert_sum = prev_perts.sum(0)
            x_crop_t = x_crop + pert_sum + pert
            x_crop_t = torch.clamp(x_crop_t, 0, 255)
            # validate the score
            outputs = tracker.model.track(x_crop_t)
            disp_score = self.log_softmax(outputs['cls'].cpu())
            disp_score = self.norm_score(disp_score)
            # original score
            outputs = tracker.model.track(x_crop)
            disp_score_org = self.log_softmax(outputs['cls'].cpu())
            disp_score_org = self.norm_score(disp_score_org)
            #
            res_path = os.path.join(self.meta_path, str(self.v_id) + '_xcrop.jpg')
            self.save_tensorasimg(res_path, x_crop)
            res_path = os.path.join(self.meta_path, str(self.v_id) + '_adv_xcrop.jpg')
            self.save_tensorasimg(res_path, x_crop_t)
            res_path = os.path.join(self.meta_path, str(self.v_id) + '_pert.jpg')
            self.save_tensorasimg(res_path, pert)
            res_path = os.path.join(self.meta_path, str(self.v_id) + '_pert_sum.jpg')
            self.save_tensorasimg(res_path, pert_sum)
            for i in range(disp_score.shape[0]):
                res_path = os.path.join(self.meta_path, str(self.v_id) + '_score' + str(i) + '.jpg')
                res_path_org = os.path.join(self.meta_path, str(self.v_id) + '_score_org' + str(i) + '.jpg')
                self.save_tensorasimg(res_path, disp_score[i, :, :, :])
                self.save_tensorasimg(res_path_org, disp_score_org[i, :, :, :])

            res_path = os.path.join(self.meta_path, str(self.v_id) + '_loss.txt')
            with open(res_path, 'w') as f:
                for x in losses:
                    f.write(str(x.cpu().detach().numpy()) + '\n')

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

        if prev_perts.shape[0]>self.accframes:
            prev_perts = prev_perts[-self.accframes:,:,:,:]
            weights = weights[-self.accframes:,:,:,:]

        adv_img = adv_x_crop

        return adv_x_crop, pert_true, prev_perts, weights, adv_img

    def ua_label(self, tracker,img):
        
        ample_pos = tracker.pos.round()
        sample_scales = tracker.target_scale * tracker.params.scale_factors
        test_xf = tracker.extract_fourier_sample(img, tracker.pos, sample_scales, tracker.img_sample_sz)
        # Compute scores
        sf = tracker.apply_filter(test_xf)
        translation_vec, scale_ind, score_maps = tracker.localize_target(sf)
        adv_cls = sf
        same_prev = True

        return adv_cls,same_prev

    def ta_label(self, tracker, scale_z, outputs):

        b, a2, h, w = outputs['cls'].size()
        center_pos = tracker.center_pos
        size = tracker.size
        desired_pos = np.array(self.target_traj[self.v_id])
        same_prev = False
        if self.v_id > 1:
            prev_desired_pos = np.array(self.target_traj[self.v_id - 1])
            if np.linalg.norm(prev_desired_pos - desired_pos) < 5:
                same_prev = True
        template_size = np.array([size[0] + cfg.TRACK.CONTEXT_AMOUNT * (size[0] + size[1]), \
                                  size[1] + cfg.TRACK.CONTEXT_AMOUNT * (size[0] + size[1])])
        context_size = template_size * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        delta = desired_pos - center_pos
        desired_pos = delta + context_size / 2
        upbound_pos = context_size
        downbound_pos = np.array([0, 0])
        desired_pos[0] = np.clip(desired_pos[0], downbound_pos[0], upbound_pos[0])
        desired_pos[1] = np.clip(desired_pos[1], downbound_pos[1], upbound_pos[1])
        desired_bbox = self._get_bbox(desired_pos, size)

        anchor_target = AnchorTarget()

        results = anchor_target(desired_bbox, w)
        overlap = results[3]
        max_val = np.max(overlap)
        max_pos = np.where(overlap == max_val)
        adv_cls = torch.zeros(results[0].shape)
        adv_cls[:, max_pos[1], max_pos[2]] = 1
        adv_cls = adv_cls.view(1, adv_cls.shape[0], adv_cls.shape[1], adv_cls.shape[2])

        self.target_pos = desired_pos

        return adv_cls, same_prev

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def oim_once(self, tracker, data):
        if cfg.CUDA:
            search = data['search'].cuda()
            pert = data['pert'].cuda()
            prev_perts = data['prev_perts'].cuda()
            adv_cls = data['adv_cls'].cuda()
            weights = data['weights'].cuda()
            momen = data['momen'].cuda()
        else:
            search = data['search']
            pert = data['pert']
            prev_perts = data['prev_perts']
            adv_cls = data['adv_cls']
            weights = data['weights']
            momen = data['momen']

        # get feature
        if self.reg_type == 'weighted':
            pert_sum = torch.mul(weights, prev_perts).sum(0)
        else:
            pert_sum = prev_perts.sum(0)

        # ------- ECO Score Map ------- # 
        # Get sample
        sample_pos = tracker.pos.round()
        sample_scales = tracker.target_scale * tracker.params.scale_factors
        img = search + pert + pert_sum.view(1, prev_perts.shape[1], prev_perts.shape[2], prev_perts.shape[3])
        sample_scale = sample_scales[0]
        im_patch = sample_patch(img, sample_pos, sample_scale*tracker.img_sample_sz, tracker.img_sample_sz)
        
        # extraction features        
        x = TensorList([f.get_feature(im_patch) for f in self.features]).unroll()

        test_xf = tracker.preprocess_sample(tracker.project_sample(x))
        # Compute scores
        sf = complex.mult(tracker.filter, test_xf).sum(1, keepdim=True)
        #score_maps = fourier.sample_fs(fourier.sum_fs(sf), tracker.output_sz)
        cls = sf

        # ------- End ECO Score Map ------- # 
        criterion = torch.nn.MSELoss(size_average=False).cuda()
        
        # get loss1
        #print("cls",cls.size())
        #print("adv_cls",adv_cls.size())

        cls_loss = 0.
        for i in range(len(sf)): 
            sf[i].requires_grad = True
            #test_xf[i].requires_grad = True
            cls_loss += criterion(cls[i],adv_cls[i])

        # regularization loss
        if cfg.CUDA:
            c_prev_perts = torch.cat((prev_perts, pert), 0).cuda()
            #c_weights = torch.cat((weights, torch.ones(pert.size()).cuda()), 0).cuda()
        else:
            c_prev_perts = torch.cat((prev_perts, pert), 0)
            #c_weights = torch.cat((weights, torch.ones(pert.size())), 0)

        t_prev_perts = c_prev_perts.view(c_prev_perts.shape[0]*c_prev_perts.shape[1],
                                         c_prev_perts.shape[2] * c_prev_perts.shape[3])
        #t_weights = c_weights.view(c_weights.shape[0],
        #  c_weights.shape[1] * c_weights.shape[2] * c_weights.shape[3])

        if self.reg_type == 'L21':
            reg_loss = torch.norm(t_prev_perts, 2, 1).sum()  # +torch.norm(pert,2)
        elif self.reg_type == 'L2':
            reg_loss = torch.norm(t_prev_perts, 2)
        elif self.reg_type == 'weighted':
            reg_loss = torch.norm(t_prev_perts, 2, 1).sum()
        else:
            reg_loss = 0.

        total_loss = cls_loss #+ self.lamb * reg_loss
        total_loss.backward()

        if self.type == "TA":
            x_grad = -data['pert'].grad
        elif self.type == "UA":
            x_grad = data['pert'].grad

        print("x_grad",x_grad)

        if self.reg_type == 'weighted':
            pert_sum = torch.mul(weights, prev_perts).sum(0)

        adv_x = search
        if self.norm_type == 'L_inf':
            x_grad = torch.sign(x_grad)
            adv_x = adv_x + pert+ pert_sum + self.eplison * x_grad
            pert = adv_x - search-pert_sum
            pert = torch.clamp(pert, -self.inta, self.inta)
        elif self.norm_type == 'L_1':
            x_grad = x_grad / torch.norm(x_grad, 1)
            adv_x = adv_x + pert + pert_sum  + self.eplison * x_grad
            pert = adv_x - search-pert_sum
            pert = torch.clamp(pert / torch.norm(pert, 1), -self.inta, self.inta)
        elif self.norm_type == 'L_2':
            x_grad = x_grad / torch.norm(x_grad, 2)
            adv_x = adv_x + pert + pert_sum  + self.eplison * x_grad
            pert = adv_x - search-pert_sum
            pert = torch.clamp(pert / torch.norm(pert, 2), -self.inta, self.inta)
        elif self.norm_type=='Momen':
            momen = self.lamb_momen*momen+x_grad/torch.norm(x_grad,1)
            adv_x = adv_x + pert + pert_sum + self.eplison*torch.sign(momen)
            pert = adv_x-search-pert_sum
            pert = torch.clamp(pert,-self.inta,self.inta)

        p_search = search + pert + pert_sum
        p_search = torch.clamp(p_search, 0, 255)
        pert = p_search - search - prev_perts.sum(0)

        return pert, total_loss, cls, momen, weights

    def target_traj_gen(self, init_rect, vid_h, vid_w, vid_l):

        target_traj = []
        pos = []
        pos.append(init_rect[0] + init_rect[2] / 2)
        pos.append(init_rect[1] + init_rect[3] / 2)
        target_traj.append(pos)
        for i in range(0, vid_l - 1):
            tpos = []
            if i % 50 == 0:
                deltay = random.randint(-10, 10)
                deltax = random.randint(-10, 10)
            elif i % 10 == 0:
                deltay = random.randint(0, 1)
                deltax = random.randint(0, 1)
            elif i % 5 == 0:
                deltay = random.randint(-1, 0)
                deltax = random.randint(-1, 0)
            tpos.append(np.clip(target_traj[i][0] + deltax, 0, vid_w - 1))
            tpos.append(np.clip(target_traj[i][1] + deltay, 0, vid_h - 1))
            target_traj.append(tpos)
        self.target_traj = target_traj
        return target_traj

    def target_traj_gen_supervised(self, init_rect, vid_h, vid_w, vid_l, gt_traj):

        target_traj = []
        w, h = gt_traj[0][2], gt_traj[0][3]
        w_z = w + cfg.TRACK.CONTEXT_AMOUNT * np.sum(np.array([w, h]))
        h_z = h + cfg.TRACK.CONTEXT_AMOUNT * np.sum(np.array([w, h]))
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        deltax, deltay = -s_x / 5, -s_x / 5
        for i in range(vid_l):
            pos = []
            cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_traj[i]))
            pos.append(cx + deltax)
            pos.append(cy + deltay)
            target_traj.append(pos)
        self.target_traj = target_traj
        return target_traj

    def target_traj_gen_custom(self, init_rect, vid_h, vid_w, vid_l, type=1):
        '''
        type 1 : ellipse
        type 2 : rectangle
        type 3 : triangle
        '''
        target_traj = []
        initpos = np.array([init_rect[0] - init_rect[2] / 2, init_rect[1] - init_rect[3] / 2])
        target_traj.append(initpos)

        # initial start_point , shape of traj
        def ellipse(t, a):
            return a * t * cos(t), a * t * sin(t)

        def rectangle(t, a):
            if t < 0.5:
                return t * a, 0
            elif t >= 0.5 and t < 1:
                return 0.5 * a, -(t - 0.5) * a
            elif t >= 1 and t < 2:
                return -(t - 1) * a + 0.5 * a, -0.5 * a
            elif t >= 2 and t < 3:
                return -0.5 * a, (t - 2) * a - 0.5 * a
            else:
                return (t - 3) * a - 0.5 * a, 0.5 * a

            return 0, 0

        def triangle(t, r):
            if t < 1:
                return 0.5 * r - r * t * cos(pi / 3), -r * t * sin(pi / 3),
            elif t < 2:
                return -(t - 1) * r * cos(pi / 3), (t - 2) * r * sin(pi / 3)
            else:
                return (t - 2.5) * r, 0

        def line(t, r, theta):
            """line genetate line traj

            Arguments:
                t {float} -- time
                r {float} -- length
                theta {float} -- angle

            Returns:
                [float,float] -- position
            """
            return t * r * cos(theta), t * r * sin(theta)

        r = 2 * min(vid_w, vid_h) / 2

        for i in range(0, vid_l - 1):
            tpos = []
            if type == 1:
                t = 6 * pi * i / vid_l
                x, y = ellipse(t, r / (pi * 8))
            if type == 2:
                t = 4. * i / vid_l
                x, y = rectangle(t, r / 2)

            if type == 3:
                t = 3. * i / vid_l
                x, y = triangle(t, r / 2)

            if type == 4:
                t = 1. * i / vid_l
                x, y = line(t, r, -pi * 0.4)

            tpos.append(np.clip(x + initpos[0], 0, vid_w - 1))
            tpos.append(np.clip(y + initpos[1], 0, vid_h - 1))
            target_traj.append(tpos)
        self.target_traj = target_traj
        return target_traj

    def _get_bbox(self, center_pos, shape):
        if len(shape) == 4:
            w, h = shape[2] - shape[0], shape[3] - shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w * scale_z
        h = h * scale_z
        cx, cy = center_pos * scale_z
        bbox = center2corner(Center(cx - w / 2, cy - h / 2, w, h))
        return bbox

    def norm_score(self, score):

        disp_score = score[:, :, :, :, 1].permute(1, 0, 2, 3)
        disp_score_min = torch.min(torch.min(disp_score, 2)[0], 2)[0]
        disp_score_max = torch.max(torch.max(disp_score, 2)[0], 2)[0]
        dispmin = torch.zeros(disp_score.size())
        dispmax = torch.zeros(disp_score.size())
        for i in range(4):
            dispmin[i, :, :, :] = disp_score_min[i].repeat(1, 1, disp_score.shape[2], disp_score.shape[3])
            dispmax[i, :, :, :] = disp_score_max[i].repeat(1, 1, disp_score.shape[2], disp_score.shape[3])
        disp_score = (disp_score - dispmin) / (dispmax - dispmin) * 255

        return disp_score
