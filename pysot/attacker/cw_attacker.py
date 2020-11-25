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
from pysot.attacker.oim_attacker import OIMAttacker

class CWAttacker(OIMAttacker):
    def __init__(self,type,max_num=10,eplison=1,inta=10,lamb=0.0001,norm_type='L_2',apts_num=2,reg_type='L21'): # lamb = 0.0001
        self.type = type
        # parameters for bim
        self.eplison = 1e-2
        self.inta = inta
        self.norm_type = 'L_2'
        self.max_num = max_num
        self.lamb = lamb
        self.v_id = 0
        self.apts_num = apts_num
        self.target_traj=[]
        self.prev_delta = None
        self.tacc = False
        self.acc_iters = 0
        self.weight_eplison = 1

        self.const = 1e4
        self.boxplus = 0.5
        self.boxmul = 0.5


    def attack(self,tracker,img,prev_perts=None,weights=None,APTS=False,OPTICAL_FLOW=False,ADAPT=False,Enable_same_prev=True):
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
        diff_cls = cls[:,:,:,:,1]-cls[:,:,:,:,0]
        label_cls = diff_cls.ge(0).float()

        if self.type=='UA':
            adv_cls,same_prev = self.ua_label(tracker, scale_z, outputs)
            adv_cls = adv_cls.long()
        elif self.type=='TA':
            adv_cls,same_prev = self.ta_label(tracker,scale_z,outputs)
            adv_cls = adv_cls.long()

        max_iteration = self.max_num
        if cfg.CUDA:
            pert = torch.zeros(x_crop.size()).cuda()
            adv_cls = adv_cls.cuda()
        else:
            pert = torch.zeros(x_crop.size())

        if prev_perts is None or (same_prev==False and Enable_same_prev==True):
            if cfg.CUDA:
                prev_perts=torch.zeros(x_crop.size()).cuda()
            else:
                prev_perts=torch.zeros(x_crop.size())
        else:
            if APTS==False:
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
                return x_crop,pert_true,prev_perts,adv_img
            else:
                max_iteration = self.apts_num
                if self.tacc ==False:
                    if cfg.CUDA:
                        prev_perts = torch.zeros(x_crop.size()).cuda()
                    else:
                        prev_perts = torch.zeros(x_crop.size())

        # start attack
        losses = []
        clsloss =[]
        regloss=[]
        m=0
        optimizer = torch.optim.Adam([pert], self.eplison)
        def tan_x(x):
            x = torch.tanh(x)*self.boxmul + self.boxplus
            return x*255

        def artan_x(x):
            x = (x.float()/255)
            x = torch.from_numpy(np.arctanh((x.cpu().numpy() - self.boxplus) / self.boxmul * 0.999999))
            if cfg.CUDA:
                x = x.cuda()

            return x

        self.acc_iters += max_iteration
        # transform x_crop
        art_x_crop = artan_x(x_crop)
        if cfg.ATTACKER.SHOW:
            vis = visdom.Visdom(env='Adversarial Example Showing')
        while m<max_iteration:
            if isinstance(tracker.model.zf, list):
                zf = torch.cat(tracker.model.zf, 0)
            else:
                zf = tracker.model.zf
            zf = zf.detach()
            art_x_crop = art_x_crop.detach()
            t_pert = pert
            prev_perts = prev_perts.detach()
            adv_cls = adv_cls.detach()
            t_pert.requires_grad = True
            pert.requires_grad = True

            adv_x_crop = tan_x(art_x_crop+t_pert)
            # start attack once
            track_model = tracker.model
            zf_list = []
            if zf.shape[0] > 1:
                for i in range(0, zf.shape[0]):
                    zf_list.append(zf[i, :, :, :].resize_(1, zf.shape[1], zf.shape[2], zf.shape[3]))
            else:
                zf_list = zf

            # get feature
            xf = track_model.backbone(adv_x_crop)
            if cfg.ADJUST.ADJUST:
                xf = track_model.neck(xf)
            cls, loc = track_model.rpn_head(zf_list, xf)

            # get loss1
            cls = track_model.log_softmax(cls)
            cls_loss = select_cross_entropy_loss(cls, adv_cls)

            # regularization loss
            if self.norm_type == 'L_inf':
                reg_loss = torch.norm(adv_x_crop-x_crop, float('inf'))
            elif self.norm_type == 'L_2':
                reg_loss = torch.norm(adv_x_crop-x_crop, 2)

            total_loss = self.const * cls_loss + reg_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            m+=1
            losses.append(total_loss)
            clsloss.append(cls_loss)
            regloss.append(reg_loss)

            # disp x_crop for each iteration and the pert
            if cfg.ATTACKER.SHOW:
                vis.images(adv_x_crop,win='X_attack')
                vis.images(adv_x_crop-x_crop,win='Pert_attack')
                vis.images(x_crop, win='X_crop')
                plt.plot(losses)
                plt.plot(clsloss)
                plt.plot(regloss)
                plt.ylabel('Loss')
                plt.xlabel('Iteration')
                vis.matplot(plt,win='Loss_attack')
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
                vis.images(disp_score, win='Response_attack')

        self.opt_flow_prev_xcrop = x_crop
        if cfg.CUDA:
            prev_perts = torch.cat((prev_perts,pert),0).cuda()
        else:
            prev_perts = torch.cat((prev_perts, pert),0)
        pert_true = adv_x_crop-x_crop
        if cfg.ATTACKER.GEN_ADV_IMG:
            adv_img = tracker.get_orgimg(img,x_crop,tracker.center_pos,
                            cfg.TRACK.INSTANCE_SIZE,
                            round(s_x), tracker.channel_average)
        else:
            adv_img = None

        return adv_x_crop,pert_true,prev_perts,weights,adv_img

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls





