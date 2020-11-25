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
from pysot.utils.bbox import center2corner, Center
from toolkit.tvnet_pytorch.model import network
from toolkit.tvnet_pytorch.train_options import arguments
from pysot.attacker.oim_attacker import OIMAttacker

class CWAttacker(OIMAttacker):
    def __init__(self,type,max_num=10,eplison=1,inta=10,lamb=0.0001,norm_type='L_inf'):

        self.type = type
        # parameters for bim
        self.eplison = eplison
        self.inta = inta
        self.norm_type = norm_type
        self.max_num = max_num
        self.lamb = lamb
        self.v_id = 0
        self.apts_num = 2
        self.target_traj=[]
        self.prev_delta = None
        self.confidence = 1.

    def attack(self,tracker,img,pert_sum=None,APTS=False,OPTICAL_FLOW=False,ADAPT=False):
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

        if cfg.ATTACKER.ATTACK_TYPE=='UA':
            adv_cls = self.ua_label(tracker, scale_z, outputs,APTS).long()
        elif cfg.ATTACKER.ATTACK_TYPE=='TA':
            adv_cls = self.ta_label(tracker,scale_z,outputs).long()

        max_iteration = self.max_num
        if pert_sum is None:
            if cfg.CUDA:
                pert_sum = torch.zeros(x_crop.size()).cuda()
            else:
                pert_sum = torch.zeros(x_crop.size())
        else:
            if ADAPT:
                self.load_optical_flow(x_crop,self.opt_flow_prev_xcrop)
                if self.opt_flow_mag>self.opt_flow_mag_thr:
                    if OPTICAL_FLOW:
                        pert_sum = self.warp_pert(pert_sum)
                    x_crop = x_crop + pert_sum
                    x_crop = torch.clamp(x_crop, 0, 255)

                    if APTS == False:
                        if cfg.ATTACKER.GEN_ADV_IMG:
                            adv_img = tracker.get_orgimg(img, x_crop, tracker.center_pos,
                                                         cfg.TRACK.INSTANCE_SIZE,
                                                         round(s_x), tracker.channel_average)
                        else:
                            adv_img = None
                        return x_crop, pert_sum, adv_img
                    else:
                        max_iteration = self.apts_num

                else:
                    if cfg.CUDA:
                        pert_sum = torch.zeros(x_crop.size()).cuda()
                    else:
                        pert_sum = torch.zeros(x_crop.size())
            else:
                if OPTICAL_FLOW:
                    self.load_optical_flow(x_crop,self.opt_flow_prev_xcrop)
                    pert_sum = self.warp_pert(pert_sum)
                x_crop = x_crop + pert_sum
                x_crop = torch.clamp(x_crop, 0, 255)

                if APTS==False:
                    if cfg.ATTACKER.GEN_ADV_IMG:
                        adv_img = tracker.get_orgimg(img, x_crop, tracker.center_pos,
                                                     cfg.TRACK.INSTANCE_SIZE,
                                                     round(s_x), tracker.channel_average)
                    else:
                        adv_img = None
                    return x_crop,pert_sum,adv_img
                else:
                    max_iteration = self.apts_num


        if cfg.ATTACKER.SHOW:
            vis = visdom.Visdom(env='Adversarial Example Showing') #(server='172.28.144.132',port=8022,env='Adversarial Example Showing')
            vis.images(x_crop,win='X_org')
            vis.images(adv_cls.permute(1,0,2,3), win='adv_cls')

        # start attack
        losses = []
        m=0

        def atanh(img):
            img = img/255.
            img = 2.* img - 1.
            img = img*(1-1e-6)
            return 0.5*torch.log((1.+img)/(1.-img))
        w = atanh(x_crop)
        optimizer = torch.optim.Adam([w],lr=0.001)

        while m<max_iteration:
            if isinstance(tracker.model.zf, list):
                zf = torch.cat(tracker.model.zf, 0)
            else:
                zf = tracker.model.zf

            data = {
                'template_zf': zf.detach(),
                'search': x_crop.detach(),
                'label_cls': label_cls.detach(),
                'adv_cls': adv_cls.detach()
            }
            data['search'].requires_grad = True

            x_crop_for_show,pert,loss,_ = self.attack_once(tracker,data, w, optimizer)

            losses.append(loss)
            m+=1
            # disp x_crop for each iteration and the pert
            if cfg.ATTACKER.SHOW:
                vis.images(x_crop_for_show,win='X_attack')
                vis.images(pert,win='Pert_attack')
                plt.plot(losses)
                plt.ylabel('Loss')
                plt.xlabel('Iteration')
                vis.matplot(plt,win='Loss_attack')
                # validate the score
                outputs = tracker.model.track(x_crop)
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

        if cfg.ATTACKER.GEN_ADV_IMG:
            adv_img = tracker.get_orgimg(img,x_crop,tracker.center_pos,
                            cfg.TRACK.INSTANCE_SIZE,
                            round(s_x), tracker.channel_average)
        else:
            adv_img = None

        self.opt_flow_prev_xcrop = x_crop

        return x_crop_for_show,pert,adv_img

    def oim_once(self,tracker,data, w, optimizer):

        if cfg.CUDA:
            zf = data['template_zf'].cuda()
            origin = data['search'].cuda()
            label_cls = data['label_cls'].cuda()
            adv_cls = data['adv_cls'].cuda()
        else:
            zf = data['template_zf']
            origin = data['search']
            label_cls = data['label_cls']
            adv_cls = data['adv_cls']

        # search : original data

        track_model = tracker.model

        zf_list=[]
        if zf.shape[0]>1:
            for i in range(0,zf.shape[0]):
                zf_list.append(zf[i,:,:,:].resize_(1,zf.shape[1],zf.shape[2],zf.shape[3]))
        else:
            zf_list = zf

        # get feature
        search = 255./2*(torch.nn.functional.tanh(w) + 1)

        xf = track_model.backbone(search)
        if cfg.ADJUST.ADJUST:
            xf = track_model.neck(xf)
        cls, _ = track_model.rpn_head(zf_list, xf)

        # get loss1
        cls = track_model.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, adv_cls)

        total_loss = cls_loss

        if self.norm_type=='L_1':
            l1_loss = torch.nn.functional.l1_loss(search,origin)
            total_loss = self.confidence*total_loss + l1_loss
        elif self.norm_type=='L_2':
            l2_loss = torch.nn.functional.mse_loss(search,origin)
            total_loss = self.confidence*total_loss + l2_loss
        elif self.norm_type == 'L_inf':
            linf_loss = torch.norm(search-origin,float('inf'))
            total_loss = self.confidence*total_loss + linf_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        w = torch.clamp(w, 0., 1.)

        p_search = 255./2*(torch.nn.functional.tanh(w) + 1)
        pert = p_search - origin

        return p_search.detach(), pert.detach(),total_loss,cls



