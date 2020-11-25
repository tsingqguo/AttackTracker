# Copyright (c) SenseTime. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import visdom
from pysot.core.config import cfg
import matplotlib.pyplot as plt
from pysot.models.loss import select_cross_entropy_loss
from toolkit.tvnet_pytorch.model.net.spatial_transformer import spatial_transformer as st
from pysot.attacker.oim_attacker import OIMAttacker

class MIFGSMAttacker(OIMAttacker):
    def __init__(self,type,max_num=10,eplison=1,inta=10,lamb=0.0001, norm_type='Momen', apts_num=2,reg_type='L21',accframes=30):

        self.type = type
        self.eplison = eplison
        self.inta = inta
        self.norm_type = 'Momen' #norm_type
        self.max_num = max_num
        self.v_id = 0
        #self.st = st()
        self.apts_num = apts_num
        self.target_traj=[]
        self.prev_delta = None
        self.tacc = False
        self.lamb_momen = 1
        self.lamb = 0 # remove the L2,1 regularization
        self.reg_type =reg_type
        self.acc_iters = 0
        self.weight_eplison = 1
        self.accframes = accframes

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

        if cfg.ATTACKER.SHOW:
            vis = visdom.Visdom(env='Adversarial Example Showing') #(server='172.28.144.132',port=8022,env='Adversarial Example Showing')
            vis.images(x_crop,win='X_org')
            vis.images(adv_cls.permute(1,0,2,3), win='adv_cls')

        # start attack
        losses = []
        m=0
        if self.norm_type == 'Momen':
            if cfg.CUDA:
                momen = torch.zeros(x_crop.size()).cuda()
            else:
                momen = torch.zeros(x_crop.size())
        self.acc_iters += max_iteration
        while m<max_iteration:
            if isinstance(tracker.model.zf, list):
                zf = torch.cat(tracker.model.zf, 0)
            else:
                zf = tracker.model.zf

            data = {
                'template_zf': zf.detach(),
                'search': x_crop.detach(),
                'pert':pert.detach(),
                'prev_perts':prev_perts.detach(),
                'label_cls': label_cls.detach(),
                'adv_cls': adv_cls.detach(),
                'momen':momen.detach()
            }
            data['pert'].requires_grad = True
            pert,loss,update_cls,momen = self.oim_once(tracker,data)
            losses.append(loss)
            m+=1
            # disp x_crop for each iteration and the pert
            if cfg.ATTACKER.SHOW:
                x_crop_t = x_crop + prev_perts.sum(0)+pert
                x_crop_t = torch.clamp(x_crop_t, 0, 255)
                vis.images(x_crop_t,win='X_attack')
                vis.images(pert,win='Pert_attack')
                plt.plot(losses)
                plt.ylabel('Loss')
                plt.xlabel('Iteration')
                vis.matplot(plt,win='Loss_attack')
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
            prev_perts = torch.cat((prev_perts,pert),0).cuda()
        else:
            prev_perts = torch.cat((prev_perts, pert),0)
        pert_sum = prev_perts.sum(0)
        adv_x_crop = x_crop + pert_sum
        adv_x_crop = torch.clamp(adv_x_crop, 0, 255)
        pert_true = adv_x_crop-x_crop
        if cfg.ATTACKER.GEN_ADV_IMG:
            adv_img = tracker.get_orgimg(img,x_crop,tracker.center_pos,
                            cfg.TRACK.INSTANCE_SIZE,
                            round(s_x), tracker.channel_average)
        else:
            adv_img = None

        return adv_x_crop,pert_true,prev_perts,weights,adv_img

    def oim_once(self,tracker,data):

        if cfg.CUDA:
            zf = data['template_zf'].cuda()
            search = data['search'].cuda()
            pert = data['pert'].cuda()
            prev_perts = data['prev_perts'].cuda()
            adv_cls = data['adv_cls'].cuda()
            momen = data['momen'].cuda()
        else:
            zf = data['template_zf']
            search = data['search']
            pert = data['pert']
            prev_perts = data['prev_perts']
            adv_cls = data['adv_cls']
            momen = data['momen']

        track_model = tracker.model

        zf_list=[]
        if zf.shape[0]>1:
            for i in range(0,zf.shape[0]):
                zf_list.append(zf[i,:,:,:].resize_(1,zf.shape[1],zf.shape[2],zf.shape[3]))
        else:
            zf_list = zf

        # get feature
        xf = track_model.backbone(search+pert+prev_perts.sum(0).view(1,prev_perts.shape[1],prev_perts.shape[2],prev_perts.shape[3]))
        if cfg.ADJUST.ADJUST:
            xf = track_model.neck(xf)
        cls, loc = track_model.rpn_head(zf_list, xf)

        # get loss1
        cls = track_model.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, adv_cls)
        # regularization loss
        t_prev_perts = prev_perts.view(prev_perts.shape[0]*prev_perts.shape[1],prev_perts.shape[2]*prev_perts.shape[3])
        reg_loss = torch.norm(t_prev_perts,2,1).sum()+torch.norm(pert,2)

        total_loss = cls_loss+self.lamb*reg_loss
        total_loss.backward()

        x_grad = -data['pert'].grad
        adv_x = search
        if self.norm_type=='L_inf':
            x_grad = torch.sign(x_grad)
            adv_x = adv_x+pert+self.eplison*x_grad
            pert = adv_x-search
            pert = torch.clamp(pert,-self.inta,self.inta)
        elif self.norm_type=='L_1':
            x_grad = x_grad/torch.norm(x_grad,1)
            adv_x = adv_x+pert+self.eplison*x_grad
            pert = adv_x-search
            pert = torch.clamp(pert/np.linalg.norm(pert,1),-self.inta,self.inta)
        elif self.norm_type=='L_2':
            x_grad = x_grad/torch.norm(x_grad,2)
            adv_x = adv_x+pert+self.eplison*x_grad
            pert = adv_x-search
            pert = torch.clamp(pert/np.linalg.norm(pert,2),-self.inta,self.inta)
        elif self.norm_type=='Momen':
            momen = self.lamb_momen*momen+x_grad/torch.norm(x_grad,1)
            adv_x = adv_x+pert+self.eplison*torch.sign(momen)
            pert = adv_x-search
            pert = torch.clamp(pert,-self.inta,self.inta)

        p_search = search+pert+prev_perts.sum(0)
        p_search = torch.clamp(p_search,0,255)
        pert = p_search-search-prev_perts.sum(0)

        return pert,total_loss,cls,momen