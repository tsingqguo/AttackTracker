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
from toolkit.tvnet_pytorch.model.net.spatial_transformer import spatial_transformer as st
import scipy.io as sio
from toolkit.tvnet_pytorch.model import network
from toolkit.tvnet_pytorch.train_options import arguments
from math import cos,sin,pi

class BIMAttacker():
    def __init__(self,type,max_num=10,eplison=1,inta=10,lamb=1,norm_type='L_inf'):

        self.type = type
        # parameters for bim
        self.eplison = eplison
        self.inta = inta
        self.norm_type = norm_type
        self.max_num = max_num
        self.lamb = lamb
        self.v_id = 0
        self.st = st()
        self.apts_num = 2
        self.target_traj=[]
        self.prev_delta = None
        self.opt_flow_model = None
        self.opt_flow_mag_thr=200000
        self.opt_flow_u = None
        self.opt_flow_v = None
        self.opt_flow_mag = None
        self.opt_flow_prev_xcrop = None

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
                print('starting paragation.')
                self.load_optical_flow(x_crop,self.opt_flow_prev_xcrop)
                if self.opt_flow_mag<self.opt_flow_mag_thr:
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
            x_crop,pert,loss,update_cls = self.bim_once(tracker,data)
            losses.append(loss)
            m+=1
            # disp x_crop for each iteration and the pert
            if cfg.ATTACKER.SHOW:
                vis.images(x_crop,win='X_attack')
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

        return x_crop,pert,adv_img

    def ua_label(self, tracker, scale_z, outputs,APTS):

        score = tracker._convert_score(outputs['cls'])
        pred_bbox = tracker._convert_bbox(outputs['loc'], tracker.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

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
        obj_pos  = np.array([obj_bbox[0],obj_bbox[1]])

        b, a2, h, w = outputs['cls'].size()
        size = tracker.size
        template_size = np.array([size[0] + cfg.TRACK.CONTEXT_AMOUNT * (size[0] + size[1]), \
                                  size[1] + cfg.TRACK.CONTEXT_AMOUNT * (size[0] + size[1])])
        context_size = template_size * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)

        same_prev = False
        # validate delta meets the requirement of obj
        if self.prev_delta is not None:
            diff_pos = np.abs(self.prev_delta - obj_pos)
            if (size[0]//2<diff_pos[0] and diff_pos[0]<context_size[0]//2) \
                    and (size[1]//2<diff_pos[1] and diff_pos[1]<context_size[1]//2):
                delta = self.prev_delta
                same_prev = True
            else:
                delta = []
                delta.append(random.choice((1, -1)) * random.randint(size[0] // 2, context_size[0] // 2))
                delta.append(random.choice((1, -1)) * random.randint(size[1] // 2, context_size[1] // 2))
                delta = obj_pos + np.array(delta)
                self.prev_delta = delta
        else:
            delta = []
            delta.append(random.choice((1, -1)) * random.randint(size[0] // 2, context_size[0] // 2-size[0] // 2))
            delta.append(random.choice((1, -1)) * random.randint(size[1] // 2, context_size[1] // 2-size[1] // 2))
            delta = obj_pos + np.array(delta)
            self.prev_delta = delta

        desired_pos = context_size/2 + delta
        upbound_pos = context_size
        downbound_pos = np.array([0, 0])
        desired_pos[0] = np.clip(desired_pos[0], downbound_pos[0], upbound_pos[0])
        desired_pos[1] = np.clip(desired_pos[1], downbound_pos[1], upbound_pos[1])
        if cfg.ATTACKER.EVAL:
            desired_pos_abs = tracker.center_pos+delta
            tpos = []
            tpos.append(desired_pos_abs[0])
            tpos.append(desired_pos_abs[1])
            self.target_traj.append(tpos)

        desired_bbox = self._get_bbox(desired_pos, size)

        anchor_target = AnchorTarget()

        results = anchor_target(desired_bbox, w)
        overlap = results[3]
        max_val = np.max(overlap)
        max_pos = np.where(overlap == max_val)
        adv_cls = torch.zeros(results[0].shape)
        adv_cls[:, max_pos[1], max_pos[2]] = 1
        adv_cls = adv_cls.view(1, adv_cls.shape[0], adv_cls.shape[1], adv_cls.shape[2])

        return adv_cls,same_prev

    def ta_label(self, tracker, scale_z, outputs):

        b, a2, h, w = outputs['cls'].size()
        center_pos = tracker.center_pos
        size = tracker.size
        desired_pos = np.array(self.target_traj[self.v_id])
        same_prev = False
        if self.v_id > 1:
            prev_desired_pos = np.array(self.target_traj[self.v_id-1])
            if np.linalg.norm(prev_desired_pos-desired_pos)<5:
                same_prev = True
        template_size = np.array([size[0]+cfg.TRACK.CONTEXT_AMOUNT * (size[0]+ size[1]), \
                                 size[1]+cfg.TRACK.CONTEXT_AMOUNT * (size[0] + size[1])])
        context_size = template_size * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        delta = desired_pos-center_pos
        desired_pos =delta+context_size/2
        upbound_pos = context_size
        downbound_pos = np.array([0,0])
        desired_pos[0] = np.clip(desired_pos[0],downbound_pos[0],upbound_pos[0])
        desired_pos[1] = np.clip(desired_pos[1],downbound_pos[1],upbound_pos[1])
        desired_bbox = self._get_bbox(desired_pos,size)

        anchor_target = AnchorTarget()

        results = anchor_target(desired_bbox, w)
        overlap = results[3]
        max_val=np.max(overlap)
        max_pos = np.where(overlap==max_val)
        adv_cls = torch.zeros(results[0].shape)
        adv_cls[:,max_pos[1],max_pos[2]]=1
        adv_cls = adv_cls.view(1,adv_cls.shape[0],adv_cls.shape[1],adv_cls.shape[2])

        return adv_cls,same_prev

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def bim_once(self,tracker,data):

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

        track_model = tracker.model

        zf_list=[]
        if zf.shape[0]>1:
            for i in range(0,zf.shape[0]):
                zf_list.append(zf[i,:,:,:].resize_(1,zf.shape[1],zf.shape[2],zf.shape[3]))
        else:
            zf_list = zf

        # get feature
        xf = track_model.backbone(search)
        if cfg.ADJUST.ADJUST:
            xf = track_model.neck(xf)
        cls, loc = track_model.rpn_head(zf_list, xf)

        # get loss1
        cls = track_model.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, adv_cls)

        total_loss = cls_loss
        total_loss.backward()

        x_grad = -data['search'].grad
        adv_x = search
        if self.norm_type=='L_inf':
            x_grad = torch.sign(x_grad)
            adv_x = adv_x+self.eplison*x_grad
            pert = adv_x-search
            pert = torch.clamp(pert,-self.inta,self.inta)
        elif self.norm_type=='L_1':
            x_grad = x_grad/torch.norm(x_grad,1)
            adv_x = adv_x+self.eplison*x_grad
            pert = adv_x-search
            pert = torch.clamp(pert/np.linalg.norm(pert,1),-self.inta,self.inta)
        elif self.norm_type=='L_2':
            x_grad = x_grad/torch.norm(x_grad,2)
            adv_x = adv_x+self.eplison*x_grad
            pert = adv_x-search
            pert = torch.clamp(pert/np.linalg.norm(pert,2),-self.inta,self.inta)
        # elif self.norm_type=='Momen':
        #     momen = self.lamb*momen+x_grad/np.linalg.norm(x_grad,1)
        #     adv_x = adv_x+self.eplison*np.sign(momen)
        #     pert = adv_x-search
        #     pert = torch.clamp(pert,-self.inta,self.inta)

        p_search = search+pert
        p_search = torch.clamp(p_search,0,255)
        pert = p_search-search

        return p_search, pert,total_loss,cls


    def target_traj_gen(self,init_rect,vid_h,vid_w,vid_l):

        target_traj = []
        pos = []
        pos.append(init_rect[0]+init_rect[2]/2)
        pos.append(init_rect[1]+init_rect[3]/2)
        target_traj.append(pos)
        for i in range(0,vid_l-1):
            tpos = []
            if i%50==0:
                deltay =random.randint(-10,10)
                deltax =random.randint(-10,10)
            elif i%10==0:
                deltay =random.randint(0,1)
                deltax =random.randint(0,1)
            elif i%5==0:
                deltay =random.randint(-1,0)
                deltax =random.randint(-1,0)
            tpos.append(np.clip(target_traj[i][0]+deltax,0,vid_w-1))
            tpos.append(np.clip(target_traj[i][1]+deltay,0,vid_h-1))
            target_traj.append(tpos)
        self.target_traj = target_traj
        return target_traj

    def target_traj_gen_custom(self, init_rect, vid_h, vid_w, vid_l, type=2):
        '''
        type 1 : ellipse
        type 2 : rectangle
        type 3 : triangle
        '''
        target_traj = []
        pos = []
        pos.append(init_rect[0] + init_rect[2] / 2)
        pos.append(init_rect[1] + init_rect[3] / 2)
        target_traj.append(pos)

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

        r = min(vid_w, vid_h) / 2

        for i in range(0, vid_l - 1):
            tpos = []
            if type == 1:
                t = 4 * pi * i / vid_l
                x, y = ellipse(t, r / (pi * 8))
            if type == 2:
                t = 4. * i / vid_l
                x, y = rectangle(t, r / 2)

            if type == 3:
                t = 3. * i / vid_l
                x, y = triangle(t, r / 2)

            tpos.append(np.clip(x + init_rect[0], 0, vid_w - 1))
            tpos.append(np.clip(y + init_rect[1], 0, vid_h - 1))
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
        cx, cy = center_pos* scale_z
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def warp_pert(self,pert):

        u,v=self.opt_flow_u,self.opt_flow_v
        u_flat = u.view(pert.size(0), 1, pert.size(2) * pert.size(3))
        v_flat = v.view(pert.size(0), 1, pert.size(2) * pert.size(3))
        pert_ = self.warp_image(pert,u_flat,v_flat)
        pert_ = pert_.view(pert.size())
        if cfg.ATTACKER.SHOW:
            vis = visdom.Visdom(env='Adversarial Example Showing')
            vis.images(pert_, win='X_prev')
            vis.images(self.warp_image(pert_,u_flat,v_flat).view(pert_.size()), win='pert_')
        return pert_

    def warp_image(self, x, u, v):
        assert len(x.size()) == 4
        assert len(u.size()) == 3
        assert len(v.size()) == 3

        u = u / x.size(3) * 2
        v = v / x.size(2) * 2
        if cfg.CUDA:
            theta = torch.cat((u, v), dim=1).cuda().float()
        else:
            theta = torch.cat((u, v), dim=1).float()

        trans_image = self.st(x, theta, x.size()[2:])

        return trans_image

    def load_optical_flow(self,X_,X_prev):
        if self.opt_flow_model is None:
            params = arguments().parse()
            params.data_size = [1, 3, X_.shape[2], X_.shape[3]]
            if cfg.CUDA:
                self.opt_flow_model = network.model(params).cuda()
            else:
                self.opt_flow_model = network.model(params)
        with torch.no_grad():
            u,v,_= self.opt_flow_model.forward(X_, X_prev, need_result=True)
        self.opt_flow_u,self.opt_flow_v = u,v
        u_flat = u.view(u.size(0), 1, u.size(2) * u.size(3))
        v_flat = v.view(v.size(0), 1, v.size(2) * v.size(3))
        self.opt_flow_mag = torch.sqrt(u_flat.pow(2) + v_flat.pow(2)).sum()
        if cfg.ATTACKER.SHOW:
            vis = visdom.Visdom(env='Adversarial Example Showing')
            vis.images(X_prev, win='X_prev')
            vis.images(self.warp_image(X_prev,u_flat,v_flat).view(X_prev.size()), win='X_prev')


