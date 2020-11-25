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
import matplotlib.pyplot as plt
from pysot.models.loss import select_cross_entropy_loss
from pysot.datasets.anchor_target_cross import CrossAnchorTarget
from pysot.utils.bbox import center2corner, Center
from math import cos,sin,pi

class CROSSOIMAttacker():
    def __init__(self,type,max_num=10,eplison=3e-1,inta=10,lamb=0.00001,norm_type='L_inf',cfg=None):

        self.type = type
        # parameters for bim
        self.eplison = eplison
        self.inta = inta
        self.norm_type = norm_type
        self.max_num = max_num
        self.lamb = lamb
        self.v_id = 0
        #self.st = st()
        self.apts_num = 2
        self.target_traj=[]
        self.prev_delta = None
        self.tacc = True
        self.cfg = cfg

    def attack(self,tracker,img,prev_perts=None,APTS=False,OPTICAL_FLOW=False,ADAPT=False,Enable_same_prev=False):
        """
        args:
            tracker, img(np.ndarray): BGR image
        return:
            adversirial image
        """
        cfg = self.cfg
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

        img = torch.from_numpy(img).float().cuda()
        img = img.permute(2,0,1).unsqueeze(0)

        if cfg.CUDA:
            pert = torch.zeros(img.size()).cuda()
        else:
            pert = torch.zeros(img.size())

        if prev_perts is None or (same_prev==False and Enable_same_prev == True):
            if cfg.CUDA:
                prev_perts=torch.zeros(img.size()).cuda()
            else:
                prev_perts=torch.zeros(img.size())
        else:
            if APTS==False:
                pert_sum = prev_perts.sum(0)
                adv_img = img + pert_sum
                adv_img = torch.clamp(adv_img, 0, 255)
                pert_true = adv_img - img

                return None,pert_true,prev_perts,adv_img
            else:
                max_iteration = self.apts_num
                if self.tacc ==False:
                    if cfg.CUDA:
                        prev_perts = torch.zeros(img.size()).cuda()
                    else:
                        prev_perts = torch.zeros(img.size())


        if cfg.ATTACKER.SHOW:
            vis = visdom.Visdom(env='Adversarial Example Showing') #(server='172.28.144.132',port=8022,env='Adversarial Example Showing')
            vis.images(img, win='img')
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
                'img': img.detach(),
                'pert':pert.detach(),
                'prev_perts':prev_perts.detach(),
                'label_cls': label_cls.detach(),
                'adv_cls': adv_cls.detach()
            }

            data['pert'].requires_grad = True
            pert,loss,update_cls = self.oim_once(tracker,data)
            losses.append(loss)
            m+=1

            # disp x_crop for each iteration and the pert
            if cfg.ATTACKER.SHOW:
                img_t = img + prev_perts.sum(0)+pert
                img_t = torch.clamp(img_t, 0, 255)
                vis.images(img_t,win='X_attack')
                vis.images(pert,win='Pert_attack')
                plt.plot(losses)
                plt.ylabel('Loss')
                plt.xlabel('Iteration')
                vis.matplot(plt,win='Loss_attack')

                # validate the score
                # outputs = tracker.model.track(img_t)
                # disp_score = self.log_softmax(outputs['cls'].cpu())
                # disp_score = disp_score[:, :, :, :, 1].permute(1, 0, 2, 3)
                # disp_score_min = torch.min(torch.min(disp_score, 2)[0], 2)[0]
                # disp_score_max = torch.max(torch.max(disp_score, 2)[0], 2)[0]
                # dispmin = torch.zeros(disp_score.size())
                # dispmax = torch.zeros(disp_score.size())
                # for i in range(4):
                #     dispmin[i, :, :, :] = disp_score_min[i].repeat(1, 1, disp_score.shape[2], disp_score.shape[3])
                #     dispmax[i, :, :, :] = disp_score_max[i].repeat(1, 1, disp_score.shape[2], disp_score.shape[3])
                # disp_score = (disp_score - dispmin) / (dispmax - dispmin) * 255
                # vis.images(disp_score, win='Response_attack')

        self.opt_flow_prev_img = img

        if cfg.CUDA:
            prev_perts = torch.cat((prev_perts,pert),0).cuda()
        else:
            prev_perts = torch.cat((prev_perts, pert),0)
        pert_sum = prev_perts.sum(0)

        adv_img = img + pert_sum
        adv_img = torch.clamp(adv_img, 0, 255)
        pert_true = adv_img-img

        return None,pert_true,prev_perts,adv_img

    def ua_label(self, tracker, scale_z, outputs):

        cfg = self.cfg

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

        anchor_target = CrossAnchorTarget(cfg)

        results = anchor_target(desired_bbox, w)
        overlap = results[3]
        max_val = np.max(overlap)
        max_pos = np.where(overlap == max_val)
        adv_cls = torch.zeros(results[0].shape)
        adv_cls[:, max_pos[1], max_pos[2]] = 1
        adv_cls = adv_cls.view(1, adv_cls.shape[0], adv_cls.shape[1], adv_cls.shape[2])

        return adv_cls,same_prev

    def ta_label(self, tracker, scale_z, outputs):

        cfg = self.cfg

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

        anchor_target = CrossAnchorTarget(cfg)

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

    def oim_once(self,tracker,data):
        cfg = self.cfg

        if cfg.CUDA:
            zf = data['template_zf'].cuda()
            img = data['img'].cuda()
            pert = data['pert'].cuda()
            prev_perts = data['prev_perts'].cuda()
            adv_cls = data['adv_cls'].cuda()
        else:
            zf = data['template_zf']
            img = data['img']
            pert = data['pert']
            prev_perts = data['prev_perts']
            adv_cls = data['adv_cls']

        track_model = tracker.model

        zf_list=[]
        if zf.shape[0]>1:
            for i in range(0,zf.shape[0]):
                zf_list.append(zf[i,:,:,:].resize_(1,zf.shape[1],zf.shape[2],zf.shape[3]))
        else:
            zf_list = zf

        # get search
        w_z = tracker.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
        h_z = tracker.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
        s_z = np.sqrt(w_z * h_z)
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)


        def get_subwindow (im_tensor, pos, model_sz, original_sz, avg_chans):

            if isinstance(pos, float):
                pos = [pos, pos]
            sz = original_sz

            im = im_tensor.squeeze(0).permute(1,2,0)
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

        search = get_subwindow(img+pert+prev_perts.sum(0).view(1,prev_perts.shape[1],prev_perts.shape[2],prev_perts.shape[3]),\
                               tracker.center_pos,
                               cfg.TRACK.INSTANCE_SIZE,
                               round(s_x), tracker.channel_average)

        # get feature
        xf = track_model.backbone(search)
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

        adv_x = img
        if self.norm_type=='L_inf':
            x_grad = torch.sign(x_grad)
            adv_x = adv_x+pert+self.eplison*x_grad
            pert = adv_x-img
            pert = torch.clamp(pert,-self.inta,self.inta)
        elif self.norm_type=='L_1':
            x_grad = x_grad/torch.norm(x_grad,1)
            adv_x = adv_x+pert+self.eplison*x_grad
            pert = adv_x-img
            pert = torch.clamp(pert/np.linalg.norm(pert,1),-self.inta,self.inta)
        elif self.norm_type=='L_2':
            x_grad = x_grad/torch.norm(x_grad,2)
            adv_x = adv_x+pert+self.eplison*x_grad
            pert = adv_x-img
            pert = torch.clamp(pert/np.linalg.norm(pert,2),-self.inta,self.inta)

        p_img = img+pert+prev_perts.sum(0)
        p_img = torch.clamp(p_img,0,255)
        pert = p_img-img-prev_perts.sum(0)

        return pert,total_loss,cls

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

        def line(t,r, theta):
            """line genetate line traj
            
            Arguments:
                t {float} -- time
                r {float} -- length
                theta {float} -- angle
            
            Returns:
                [float,float] -- position
            """
            return t*r*cos(theta), t*r*sin(theta)


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

            if type == 4:
                t = 1.*i/vid_l
                x, y = line(t,r, pi*0.25)



            tpos.append(np.clip(x + init_rect[0], 0, vid_w - 1))
            tpos.append(np.clip(y + init_rect[1], 0, vid_h - 1))
            target_traj.append(tpos)
        self.target_traj = target_traj
        return target_traj


    def _get_bbox(self, center_pos, shape):
        cfg = self.cfg
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
        bbox = center2corner(Center(cx-w/2, cy-h/2, w, h))
        return bbox


