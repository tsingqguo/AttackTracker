# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Houwen Peng and Zhipeng Zhang
# Email: houwen.peng@microsoft.com
# Clean testing scripts for SiamFC
# New: support GENE tuning
# ------------------------------------------------------------------------------
import cv2
import numpy as np

from torch.autograd import Variable
from extern.siamdw.lib.utils.utils import cxy_wh_2_rect, load_yaml, im_to_torch, get_subwindow_tracking, make_scale_pyramid
from pysot.core.config import cfg

class SiamDWTracker(object):
    def __init__(self, model):
        super(SiamDWTracker, self).__init__()
        self.net = model

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        target_pos = [bbox[0]+bbox[2]/2,bbox[1]+bbox[3]/2]
        target_sz  = [bbox[2],bbox[3]]
        
        state = dict()
        # epoch test
        p = FCConfig()

        # single test
        p.update(cfg.SIAMDW)
        p.renew()

        net = self.net

        avg_chans = np.mean(img, axis=(0, 1))

        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))
        scale_z = p.exemplar_size / s_z

        z_crop = get_subwindow_tracking(img, target_pos, p.exemplar_size, s_z, avg_chans)

        d_search = (p.instance_size - p.exemplar_size) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad
        min_s_x = 0.2 * s_x
        max_s_x = 5 * s_x

        s_x_serise = {'s_x': s_x, 'min_s_x': min_s_x, 'max_s_x': max_s_x}
        p.update(s_x_serise)

        z = Variable(z_crop.unsqueeze(0))

        net.template(z.cuda())

        if p.windowing == 'cosine':
            window = np.outer(np.hanning(int(p.score_size) * int(p.response_up)),
                              np.hanning(int(p.score_size) * int(p.response_up)))
        elif p.windowing == 'uniform':
            window = np.ones(int(p.score_size) * int(p.response_up), int(p.score_size) * int(p.response_up))
        window /= window.sum()

        p.scales = p.scale_step ** (range(p.num_scale) - np.ceil(p.num_scale // 2))

        state['p'] = p
        state['net'] = net
        state['avg_chans'] = avg_chans
        state['window'] = window
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['im_h'] = img.shape[0]
        state['im_w'] = img.shape[1]

        self.state = state    

    def update(self, net, s_x, x_crops, target_pos, window, p):
            # refer to original SiamFC code
        response_map_org = net.track(x_crops).squeeze().permute(1, 2, 0)
        response_map = response_map_org.cpu().data.numpy()
        up_size = p.response_up * response_map.shape[0]
        response_map_up = cv2.resize(response_map, (up_size, up_size), interpolation=cv2.INTER_CUBIC)
        temp_max = np.max(response_map_up, axis=(0, 1))
        s_penaltys = np.array([p.scale_penalty, 1., p.scale_penalty])
        temp_max *= s_penaltys
        best_scale = np.argmax(temp_max)

        response_map = response_map_up[..., best_scale]
        response_map = response_map - response_map.min()
        response_map = response_map / response_map.sum()

        # apply windowing
        response_map = (1 - p.w_influence) * response_map + p.w_influence * window
        r_max, c_max = np.unravel_index(response_map.argmax(), response_map.shape)
        p_corr = [c_max, r_max]

        disp_instance_final = p_corr - np.ceil(p.score_size * p.response_up / 2)
        disp_instance_input = disp_instance_final * p.total_stride / p.response_up
        disp_instance_frame = disp_instance_input * s_x / p.instance_size
        new_target_pos = target_pos + disp_instance_frame

        return new_target_pos, best_scale, response_map_org

    def track(self, img, x_crops=None, ispert=False):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        state = self.state

        p = state['p']
        net = state['net']
        avg_chans = state['avg_chans']
        window = state['window']
        target_pos = state['target_pos']
        target_sz = state['target_sz']

        scaled_instance = p.s_x * p.scales
        scaled_target = [[target_sz[0] * p.scales], [target_sz[1] * p.scales]]

        if x_crops is None or ispert==False:
            x_crops = Variable(make_scale_pyramid(img, target_pos, scaled_instance, p.instance_size, avg_chans))

        target_pos, new_scale, score = self.update(net, p.s_x, x_crops.cuda(), target_pos, window, p)

        # scale damping and saturation
        p.s_x = max(p.min_s_x, min(p.max_s_x, (1 - p.scale_lr) * p.s_x + p.scale_lr * scaled_instance[new_scale]))

        target_sz = [(1 - p.scale_lr) * target_sz[0] + p.scale_lr * scaled_target[0][0][new_scale],
                     (1 - p.scale_lr) * target_sz[1] + p.scale_lr * scaled_target[1][0][new_scale]]

        target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
        target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
        target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
        target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['p'] = p
        state['score'] = score

        self.state = state

        bbox = cxy_wh_2_rect(target_pos, target_sz)
        best_score = np.max(score.cpu().data.numpy(), axis=(0, 1))

        return {
                'bbox': bbox,
                'best_score': best_score
               }

class FCConfig(object):
    # These are the default hyper-params for SiamFC
    num_scale = 3
    scale_step = 1.0375
    scale_penalty = 0.9745
    scale_lr = 0.590
    response_up = 16

    windowing = 'cosine'
    w_influence = 0.350

    exemplar_size = 127
    instance_size = 255
    score_size = 17
    total_stride = 8
    context_amount = 0.5

    def update(self, newparam=None):
        if newparam:
            for key, value in newparam.items():
                setattr(self, key, value)
            self.renew()

    def renew(self):
        self.exemplar_size = self.instance_size - 128
        self.score_size = (self.instance_size - self.exemplar_size) // self.total_stride + 1