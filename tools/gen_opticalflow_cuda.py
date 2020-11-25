
# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import torch
import cv2
import PIL
import PIL.Image as Image
from pysot.core.config import cfg

from toolkit.datasets import DatasetFactory
from toolkit.tvnet_pytorch.model import network
from toolkit.tvnet_pytorch.train_options import arguments
import torchvision.transforms as transforms
from toolkit.tvnet_pytorch.data.frame_dataset import frame_dataset
import scipy.io as sio
from toolkit.tvnet_pytorch.utils import *


args = arguments().parse()
cur_dir = os.path.dirname(os.path.realpath(__file__))
dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)

# create dataset
dataset = DatasetFactory.create_dataset(name=args.dataset,
                                        dataset_root=dataset_root,
                                        load_img=False)
torch.set_num_threads(1)

def main():

    def get_transfrom():
        transforms_list = []
        transforms_list = [transforms.ToTensor()]
        return transforms.Compose(transforms_list)

    to_tensor = get_transfrom()
    #save_root = '/media/vil/sda/Object_Tracking/Dataset/OTB/'
    save_root = dataset_root
    # extract optical flow
    for v_idx, video in enumerate(dataset):
        flow_path = os.path.join(save_root, video.name,'flow')
        if not os.path.exists(flow_path):
            os.mkdir(flow_path)
        args.data_size = [1,3,video.height,video.width]
        if cfg.CUDA:
            model = network.model(args).cuda()
        else:
            model = network.model(args)
        for idx,(img,gt_bbox) in enumerate(video):
            frame_name = os.path.split(video.img_names[idx])[1]
            res_path = os.path.join(flow_path,frame_name)
            res_path = res_path.replace('.jpg', '.mat')
            if os.path.exists(res_path):
                print('existing path:' + res_path)
                continue
            if idx==len(video)-1:
                break
            if cfg.CUDA:
                img1 = to_tensor(Image.open(video.img_names[idx]).convert('RGB')).float().cuda()
                img2 = to_tensor(Image.open(video.img_names[idx+1]).convert('RGB')).float().cuda()
            else:
                img1 = to_tensor(Image.open(video.img_names[idx]).convert('RGB')).float()
                img2 = to_tensor(Image.open(video.img_names[idx+1]).convert('RGB')).float()

            img1 = img1.view(1,img1.shape[0],img1.shape[1],img1.shape[2])
            img2 = img2.view(1,img2.shape[0],img2.shape[1],img2.shape[2])
            t_img1 = img1.clone().detach()
            t_img2 = img2.clone().detach()
            u1, u2,x2_warp= model.forward(t_img1, t_img2, need_result=True)

            w, h = img1.shape[3],img1.shape[2]
            # Save flow map to file, for visualization in matlab
            u1_np = np.squeeze(u1.detach().cpu().data.numpy())
            u2_np = np.squeeze(u2.detach().cpu().data.numpy())
            flow_mat = np.zeros([h, w, 2])
            flow_mat[:, :, 0] = u1_np
            flow_mat[:, :, 1] = u2_np

            print('save path:'+res_path)
            sio.savemat(res_path, {'flow': flow_mat})
            del img1,img2,t_img1,t_img2
            torch.cuda.empty_cache()
            if args.vis:
                # note that the flow map can also be visulized using library cv2
                # Sample code using cv2:
                #PIL.imshow(img)
                #Use Hue, Saturation, Value colour model
                hsv = np.zeros(img.shape, dtype=np.uint8)
                hsv[..., 1] = 255
                mag, ang = cv2.cartToPolar(u1.detach().squeeze().numpy(), u2.detach().squeeze().numpy())
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                cv2.imshow("colored flow", bgr)
                cv2.waitKey(1)
                #cv2.destroyAllWindows()


if __name__ == '__main__':
    main()