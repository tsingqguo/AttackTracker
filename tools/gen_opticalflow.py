
# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import PIL.Image as Image
from pysot.core.config import cfg

from toolkit.tvnet.tvnet import *
from toolkit.datasets import DatasetFactory


parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', type=str,
        help='datasets')
parser.add_argument('--vis', type=bool,
        default=False)
args = parser.parse_args()

cur_dir = os.path.dirname(os.path.realpath(__file__))
dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)

# create dataset
dataset = DatasetFactory.create_dataset(name=args.dataset,
                                        dataset_root=dataset_root,
                                        load_img=False)
torch.set_num_threads(1)

def main():

    net = TVNet(max_iterations=50)
    # extract optical flow
    for v_idx, video in enumerate(dataset):
        flow_path = os.path.join(dataset_root, video.name,'flow')
        if not os.path.exists(flow_path):
            os.mkdir(flow_path)
        for idx, (img, gt_bbox) in enumerate(video):
            res_path = os.path.join(flow_path,video.img_names[idx])
            res_path = res_path.replace('/img/','/flow/')
            res_path = res_path.replace('.jpg', '.mat')
            if os.path.exists(res_path):
                print('existing path:' + res_path)
                continue
            if idx==len(video)-1:
                break
            img1 = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img2 = Image.fromarray(cv2.cvtColor(cv2.imread(video.img_names[idx+1]), cv2.COLOR_BGR2RGB))
            u1, u2, rho = net(img1, img2)
            w, h = img1.size
            # Save flow map to file, for visualization in matlab
            u1_np = np.squeeze(u1.detach().numpy())
            u2_np = np.squeeze(u2.detach().numpy())
            flow_mat = np.zeros([h, w, 2])
            flow_mat[:, :, 0] = u1_np
            flow_mat[:, :, 1] = u2_np

            print('save path:'+res_path)
            sio.savemat(res_path, {'flow': flow_mat})

            if args.vis:
                # note that the flow map can also be visulized using library cv2
                # Sample code using cv2:
                plt.imshow(img)
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

        print('({:3d}) Video: { : 12s}'.format(v_idx+1, video.name))


if __name__ == '__main__':
    main()