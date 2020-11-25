import os
import numpy as np
from toolkit.tvnet_pytorch.train_options import arguments
from toolkit.tvnet_pytorch.model.network import model

import scipy.io as sio
import cv2
import PIL.Image as Image
from toolkit.tvnet_pytorch.utils import *
from toolkit.datasets import DatasetFactory
from torchvision import transforms

args = arguments().parse()

cur_dir = os.path.dirname(os.path.realpath(__file__))
dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)

# create dataset
dataset = DatasetFactory.create_dataset(name=args.dataset,
                                        dataset_root=dataset_root,
                                        load_img=False)
save_root = '/media/vil/sda/Object_Tracking/Dataset/OTB/'

torch.set_num_threads(1)

def get_transfrom():
    transforms_list = []
    transforms_list = [transforms.ToTensor()]
    return transforms.Compose(transforms_list)

to_tensor = get_transfrom()

def main():
    args.data_size = [1, 3, 224, 224]
    if cfg.CUDA:
        tvnet = model(args).cuda()
    else:
        tvnet = model(args)

    # extract optical flow
    for v_idx, video in enumerate(dataset):
        toc = 0
        flow_path = os.path.join(save_root, video.name, 'flow')
        if not os.path.exists(flow_path):
            os.mkdir(flow_path)
        for idx, (img, gt_bbox) in enumerate(video):
            res_path = os.path.join(flow_path, video.img_names[idx])
            res_path = res_path.replace('/img/', '/flow/')
            res_path = res_path.replace('.jpg', '.mat')
            if os.path.exists(res_path):
                print('existing path:' + res_path)
                continue
            tic = cv2.getTickCount()
            if idx + 1 == len(video):
                break
            img1 = Image.fromarray(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            img2 = Image.fromarray(cv2.cvtColor(cv2.imread(
                video.img_names[idx+1]), cv2.COLOR_BGR2RGB))
            
            w, h = img1.size

            if args.data_size[2] != h or args.data_size[3] != w:
                args.data_size[2] = h
                args.data_size[3] = w
                tvnet = model(args).cuda()

            img1 = to_tensor(img1).cuda()
            img2 = to_tensor(img2).cuda()

            img1 = img1.view([1]+list(img1.size()))
            img2 = img2.view([1]+list(img2.size()))
            with torch.no_grad():
                u1, u2, rho = tvnet(img1, img2, True)
            # Save flow map to file, for visualization in matlab
            u1_np = np.squeeze(u1.detach().cpu().numpy())
            u2_np = np.squeeze(u2.detach().cpu().numpy())
            flow_mat = np.zeros([h, w, 2])
            flow_mat[:, :, 0] = u1_np
            flow_mat[:, :, 1] = u2_np

            print('save path:'+res_path)
            sio.savemat(res_path, {'flow': flow_mat})

            if args.vis:
                # note that the flow map can also be visulized using library cv2
                # Sample code using cv2:
                plt.imshow(img)
                # Use Hue, Saturation, Value colour model
                hsv = np.zeros(img.shape, dtype=np.uint8)
                hsv[..., 1] = 255
                mag, ang = cv2.cartToPolar(
                    u1.detach().squeeze().cpu().numpy(), u2.detach().squeeze().cpu().numpy())
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                cv2.imshow("colored flow", bgr)
                cv2.waitKey(1)
                # cv2.destroyAllWindows()

            toc += cv2.getTickCount() - tic

        toc /= cv2.getTickFrequency()

        print('({:3d}) Video: { : 12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx+1, video.name, toc, idx / toc))


if __name__ == '__main__':
    main()
