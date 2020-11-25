# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np
import pickle

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.attacker.attacker_builder import build_attacker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory

parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', type=str,
        help='datasets')
parser.add_argument('--config', default='', type=str,
        help='config file')
parser.add_argument('--snapshot', default='', type=str,
        help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', action='store_true',
        help='whether visualzie result')
parser.add_argument('--max_num', type=int,
                    default=10)
parser.add_argument('--interval', type=int,
                    default=-1,
        help='-1:adapt,>0:attack with fixed interval, 1000:only use the perturation from the first frame')
parser.add_argument('--opt_flow', action='store_true',
                    help='whether using optical flow')
parser.add_argument('--apts', action='store_true',
                    help='whether attacking apts')

args = parser.parse_args()

torch.set_num_threads(1)

def main():
    # load config
    cfg.merge_from_file(args.config)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)

    # create model
    model = ModelBuilder()

    # load model
    if cfg.CUDA:
        model = load_pretrain(model, args.snapshot).cuda().eval()
    else:
        model = load_pretrain(model, args.snapshot)

    # build tracker
    tracker = build_tracker(model)

    # build attacker
    attacker = build_attacker('UA', args.max_num)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.snapshot.split('/')[-1].split('.')[0]
    total_lost = 0

    # OPE tracking
    for v_idx, video in enumerate(dataset):
        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
        toc = 0
        pred_bboxes = []
        scores = []
        track_times = []
        attack_times = []

        for idx, (img, gt_bbox) in enumerate(video):
            print('Processing frame:'+str(idx))
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                scores.append(None)
                if 'VOT2018-LT' == args.dataset:
                    pred_bboxes.append([1])
                else:
                    pred_bboxes.append(pred_bbox)
                attack_toc = tic
            else:
                # start attacking
                attack_tic = cv2.getTickCount()
                attacker.v_id = idx
                if idx == 1:
                    prev_perts,APTS,OPTICAL_FLOW,ADAPT=None,False,False,False
                # only attack the first frame
                elif args.interval==1000:
                    APTS, OPTICAL_FLOW, ADAPT = False, False, False
                # attack with an interval
                elif args.interval>0:
                    if (idx-1)%args.interval==0:
                        prev_perts,APTS, OPTICAL_FLOW, ADAPT = None,args.apts, args.opt_flow, False
                    else:
                        APTS, OPTICAL_FLOW, ADAPT= args.apts, args.opt_flow, False
                # adative attack
                elif args.interval==-1:
                    APTS, OPTICAL_FLOW, ADAPT = args.apts, args.opt_flow, True
                if prev_perts is None:
                    t_prev_perts = prev_perts
                else:
                    t_prev_perts = prev_perts.clone().detach()
                x_crop, prev_perts, img = attacker.attack(tracker,img,t_prev_perts,APTS,OPTICAL_FLOW,ADAPT)

                attack_toc = cv2.getTickCount()
                attack_times.append((attack_toc - attack_tic)/cv2.getTickFrequency())

                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                tracker.center_pos = np.array([gt_bbox_[0] + (gt_bbox_[2] - 1) / 2,
                                               gt_bbox_[1] + (gt_bbox_[3] - 1) / 2])
                tracker.size = np.array([gt_bbox_[2], gt_bbox_[3]])

                # start tracking
                torch.cuda.empty_cache()
                outputs = tracker.track(img,x_crop,True)
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
                scores.append(outputs['best_score'])
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - attack_toc)/cv2.getTickFrequency())

            if args.vis and idx == 0:
                cv2.destroyAllWindows()
            if (args.vis or cfg.ATTACKER.SAVE_VIDEO) and idx > 0:
                gt_bbox = list(map(int, gt_bbox))
                pred_bbox = list(map(int, pred_bbox))
                cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                              (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                              (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow(video.name, img)
                cv2.waitKey(1)
        toc /= cv2.getTickFrequency()

if __name__ == '__main__':
    main()
