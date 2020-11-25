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
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--att_method', default='OIM', type=str,
        help='select different kinds of attacker.')
parser.add_argument('--att_type', default='TA', type=str,
        help='UA or TA.')
parser.add_argument('--result_path_config', default='', type=str,
        help='result config file')

args = parser.parse_args()

def main():
    # load config
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)


    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)
    
    # OPE tracking
    for v_idx, video in enumerate(dataset):
        toc = 0
        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
        vid_root = os.path.join('visualize')
        if not os.path.exists(vid_root):
            os.makedirs(vid_root)
        vid_path = video.name + '.avi'
        vid_path = os.path.join(vid_root, vid_path)
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        out = cv2.VideoWriter(vid_path, fourcc, 10.0, (video.width,video.height))

        if args.att_type=='TA':
            traj_path = os.path.join(dataset_root,video.name+'_traj_custom.pkl')
            if os.path.exists(traj_path):
                traj_file = open(traj_path, 'rb')
                target_traj = pickle.load(traj_file)
            else:
                raise Exception('No Traj File')

        result_paths = open(args.result_path_config).readlines()
        results = {r.split(' ')[1].strip():open(os.path.join('results',args.dataset,r.split(' ')[0].strip(), \
            '%s.txt' % video.name)) for r in result_paths}
        colors = ([255,0,0], \
                  [0,255,0], \
                  [0,0,255])
        for idx, (img, gt_bbox) in enumerate(video):
            print(video.name+'-Processing frame:'+str(idx))

            gt_bbox = list(map(int, gt_bbox))
            cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                              (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 0, 0), 3)
            ck=0
            for k in results:
                pred_bbox = results[k].readline().strip().split(',')
                print(pred_bbox)
                pred_bbox = list(map(int,map(float, pred_bbox)))
                cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]),colors[ck], 3)
                cv2.putText(img, k, (pred_bbox[0], pred_bbox[1]), \
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                ck+=1

            if args.att_type == 'TA':
                cv2.rectangle(img, (int(target_traj[idx][0])-5, int(target_traj[idx][1])-5),
                                (int(target_traj[idx][0])+5, int(target_traj[idx][1])+5), (255, 0, 0), 3)
                pts = np.array(target_traj,np.int32)
                pts = pts.reshape((-1,1,2))
                img = cv2.polylines(img,[pts],False,(255,0,0))
                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                out.write(img)

        out.release()
        break

if __name__ == '__main__':
    main()
