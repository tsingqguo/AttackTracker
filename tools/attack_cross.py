# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pickle

from pysot.core.config import cfg as cfg
from pysot.models.model_builder import ModelBuilder
from pysot.models.model_builder_cross import CROSSModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.tracker.tracker_builder_cross import build_trackerC
from pysot.attacker.oim_attacker_cross import CROSSOIMAttacker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from torchvision import transforms

parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', type=str,
        help='datasets')
parser.add_argument('--config', default='', type=str,
        help='config file')
parser.add_argument('--configC', default='', type=str,
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
parser.add_argument('--apts_num', type=int,
                    default=2)
parser.add_argument('--eplison', type=float,
                    default=1)
parser.add_argument('--att_method', default='OIM', type=str,
        help='select different kinds of attacker.')
parser.add_argument('--att_type', default='UA', type=str,
        help='UA or TA.')

parser.add_argument('--reg_type', default='L21', type=str,
        help='L21, L2, None,')

parser.add_argument('--norm_type', default='L_inf', type=str,
        help='L_inf, L_1 or L_2, Momen')

parser.add_argument('--name_suffix', default='', type=str,
        help='L_inf, L_1 or L_2')

parser.add_argument('--enable_same_pert', action='store_true',
                    help='whether tell the same objective!')

parser.add_argument('--accframes', type=int,
                    default=30)

args = parser.parse_args()

torch.set_num_threads(1)

def main():

    cfgC = cfg.clone()
    cfg.merge_from_file(args.config)

    # load config
    if args.configC =='alex' or args.configC =='alex_online':
        pathC = '../siamrpn_alex_dwxcorr/'
    elif args.configC == 'mobile' or args.configC =='mobile_online':
        pathC = '../siamrpn_mobilev2_l234_dwxcorr/'
    elif args.configC == 'r50' or args.configC =='r50_online':
        pathC = '../siamrpn_r50_l234_dwxcorr/'

    print('Attacking model:'+pathC)

    configC = pathC+'config.yaml'
    cfgC.merge_from_file(configC)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)

    # create model
    if cfg.TRACK.TYPE=='DiMPTracker' or cfg.TRACK.TYPE=='ATOMTracker' or cfg.TRACK.TYPE=='ECOTracker':
        model=[]
    else:
        # create model
        model = ModelBuilder()
        # load model
        if cfg.CUDA:
            model = load_pretrain(model, args.snapshot).cuda().eval()
        else:
            model = load_pretrain(model, args.snapshot)

    modelC = CROSSModelBuilder(cfgC)

    # load model
    if cfg.CUDA:
        #model = load_pretrain(model, args.snapshot).cuda().eval()
        modelC =load_pretrain(modelC, pathC+args.snapshot).cuda().eval()
    else:
        #model = load_pretrain(model, args.snapshot)
        modelC = load_pretrain(modelC, pathC + args.snapshot)

    # build tracker
    tracker = build_tracker(model)
    trackerC = build_trackerC(modelC,cfgC)

    # build attacker
    attacker =CROSSOIMAttacker(type=args.att_type,max_num=args.max_num,cfg=cfgC,eplison=args.eplison)

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
        if cfg.ATTACKER.SAVE_VIDEO:
            vid_root = os.path.join('results', args.dataset, model_name + args.att_method + args.att_type)
            vid_path = video.name + '_' + model_name + args.att_method + args.att_type+'.avi'
            vid_path = os.path.join(vid_root, vid_path)
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            out = cv2.VideoWriter(vid_path, fourcc, 10.0, (video.width,video.height))

        if args.att_type=='TA':
            traj_path = os.path.join(dataset_root,video.name+'_traj.pkl')
            if os.path.exists(traj_path):
                traj_file = open(traj_path, 'rb')
                target_traj = pickle.load(traj_file)
                attacker.target_traj = target_traj
            else:
                traj_file = open(traj_path,'wb')
                target_traj = attacker.target_traj_gen(video.init_rect,video.height,video.width,len(video.img_names))
                pickle.dump(target_traj,traj_file)
                traj_file.close()

        if cfg.ATTACKER.EVAL:
            pert_degs = []

        # check if existing
        save_name = model_name+'_'+args.att_method+'_'+args.att_type
        if args.opt_flow:
            save_name += '_FLOW'
        else:
            save_name += '_noFLow'

        if args.interval==-1:
            save_name += '_ADAPT'
        else:
            save_name += '_noADAPT'+str(args.interval)

        if args.apts:
            save_name += '_APTS'+str(attacker.apts_num)
        else:
            save_name += '_noAPTS'

        if args.enable_same_pert:
            save_name += '_SPERT'
        else:
            save_name += '_noSPERT'

        #if args.reg_type!='L21':
        save_name += '_REG'+args.reg_type
        #
        save_name += '_NORM'+args.norm_type
        save_name += args.name_suffix

        save_name +='_'+args.configC

        model_path = os.path.join('results', \
                                  args.dataset, save_name)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        result_path = os.path.join(model_path, '{}.txt'.format(video.name))
        pert_path = os.path.join(model_path, '{}_pert.txt'.format(video.name))
        attack_time_path = os.path.join(model_path, '{}_attack_time.txt'.format(video.name))
        track_time_path = os.path.join(model_path, '{}_track_time.txt'.format(video.name))

        if cfg.ATTACKER.CHECK_EXIST and os.path.exists(result_path) and os.path.exists(pert_path) \
            and os.path.exists(attack_time_path) and os.path.exists(track_time_path):
            continue

        for idx, (img, gt_bbox) in enumerate(video):
            #print(video.name+'-Processing frame:'+str(idx))
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]

                tracker.init(img, gt_bbox_)
                trackerC.init(img, gt_bbox_)

                pred_bbox = gt_bbox_
                scores.append(None)
                if 'VOT2018-LT' == args.dataset:
                    pred_bboxes.append([1])
                else:
                    pred_bboxes.append(pred_bbox)
                attack_toc = tic
                if cfg.ATTACKER.EVAL:
                    pert_degs.append([0])
                    if args.att_type=='UA':
                        tpos = []
                        tpos.append(cx)
                        tpos.append(cy)
                        attacker.target_traj.append(tpos)
                    attack_times.append([0])
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

                trackerC.center_pos = tracker.center_pos
                trackerC.size = tracker.size

                x_crop, pert_true, prev_perts, adv_img = attacker.attack(trackerC, img, t_prev_perts, APTS, OPTICAL_FLOW,ADAPT)

                if x_crop.shape[2]!=cfg.TRACK.INSTANCE_SIZE:
                    x_crop = F.interpolate(x_crop,size=(cfg.TRACK.INSTANCE_SIZE,cfg.TRACK.INSTANCE_SIZE))

                attack_toc = cv2.getTickCount()
                attack_times.append((attack_toc - attack_tic)/cv2.getTickFrequency())

                if cfg.ATTACKER.EVAL:
                    pert_deg = []
                    pert_deg.append(torch.norm(pert_true,1).data.cpu().numpy())
                    pert_deg.append(torch.norm(pert_true,2).data.cpu().numpy())
                    pert_deg.append(torch.norm(pert_true,float('inf')).data.cpu().numpy())
                    pert_degs.append(pert_deg)

                # start tracking
                torch.cuda.empty_cache()
                if cfg.TRACK.TYPE == 'DiMPTracker' or cfg.TRACK.TYPE == 'ATOMTracker' or cfg.TRACK.TYPE == 'ECOTracker':
                    outputs = tracker.track(img)
                else:
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
                if args.att_type == 'TA':
                    cv2.rectangle(img, (int(target_traj[idx][0])-5, int(target_traj[idx][1])-5),
                                (int(target_traj[idx][0])+5, int(target_traj[idx][1])+5), (255, 0, 0), 3)
                    pts = np.array(target_traj,np.int32)
                    pts = pts.reshape((-1,1,2))
                    img = cv2.polylines(img,[pts],False,(255,0,0))
                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                if cfg.ATTACKER.SAVE_VIDEO:
                    out.write(img)
                cv2.imshow(video.name, img)
                cv2.waitKey(1)

        toc /= cv2.getTickFrequency()
        if cfg.ATTACKER.SAVE_VIDEO:
            out.release()

        if cfg.ATTACKER.EVAL and args.att_type=='UA' :
            # save random target_traj
            traj_path = os.path.join(dataset_root, video.name + '_ua_traj.pkl')
            traj_file = open(traj_path, 'wb')
            pickle.dump(attacker.target_traj, traj_file)
            traj_file.close()


        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x])+'\n')
        with open(pert_path, 'w') as f:
            for x in pert_degs:
                f.write(','.join([str(i) for i in x])+'\n')
        with open(attack_time_path, 'w') as f:
            f.write(','.join([str(i) for i in attack_times]) + '\n')
        with open(track_time_path, 'w') as f:
            f.write(','.join([str(i) for i in track_times]) + '\n')

        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx+1, video.name, toc, idx/toc))

if __name__ == '__main__':
    main()
