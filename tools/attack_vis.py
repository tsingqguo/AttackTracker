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
parser.add_argument('--apts_num', type=int,
                    default=2)
parser.add_argument('--eplison', type=float,
                    default=3e-1)
parser.add_argument('--att_method', default='OIM', type=str,
        help='select different kinds of attacker.')
parser.add_argument('--att_type', default='UA', type=str,
        help='UA or TA.')
parser.add_argument('--reg_type', default='L21', type=str,
        help='L21, L2 or None')
parser.add_argument('--norm_type', default='L_inf', type=str,
        help='L_inf, L_1 or L_2, Momen')
parser.add_argument('--name_suffix', default='', type=str,
        help='L_inf, L_1 or L_2')
parser.add_argument('--enable_same_pert', action='store_true',
                    help='whether tell the same objective!')
parser.add_argument('--accframes', type=int,
                    default=30)
parser.add_argument('--traj_type', type=int,
                    default=1)
parser.add_argument('--forbid_att', action='store_true',
                    help='whether attacking')
parser.add_argument('--result_path_config', default='visual_results_pathes.txt', type=str,
        help='result config file')

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
    attacker = build_attacker(args.att_method,args.att_type,args.max_num,apts_num=2,reg_type=args.reg_type)

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

        if args.att_type=='TA':
            traj_path = os.path.join(dataset_root,video.name+'_traj_custom.pkl')
            #if os.path.exists(traj_path):
            #    traj_file = open(traj_path, 'rb')
            #    target_traj = pickle.load(traj_file)
            #    attacker.target_traj = target_traj
            #else:
            traj_file = open(traj_path,'wb')
            target_traj = attacker.target_traj_gen_custom(video.init_rect,video.height,video.width,len(video.img_names),args.traj_type)
            #target_traj = attacker.target_traj_gen_supervised(video.init_rect, video.height, video.width,len(video.img_names),video.gt_traj)
            pickle.dump(target_traj,traj_file)
            traj_file.close()

        if cfg.ATTACKER.EVAL:
            pert_degs = []

        # check if existing
        save_name = 'vis_'+model_name+'_'+args.att_method+'_'+args.att_type
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

            # if args.reg_type!='L21':
        save_name += '_REG' + args.reg_type
        #
        save_name += '_NORM' + args.norm_type
        save_name += args.name_suffix

        model_path = os.path.join('results', \
                                  args.dataset, save_name)

        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        result_path = os.path.join(model_path, '{}.txt'.format(video.name))
        pert_path = os.path.join(model_path, '{}_pert.txt'.format(video.name))
        dist_path = os.path.join(model_path, '{}_dist-to-ta.txt'.format(video.name))
        tatraj_path = os.path.join(model_path, '{}_ta_traj.txt'.format(video.name))
        attack_time_path = os.path.join(model_path, '{}_attack_time.txt'.format(video.name))
        track_time_path = os.path.join(model_path, '{}_track_time.txt'.format(video.name))

        print(args.att_method)

        if (cfg.ATTACKER.CHECK_EXIST and args.att_method!='OIM') and os.path.exists(result_path) and os.path.exists(pert_path) \
            and os.path.exists(attack_time_path) and os.path.exists(track_time_path):
            continue

        if cfg.ATTACKER.SAVE_VIDEO or args.att_method=='OIM':
            vid_root = os.path.join(model_path)
            vid_path = video.name + '.avi'
            vid_path = os.path.join(vid_root, vid_path)
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            print(vid_path)
            out = cv2.VideoWriter(vid_path, fourcc, 10.0, (video.width,video.height))
            result_paths = open(args.result_path_config).readlines()
            results = {r.split(' ')[1].strip(): open(os.path.join('results', args.dataset, r.split(' ')[0].strip(), \
                                                                      '%s.txt' % video.name)) for r in result_paths}
        for idx, (img, gt_bbox) in enumerate(video):
            print(video.name+'-Processing frame:'+str(idx))
            if cfg.ATTACKER.SAVE_META:
                frames_path = os.path.join(model_path, video.name)
                attacker.meta_path = frames_path
                if os.path.exists(frames_path) is False:
                    os.mkdir(frames_path)

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
                if args.forbid_att==False:
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
                    x_crop, pert_true, prev_perts, img = attacker.attack(tracker, img, t_prev_perts, APTS, OPTICAL_FLOW,ADAPT)
                else:
                    x_crop = []

                attack_toc = cv2.getTickCount()
                attack_times.append((attack_toc - attack_tic)/cv2.getTickFrequency())

                if cfg.ATTACKER.EVAL and not args.forbid_att:
                    pert_deg = []
                    pert_deg.append(torch.norm(pert_true,1).data.cpu().numpy())
                    pert_deg.append(torch.norm(pert_true,2).data.cpu().numpy())
                    pert_deg.append(torch.norm(pert_true,float('inf')).data.cpu().numpy())
                    pert_degs.append(pert_deg)

                # start tracking
                torch.cuda.empty_cache()
                outputs = tracker.track(img,x_crop,not args.forbid_att)
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
                scores.append(outputs['best_score'])
                # force the ground truth to be center
                #cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                #tracker.center_pos = np.array([cx, cy])
                tracker.size = np.array([w, h])
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - attack_toc)/cv2.getTickFrequency())

            if args.vis and idx == 0:
                cv2.destroyAllWindows()

            if (args.vis or cfg.ATTACKER.SAVE_VIDEO or args.att_method=='OIM') and idx > 0:
                colors = ([0, 255, 0], \
                          [255, 0, 0])

                gt_bbox = list(map(int, gt_bbox))
                pred_bbox = list(map(int, pred_bbox))
                img = cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                              (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 0, 0), 3)
                img = cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                              (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 0, 255), 3)
                cv2.putText(img, 'SPARK', (pred_bbox[0], pred_bbox[1]), \
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                if args.att_type == 'TA':
                    cv2.rectangle(img, (int(target_traj[idx][0])-5, int(target_traj[idx][1])-5),
                                (int(target_traj[idx][0])+5, int(target_traj[idx][1])+5), (0, 255, 255), 3)
                    pts = np.array(target_traj,np.int32)
                    pts = pts.reshape((-1,1,2))
                    img = cv2.polylines(img,[pts],False,(255,0,0))

                ck = 0
                for k in results:
                    pred_bbox_ = results[k].readline().strip().split(',')
                    pred_bbox_ = list(map(int, map(float, pred_bbox_)))
                    cv2.rectangle(img, (pred_bbox_[0], pred_bbox_[1]),
                                      (pred_bbox_[0] + pred_bbox_[2], pred_bbox_[1] + pred_bbox_[3]), colors[ck], 3)
                    cv2.putText(img, k, (pred_bbox_[0], pred_bbox_[1]), \
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, colors[ck], 2)
                    ck += 1
                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                if cfg.ATTACKER.SAVE_VIDEO or args.att_method=='OIM':
                    out.write(img)
                #cv2.imshow(video.name, img)
                #cv2.waitKey(1)
            if cfg.ATTACKER.SAVE_META:
                frame_name = os.path.split(video.img_names[idx])[1]
                cv2.imwrite(frames_path+'/'+frame_name, img)
        toc /= cv2.getTickFrequency()
        if cfg.ATTACKER.SAVE_VIDEO or args.att_method=='OIM':
            out.release()
            print('Save video successfully!')

        if cfg.ATTACKER.EVAL and args.att_type=='UA' and not args.forbid_att:
            # save random target_traj
            traj_path = os.path.join(dataset_root, video.name + '_ua_traj.pkl')
            traj_file = open(traj_path, 'wb')
            pickle.dump(attacker.target_traj, traj_file)
            traj_file.close()

        # calculate the center pos of predicted results
        pred_dist = []
        idx = 0
        for x in pred_bboxes:
            cx, cy, w, h = get_axis_aligned_bbox(np.array(x))
            tx, ty = target_traj[idx][0],target_traj[idx][1]
            dist = np.sqrt(np.square(cx-tx)+np.square(cy-ty))
            pred_dist.append(np.array(dist))
            idx +=1

        with open(tatraj_path, 'w') as f:
            for x in target_traj:
                f.write(','.join([str(i) for i in x])+'\n')
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x])+'\n')
        with open(dist_path, 'w') as f:
            for x in pred_dist:
                f.write(str(x)+'\n')
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
