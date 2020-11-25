from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import pickle

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from toolkit.datasets import OTBDataset, UAVDataset, LaSOTDataset, \
        VOTDataset, NFSDataset, VOTLTDataset,OPEDataset
from toolkit.evaluation import OPEBenchmark, AccuracyRobustnessBenchmark, \
        EAOBenchmark, F1Benchmark

parser = argparse.ArgumentParser(description='targeted-attack tracking evaluation')
parser.add_argument('--tracker_path', '-p', type=str,
                    help='tracker result path')
parser.add_argument('--dataset', '-d', type=str,
                    help='dataset name')
parser.add_argument('--pert_type', '-pe', type=str,
                    help='perturbation eval type')
parser.add_argument('--num', '-n', default=50, type=int,
                    help='number of thread to eval')
parser.add_argument('--tracker_prefix', '-t', default='',
                    type=str, help='tracker name')
parser.add_argument('--cal_iternum', '-i', dest='cal_iternum',
                    action='store_true')
parser.add_argument('--show_video_level', '-s', dest='show_video_level',
                    action='store_true')
parser.set_defaults(show_video_level=False)

args = parser.parse_args()


def main():
    tracker_dir = os.path.join(args.tracker_path, args.dataset)
    trackers = glob(os.path.join(args.tracker_path,
                                 args.dataset,
                                 args.tracker_prefix+'*'))
    trackers = [x.split('/')[-1] for x in trackers]

    assert len(trackers) > 0
    args.num = min(args.num, len(trackers))

    root = os.path.realpath(os.path.join(os.path.dirname(__file__),
                            '../testing_dataset'))
    root = os.path.join(root, args.dataset)
    if 'OTB' in args.dataset or 'UAV' in args.dataset or 'LaSOT' in args.dataset:
        if 'UAV' in args.dataset:
            dataset = UAVDataset(args.dataset, root)
        elif 'OTB' in args.dataset:
            dataset = OTBDataset(args.dataset, root)
        elif 'LaSOT' in args.dataset:
            dataset = LaSOTDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        dataset = add_ta_traj(dataset)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_ta_precision,
                 trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        MAP_ret = {}
        if args.pert_type=='P':
            with Pool(processes=args.num) as pool:
                for ret in tqdm(pool.imap_unordered(benchmark.eval_p,
                    trackers), desc='eval map', total=len(trackers), ncols=100):
                    MAP_ret.update(ret)
        elif args.pert_type=='MAP':
            with Pool(processes=args.num) as pool:
                for ret in tqdm(pool.imap_unordered(benchmark.eval_map,
                    trackers), desc='eval map', total=len(trackers), ncols=100):
                    MAP_ret.update(ret)

        #AvrIterNum = {}
        #with Pool(processes=args.num) as pool:
        #    for ret in tqdm(pool.imap_unordered(benchmark.eval_AvrIterNum,
        #         trackers), desc='eval iter_num', total=len(trackers), ncols=100):
        #        AvrIterNum.update(ret)
        AvrIterNum = None

        #TIME_ret = {}
        #with Pool(processes=args.num) as pool:
        #    for ret in tqdm(pool.imap_unordered(benchmark.eval_time,
        #        trackers), desc='eval time', total=len(trackers), ncols=100):
        #        TIME_ret.update(ret)
        TIME_ret = None

        # attr_success_ret = {}
        # attr_precision_ret = {}
        # attr_MAP_ret = {}
        # for attr in dataset.attr:
        #     print('Now deal with attr %s' % attr)
        #     videos = dataset.attr[attr]
        #     for model in success_ret:
        #         attr_success_ret[model] = {video:success_ret[model][video] for video in success_ret[model] if video in videos}
        #         attr_precision_ret[model] = {video:precision_ret[model][video] for video in precision_ret[model] if video in videos}
        #         attr_MAP_ret[model] = {video: MAP_ret[model][video] for video in MAP_ret[model] if video in videos}
        #     benchmark.show_result_attack(attr_success_ret, attr_precision_ret,attr_MAP_ret,TIME_ret,
        #                           show_video_level=args.show_video_level)
        #     attr_success_ret.clear()
        #     attr_precision_ret.clear()
        #     attr_MAP_ret.clear()

        if args.cal_iternum:
            benchmark.show_result_attack(success_ret, precision_ret, AvrIterNum, TIME_ret,
                                         show_video_level=args.show_video_level)
        else:
            benchmark.show_result_attack(success_ret, precision_ret, MAP_ret, TIME_ret,
                                         show_video_level=args.show_video_level)

    elif 'VOT' in args.dataset:
        dataset = OPEDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        dataset = add_ta_traj(dataset)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_ta_precision,
                 trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        MAP_ret = {}
        if args.pert_type=='P':
            with Pool(processes=args.num) as pool:
                for ret in tqdm(pool.imap_unordered(benchmark.eval_p,
                    trackers), desc='eval map', total=len(trackers), ncols=100):
                    MAP_ret.update(ret)
        elif args.pert_type=='MAP':
            with Pool(processes=args.num) as pool:
                for ret in tqdm(pool.imap_unordered(benchmark.eval_map,
                    trackers), desc='eval map', total=len(trackers), ncols=100):
                    MAP_ret.update(ret)

        TIME_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_time,
                trackers), desc='eval time', total=len(trackers), ncols=100):
                TIME_ret.update(ret)

        benchmark.show_result_attack(success_ret, precision_ret, MAP_ret,TIME_ret,
                show_video_level=args.show_video_level)

def add_ta_traj(dataset):

    for video in dataset:
        traj_path = os.path.join(dataset.dataset_root, video.name + '_traj.pkl')
        traj_file = open(traj_path, 'rb')
        target_traj = pickle.load(traj_file)
        traj_file.close()
        video.ta_traj = target_traj

    return dataset

if __name__ == '__main__':
    main()
