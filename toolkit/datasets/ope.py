import json
import os
import numpy as np

from PIL import Image
from tqdm import tqdm
from glob import glob

from .dataset import Dataset
from .video import Video
from pysot.utils.bbox import get_axis_aligned_bbox,cxy_wh_2_rect

class OPEVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        video_dir: video directory
        init_rect: init rectangle
        img_names: image names
        gt_rect: groundtruth rectangle
        attr: attribute of video
    """
    def __init__(self, name, root, video_dir, init_rect, img_names,
            gt_rect, attr, load_img=False):
        new_gt_rect = []
        if len(gt_rect[0])==8:
            for i in range(len(gt_rect)):
                t_get_rect = get_axis_aligned_bbox(np.array(gt_rect[i]))
                t_get_rect=cxy_wh_2_rect(t_get_rect[0:2],t_get_rect[2:4])
                new_gt_rect.append(t_get_rect)
            gt_rect = np.array(new_gt_rect)
        super(OPEVideo, self).__init__(name, root, video_dir,
                init_rect, img_names, gt_rect, attr, load_img)

    def load_tracker(self, path, tracker_names=None, store=True):
        """
        Args:
            path(str): path to result
            tracker_name(list): name of tracker
        """
        if not tracker_names:
            tracker_names = [x.split('/')[-1] for x in glob(path)
                    if os.path.isdir(x)]
        if isinstance(tracker_names, str):
            tracker_names = [tracker_names]
        for name in tracker_names:
            traj_file = os.path.join(path, name, self.name+'.txt')
            if os.path.exists(traj_file):
                with open(traj_file, 'r') as f :
                    pred_traj = [list(map(float, x.strip().split(',')))
                            for x in f.readlines()]
                    if len(pred_traj) != len(self.gt_traj):
                        print(name, len(pred_traj), len(self.gt_traj), self.name)
                    if store:
                        self.pred_trajs[name] = pred_traj
                    else:
                        return pred_traj
            else:
                print(traj_file)
        self.tracker_names = list(self.pred_trajs.keys())

    def load_map(self, path, tracker_names=None, store=True):
        """
        Args:
            path(str): path to result
            tracker_name(list): name of tracker
        """
        if not tracker_names:
            tracker_names = [x.split('/')[-1] for x in glob(path)
                    if os.path.isdir(x)]
        if isinstance(tracker_names, str):
            tracker_names = [tracker_names]
        for name in tracker_names:
            map_file = os.path.join(path, name, self.name+'_pert.txt')
            if os.path.exists(map_file):
                with open(map_file, 'r') as f :
                    map_res = [list(map(float, x.strip().split(',')))
                            for x in f.readlines()]
                map_l1 =[]
                for map_ in map_res:
                    map_l1.append(map_[0]/(255*255*3))
                if store:
                    self.map_res[name] = map_l1
                else:
                    return map_l1
            else:
                print(map_file)
        self.tracker_names = list(self.map_res.keys())

    def load_p(self, path, tracker_names=None, store=True):
        """
        Args:
            path(str): path to result
            tracker_name(list): name of tracker
        """
        if not tracker_names:
            tracker_names = [x.split('/')[-1] for x in glob(path)
                    if os.path.isdir(x)]
        if isinstance(tracker_names, str):
            tracker_names = [tracker_names]
        for name in tracker_names:
            map_file = os.path.join(path, name, self.name+'_pert.txt')
            if os.path.exists(map_file):
                with open(map_file, 'r') as f :
                    map_res = [list(map(float, x.strip().split(',')))
                            for x in f.readlines()]
                map_l1 =[]
                for map_ in map_res[1:]:
                    #map_l1.append(1000*np.sqrt(np.square(map_[2]/(287*287*3))))
                    map_l1.append(map_[2])
                if store:
                    self.map_res[name] = map_l1
                else:
                    return map_l1
            else:
                print(map_file)
        self.tracker_names = list(self.map_res.keys())

    def load_time(self, path, tracker_names=None, store=True):
        """
        Args:
            path(str): path to result
            tracker_name(list): name of tracker
        """
        if not tracker_names:
            tracker_names = [x.split('/')[-1] for x in glob(path)
                             if os.path.isdir(x)]
        if isinstance(tracker_names, str):
            tracker_names = [tracker_names]
        for name in tracker_names:
            time_file = os.path.join(path, name, self.name + '_attack_time.txt')
            if os.path.exists(time_file):
                with open(time_file, 'r') as f:
                    tmpstr = f.readline()
                    tmplist = tmpstr.split(',')
                    tmplist = tmplist[1:]

                times = []
                for iter_ in tmplist:
                    times.append(float(iter_))

                if store:
                    self.time_res[name] = times
                else:
                    return times
            else:
                print(time_file)
        self.tracker_names = list(self.time_res.keys())

class OPEDataset(Dataset):
    """
    Args:
        name: dataset name, should be 'OTB100', 'CVPR13', 'OTB50'
        dataset_root: dataset root
        load_img: wether to load all imgs
    """
    def __init__(self, name, dataset_root, load_img=False):
        super(OPEDataset, self).__init__(name, dataset_root)
        with open(os.path.join(dataset_root, name+'.json'), 'r') as f:
            meta_data = json.load(f)

        # load videos
        pbar = tqdm(meta_data.keys(), desc='loading '+name, ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)
            self.videos[video] = OPEVideo(video,
                                          dataset_root,
                                          meta_data[video]['video_dir'],
                                          meta_data[video]['init_rect'],
                                          meta_data[video]['img_names'],
                                          meta_data[video]['gt_rect'],
                                          None,
                                          load_img)
