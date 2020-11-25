import os
import json

from tqdm import tqdm
from glob import glob

from .dataset import Dataset
from .video import Video


class UAVVideo(Video):
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
        super(UAVVideo, self).__init__(name, root, video_dir,
                                       init_rect, img_names, gt_rect, attr, load_img)

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
            map_file = os.path.join(path, name, self.name + '_pert.txt')
            if os.path.exists(map_file):
                with open(map_file, 'r') as f:
                    map_res = [list(map(float, x.strip().split(',')))
                               for x in f.readlines()]
                map_l1 = []
                for map_ in map_res:
                    map_l1.append(map_[0] / (255 * 255 * 3))
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
            map_file = os.path.join(path, name, self.name + '_pert.txt')
            if os.path.exists(map_file):
                with open(map_file, 'r') as f:
                    map_res = [list(map(float, x.strip().split(',')))
                               for x in f.readlines()]
                map_l1 = []
                for map_ in map_res[1:]:
                    map_l1.append(1000 * np.sqrt(np.square(map_[1] / (287 * 287 * 3))))
                if store:
                    self.map_res[name] = map_l1
                else:
                    return map_l1
            else:
                print(map_file)
        self.tracker_names = list(self.map_res.keys())

    def load_iter(self, path, tracker_names=None, store=True):
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
            iter_file = os.path.join(path, name, self.name + '_iternum.txt')
            if os.path.exists(iter_file):
                with open(iter_file, 'r') as f:
                    iter_res = [list(map(float, x.strip().split(',')))
                                for x in f.readlines()]
                # tmp_iter_res = np.array(iter_res[1:])-np.array(iter_res[0:len(iter_res)-1])
                # tmp_iter_res = tmp_iter_res.tolist()
                # iter_res[1:]=tmp_iter_res
                iternums = []
                for iter_ in iter_res[1:]:
                    iternums.append(iter_)
                if store:
                    self.iter_res[name] = iternums
                else:
                    return iternums
            else:
                print(iter_file)
        self.tracker_names = list(self.iter_res.keys())

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


class UAVDataset(Dataset):
    """
    Args:
        name: dataset name, should be 'UAV123', 'UAV20L'
        dataset_root: dataset root
        load_img: wether to load all imgs
    """

    def __init__(self, name, dataset_root, load_img=False):
        super(UAVDataset, self).__init__(name, dataset_root)
        with open(os.path.join(dataset_root, name + '.json'), 'r') as f:
            meta_data = json.load(f)

        # load videos
        pbar = tqdm(meta_data.keys(), desc='loading ' + name, ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)
            self.videos[video] = UAVVideo(video,
                                          dataset_root,
                                          meta_data[video]['video_dir'],
                                          meta_data[video]['init_rect'],
                                          meta_data[video]['img_names'],
                                          meta_data[video]['gt_rect'],
                                          meta_data[video]['attr'])

        # set attr
        attr = []
        for x in self.videos.values():
            attr += x.attr
        attr = set(attr)
        self.attr = {}
        self.attr['ALL'] = list(self.videos.keys())
        for x in attr:
            self.attr[x] = []
        for k, v in self.videos.items():
            for attr_ in v.attr:
                self.attr[attr_].append(k)
