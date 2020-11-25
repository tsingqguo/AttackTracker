    # Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import math
import time
import torch

from pysot.core.config import cfg

from extern.pycftrackers.lib.utils import get_img_list,get_ground_truthes
from extern.pycftrackers.cftracker.mosse import MOSSE
from extern.pycftrackers.cftracker.csk import CSK
from extern.pycftrackers.cftracker.kcf import KCF
from extern.pycftrackers.cftracker.cn import CN
from extern.pycftrackers.cftracker.dsst import DSST
from extern.pycftrackers.cftracker.staple import Staple
from extern.pycftrackers.cftracker.dat import DAT
from extern.pycftrackers.cftracker.eco import ECO
from extern.pycftrackers.cftracker.bacf import BACF
from extern.pycftrackers.cftracker.csrdcf import CSRDCF
from extern.pycftrackers.cftracker.samf import SAMF
from extern.pycftrackers.cftracker.ldes import LDES
from extern.pycftrackers.cftracker.mkcfup import MKCFup
from extern.pycftrackers.cftracker.strcf import STRCF
from extern.pycftrackers.cftracker.mccth_staple import MCCTHStaple
from extern.pycftrackers.lib.eco.config import otb_deep_config,otb_hc_config
from extern.pycftrackers.cftracker.config import staple_config,ldes_config,dsst_config,csrdcf_config,mkcf_up_config,mccth_staple_config

class CFTracker():

    def __init__(self):

        self.tracker_type = cfg.CFTRACKING.TRACKER_TYPE

        if self.tracker_type == 'MOSSE':
            self.tracker=MOSSE()
        elif self.tracker_type=='CSK':
            self.tracker=CSK()
        elif self.tracker_type=='CN':
            self.tracker=CN()
        elif self.tracker_type=='DSST':
            self.tracker=DSST(dsst_config.DSSTConfig())
        elif self.tracker_type=='Staple':
            self.tracker=Staple(config=staple_config.StapleConfig())
        elif self.tracker_type=='Staple-CA':
            self.tracker=Staple(config=staple_config.StapleCAConfig())
        elif self.tracker_type=='KCF_CN':
            self.tracker=KCF(features='cn',kernel='gaussian')
        elif self.tracker_type=='KCF_GRAY':
            self.tracker=KCF(features='gray',kernel='gaussian')
        elif self.tracker_type=='KCF_HOG':
            self.tracker=KCF(features='hog',kernel='gaussian')
        elif self.tracker_type=='DCF_GRAY':
            self.tracker=KCF(features='gray',kernel='linear')
        elif self.tracker_type=='DCF_HOG':
            self.tracker=KCF(features='hog',kernel='linear')
        elif self.tracker_type=='DAT':
            self.tracker=DAT()
        elif self.tracker_type=='ECO-HC':
            self.tracker=ECO(config=otb_hc_config.OTBHCConfig())
        elif self.tracker_type=='ECO':
            self.tracker=ECO(config=otb_deep_config.OTBDeepConfig())
        elif self.tracker_type=='BACF':
            self.tracker=BACF()
        elif self.tracker_type=='CSRDCF':
            self.tracker=CSRDCF(config=csrdcf_config.CSRDCFConfig())
        elif self.tracker_type=='CSRDCF-LP':
            self.tracker=CSRDCF(config=csrdcf_config.CSRDCFLPConfig())
        elif self.tracker_type=='SAMF':
            self.tracker=SAMF()
        elif self.tracker_type=='LDES':
            self.tracker=LDES(ldes_config.LDESDemoLinearConfig())
        elif self.tracker_type=='DSST-LP':
            self.tracker=DSST(dsst_config.DSSTLPConfig())
        elif self.tracker_type=='MKCFup':
            self.tracker=MKCFup(config=mkcf_up_config.MKCFupConfig())
        elif self.tracker_type=='MKCFup-LP':
            self.tracker=MKCFup(config=mkcf_up_config.MKCFupLPConfig())
        elif self.tracker_type=='STRCF':
            self.tracker=STRCF()
        elif self.tracker_type=='MCCTH-Staple':
            self.tracker=MCCTHStaple(config=mccth_staple_config.MCCTHOTBConfig())
        else:
            raise NotImplementedError

    def init(self, im_narray, bbox):
        init_gt=tuple(bbox)
        self.tracker.init(im_narray,init_gt)

    def track(self, image):

        current_frame = image
        bbox,max_score = self.tracker.update(current_frame)

        return {
                'bbox': bbox,
                'best_score': max_score
               }



