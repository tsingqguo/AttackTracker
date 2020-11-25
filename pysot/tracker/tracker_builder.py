# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.core.config import cfg
from pysot.tracker.siamrpn_tracker import SiamRPNTracker
from pysot.tracker.siammask_tracker import SiamMaskTracker
from pysot.tracker.siamrpnlt_tracker import SiamRPNLTTracker
from pysot.tracker.dsiamrpn_tracker import DSiamRPNTracker
from pysot.tracker.dimp_tracker import DiMPTracker
from pysot.tracker.atom_tracker import ATOMTracker
from pysot.tracker.eco_tracker import ECOTracker
from pysot.tracker.siamdw_tracker import SiamDWTracker
from pysot.tracker.cf_tracker import CFTracker


import os
import pickle
import importlib

TRACKS = {
          'SiamRPNTracker': SiamRPNTracker,
          'SiamMaskTracker': SiamMaskTracker,
          'SiamRPNLTTracker': SiamRPNLTTracker,
          'DSiamRPNTracker': DSiamRPNTracker,
          'DiMPTracker': DiMPTracker,
          'ATOMTracker': ATOMTracker,
          'ECOTracker': ECOTracker,
          'SiamDWTracker': SiamDWTracker,
          'CFTracker':CFTracker
         }


def build_tracker(model):
    if cfg.TRACK.TYPE in ['DiMPTracker','ATOMTracker','ECOTracker']:
        params = get_parameters()
        return TRACKS[cfg.TRACK.TYPE](params)
    if cfg.TRACK.TYPE=='CFTracker':
        return TRACKS[cfg.TRACK.TYPE]()
    else:
        return TRACKS[cfg.TRACK.TYPE](model)

def get_parameters():
    """Get parameters."""

    parameter_file = '{}/parameters.pkl'.format(cfg.PYTRACKING.PARAM_DIR)
    if os.path.isfile(parameter_file):
        return pickle.load(open(parameter_file, 'rb'))

    param_module = importlib.import_module('extern.pytracking.parameter.{}.{}'.format(cfg.PYTRACKING.TRACKER_NAME, cfg.PYTRACKING.PARAM_NAME))
    params = param_module.parameters()

    #pickle.dump(params, open(parameter_file, 'wb'))

    return params
