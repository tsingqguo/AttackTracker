# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from pysot.attacker.oim_attacker import OIMAttacker

class FGSMAttacker(OIMAttacker):
    def __init__(self,type,max_num=1,eplison=1,inta=10,lamb=0.0001,norm_type='L_inf',apts_num=2,reg_type='L21',accframes=30):
        max_num = 1
        self.type = type
        self.eplison = eplison
        self.inta = inta
        self.norm_type = norm_type
        self.max_num = max_num
        self.v_id = 0
        self.apts_num = 1
        self.target_traj=[]
        self.prev_delta = None
        self.tacc = False
        self.lamb = 0  # remove the L2,1 regularization
        self.reg_type= 'None'
        self.acc_iters = 0
        self.weight_eplison = 1
        self.accframes = accframes


