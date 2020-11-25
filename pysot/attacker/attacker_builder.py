# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.attacker.fgsm_attacker import FGSMAttacker
from pysot.attacker.bim_attacker import BIMAttacker
from pysot.attacker.oim_attacker import OIMAttacker
from pysot.attacker.oim_attacker_eco import OIMAttackerECO
from pysot.attacker.oim_attacker_siamdw import OIMAttackerSiamDW
from pysot.attacker.mifgsm_attacker import MIFGSMAttacker
from pysot.attacker.cw_attacker import CWAttacker
from pysot.attacker.sap_attacker import SAPAttacker
from pysot.attacker.curl_attacker import CURLAttacker

ATTACKERS = {
          'FGSM': FGSMAttacker,
          'BIM': BIMAttacker,
          'OIM':OIMAttacker,
          'OIMECO':OIMAttackerECO,
          'OIMSIAMDW': OIMAttackerSiamDW,
          'MIFGSM': MIFGSMAttacker,
          'CW-L2':CWAttacker,
          'SAP':SAPAttacker,
          'CURL':CURLAttacker
         }

def build_attacker(attacker_method,type,max_num=10,apts_num=2,inta=10,reg_type='L21',norm_type='L_inf',eplison=1,accframes=30):
        return ATTACKERS[attacker_method](type, max_num,inta=inta,apts_num=apts_num,reg_type=reg_type,norm_type=norm_type,eplison=eplison,accframes=accframes)
