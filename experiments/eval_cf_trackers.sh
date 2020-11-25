#!/usr/bin/env bash
# example: sh eval.sh OTB100 mobile model_OIM_UA_noFLow_noADAPT30_APTS2_noSPERT_REGL21_NORML_inf_alex ua
bash cross_model_relax.sh alex $1 ua $2 cf 1
bash cross_model_relax.sh mobile $1 ua $2 cf 1
bash cross_model_relax.sh r50 $1 ua $2 cf 1