#!/usr/bin/env bash
# example: sh eval.sh OTB100 mobile model_OIM_UA_noFLow_noADAPT30_APTS2_noSPERT_REGL21_NORML_inf_alex ua
export PYTHONPATH=/media/tsingqguo/data/AttackTracker:$PYTHONPATH

if [ "$2" = "alex" ];
then
    exp_path=/media/tsingqguo/data/AttackTracker/experiments/siamrpn_alex_dwxcorr/
elif [ "$2" = "mobile" ];
then
    exp_path=/media/tsingqguo/data/AttackTracker/experiments/siamrpn_mobilev2_l234_dwxcorr/
elif [ "$2" = "r50" ];
then
    exp_path=/media/tsingqguo/data/AttackTracker/experiments/siamrpn_r50_l234_dwxcorr/
elif [ "$2" = "alex_online" ];
then
    exp_path=/media/tsingqguo/data/AttackTracker/experiments/dsiamrpn_alex_dwxcorr/
elif [ "$2" = "mobile_online" ];
then
    exp_path=/media/tsingqguo/data/AttackTracker/experiments/dsiamrpn_mobilev2_l234_dwxcorr/
elif [ "$2" = "r50_online" ];
then
    exp_path=/media/tsingqguo/data/AttackTracker/experiments/dsiamrpn_r50_l234_dwxcorr/
elif [ "$2" = "eco" ];
then
    exp_path=/media/tsingqguo/data/AttackTracker/experiments/eco/
elif [ "$2" = "siamdwres22" ];
then
    exp_path=/media/tsingqguo/data/AttackTracker/experiments/siamdwres22/
elif [ "$2" = "siamdwnext22" ];
then
    exp_path=/media/tsingqguo/data/AttackTracker/experiments/siamdwnext22/
elif [ "$2" = "siamdwincep22" ];
then
    exp_path=/media/tsingqguo/data/AttackTracker/experiments/siamdwincep22/
fi

eval_dataset=$1
prefix=$3 #model_OIM_UA_noFLow_noADAPT30_APTS2_noSPERT_REGL21_NORML_inf_alex #model_OIM_UA_noFLow_noADAPT30_APTS2_noSPERT_REGL21_NORML_infaccframes30 #$3

cd $exp_path

if [ "$4" = "ua" ];
then
    python ../../tools/ua_eval.py 	 \
      --tracker_path ./results \
      --dataset $eval_dataset        \
      --num 1 		 \
      --tracker_prefix $prefix \
      --pert_type MAP

elif [ "$4" = "ta" ];
then
    python ../../tools/ta_eval.py 	 \
    --tracker_path ./results \
    --dataset $eval_dataset        \
    --num 1 		 \
    --tracker_prefix $prefix \
    --pert_type MAP
elif [ "$4" = "uata" ];
then
    python ../../tools/ua_eval.py 	 \
    --tracker_path ./results \
    --dataset $eval_dataset        \
    --num 1 		 \
    --tracker_prefix $prefix \
    --pert_type MAP
    #
    python ../../tools/ta_eval.py 	 \
    --tracker_path ./results \
    --dataset $eval_dataset        \
    --num 1 		 \
    --tracker_prefix $prefix \
    --pert_type MAP
elif [ "$4" = "org" ];
then
    python ../../tools/eval.py 	 \
    --tracker_path ./results \
    --dataset $eval_dataset  \
    --num 1 		 \
    --tracker_prefix $prefix
fi