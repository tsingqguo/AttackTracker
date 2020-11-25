#!/usr/bin/env bash

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
fi


attackmodel=$1
echo $exp_path
echo "ATTACKing $2 with $1 under $3 $7"
export CUDA_VISIBLE_DEVICES=$6
if [ "$3" = "ua" ];
then
    cd $exp_path
    export PYTHONPATH=/media/tsingqguo/data/AttackTracker:$PYTHONPATH

    if [ $7 = "eval" ];
    then
        python ../../tools/ua_eval.py 	 \
        --tracker_path ./results \
        --dataset OTB100 \
        --num 1 \
        --tracker_prefix model_OIM_UA_noFLow_noADAPT30_APTS2_noSPERT_REGL21_NORML_inf$5_$1 \
        --pert_type MAP
    else

        python -u ../../tools/attack_cross.py \
       --snapshot model.pth \
       --dataset OTB100 \
       --config config.yaml \
       --interval 30 \
       --apts \
       --att_method OIM \
       --att_type UA \
       --eplison $4 \
       --name_suffix $5 \
       --configC $attackmodel
    fi

elif [ "$3" = "ta" ];
then
    cd $exp_path
    export PYTHONPATH=/media/tsingqguo/data/AttackTracker:$PYTHONPATH
    python -u ../../tools/attack_cross.py \
           --snapshot model.pth \
           --dataset OTB100 \
           --config config.yaml \
           --interval 30 \
           --apts \
           --att_method OIM \
           --att_type TA \
           --eplison 3e-1 \
           --name_suffix $5 \
           --configC $attackmodel

elif [ "$3" = "uata" ];
then
    cd $exp_path
    export PYTHONPATH=/media/tsingqguo/data/AttackTracker:$PYTHONPATH
    python -u ../../tools/attack_cross.py \
           --snapshot model.pth \
           --dataset OTB100 \
           --config config.yaml \
           --interval 30 \
           --apts \
           --att_method OIM \
           --att_type UA \
           --eplison 3e-1 \
           --configC $attackmodel

    cd $exp_path
    export PYTHONPATH=/media/tsingqguo/data/AttackTracker:$PYTHONPATH
    python -u ../../tools/attack_cross.py \
           --snapshot model.pth \
           --dataset OTB100 \
           --config config.yaml \
           --interval 30 \
           --apts \
           --att_method OIM \
           --att_type TA \
           --eplison 3e-1 \
           --configC $attackmodel
fi
