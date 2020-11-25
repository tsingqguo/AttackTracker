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

if [ "$1" = "eco" ]
then
    attack_method=OIMECO
elif [ "$1" = "siamdwres22" ]
then
    attack_method=OIMSIAMDW
elif [ "$1" = "siamdwnext22" ]
then
    attack_method=OIMSIAMDW
elif [ "$1" = "siamdwincep22" ]
then
    attack_method=OIMSIAMDW
else
    attack_method=OIM
fi

test_eplison=6e-1 # default: 3e-1
attackmodel=$1
echo $exp_path
echo "Attacking $2 with $1"
export CUDA_VISIBLE_DEVICES=$4
if [ "$3" = "ua" ];
then
    cd $exp_path
    export PYTHONPATH=/media/tsingqguo/data/AttackTracker:$PYTHONPATH
    python -u ../../tools/attack_oim.py \
           --snapshot model.pth \
           --dataset OTB100 \
           --config config.yaml \
           --interval 30 \
           --apts \
           --att_method $attack_method \
           --att_type UA \
           --eplison $test_eplison

elif [ "$3" = "ta" ];
then
    cd $exp_path
    export PYTHONPATH=/media/tsingqguo/data/AttackTracker:$PYTHONPATH
    python -u ../../tools/attack_oim.py \
           --snapshot model.pth \
           --dataset OTB100 \
           --config config.yaml \
           --interval 30 \
           --apts \
           --att_method $attack_method \
           --att_type TA \
           --eplison $test_eplison

elif [ "$3" = "uata" ];
then
    cd $exp_path
    export PYTHONPATH=/media/tsingqguo/data/AttackTracker:$PYTHONPATH
    python -u ../../tools/attack_oim.py \
           --snapshot model.pth \
           --dataset OTB100 \
           --config config.yaml \
           --interval 30 \
           --apts \
           --att_method $attack_method \
           --att_type UA \
           --eplison $test_eplison

    cd $exp_path
    export PYTHONPATH=/media/tsingqguo/data/AttackTracker:$PYTHONPATH
    python -u ../../tools/attack_oim.py \
           --snapshot model.pth \
           --dataset OTB100 \
           --config config.yaml \
           --interval 30 \
           --apts \
           --att_method $attack_method \
           --att_type TA \
           --eplison $test_eplison
fi
