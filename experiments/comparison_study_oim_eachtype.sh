#!/usr/bin/env bash
#exp_path=/media/tsingqguo/data/AttackTracker/experiments/siamrpn_r50_l234_dwxcorr/
exp_path=/media/tsingqguo/data/AttackTracker/experiments/siamrpn_$1_dwxcorr/
# 'OIM' 'FGSM' 'BIM' 'MIFGSM' 'CW-L2'
attack_type=$3
method_type=$2

echo "Processing $method_type"

cd $exp_path

#export PYTHONPATH=/home/vil/Desktop/Object_Tracking/AttackTracker:$PYTHONPATH
export PYTHONPATH=/media/tsingqguo/data/AttackTracker:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$5

python -u ../../tools/attack_oim.py \
       --snapshot model.pth \
       --dataset $4 \
       --config config.yaml \
       --interval 30 \
       --apts \
       --att_method $method_type \
       --att_type $attack_type


