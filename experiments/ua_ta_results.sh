#!/usr/bin/env bash
exp_path=/home/vil/Desktop/Object_Tracking/AttackTracker/experiments/siamrpn_alex_dwxcorr/ #siamrpn_mobilev2_l234_dwxcorr
method_type='OIM'
attack_type=$1

cd $exp_path
export PYTHONPATH=/home/vil/Desktop/Object_Tracking/AttackTracker:$PYTHONPATH

python -u ../../tools/attack_oim.py \
	--snapshot model.pth \
	--dataset $2 \
	--config config.yaml \
	--interval 30 \
	--apts \
	--apts_num 2 \
	--att_method $method_type \
	--att_type $attack_type \
	--reg_type L21

python -u ../../tools/attack_oim.py \
	--snapshot model.pth \
	--dataset $2 \
	--config config.yaml \
	--interval 30 \
	--apts \
	--apts_num 2 \
	--att_method $method_type \
	--att_type $attack_type \
	--reg_type L21 \
    --enable_same_pert