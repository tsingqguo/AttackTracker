#!/usr/bin/env bash
exp_path=/home/vil/Desktop/Object_Tracking/AttackTracker/experiments/siamrpn_mobilev2_l234_dwxcorr/ #siamrpn_mobilev2_l234_dwxcorr siamrpn_alex_dwxcorr
method_type='OIM' # 'OIM'
attack_type='UA'

cd $exp_path

export PYTHONPATH=/home/vil/Desktop/Object_Tracking/AttackTracker:$PYTHONPATH

python -u ../../tools/attack_oim.py \
	--snapshot model.pth \
	--dataset OTB100 \
	--config config.yaml \
	--interval 30 \
	--apts \
	--att_method $method_type \
	--att_type $attack_type




