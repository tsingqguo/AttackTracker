#!/usr/bin/env bash
#export PYTHONPATH=/home/vil/Desktop/Object_Tracking/AttackTracker:$PYTHONPATH
#exp_path=/home/vil/Desktop/Object_Tracking/AttackTracker/experiments/siamrpn_r50_l234_dwxcorr/
export PYTHONPATH=/home/guoqing/projects/AttackTrack:$PYTHONPATH
exp_path=/home/guoqing/projects/AttackTrack/experiments/$3/

cd $exp_path

python ../../tools/eval.py 	 \
	--tracker_path ./results \
	--dataset $1        \
	--num 1 		 \
	--tracker_prefix $2 \
