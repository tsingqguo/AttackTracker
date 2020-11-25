#!/usr/bin/env bash
#exp_path=/media/tsingqguo/data/AttackTracker/experiments/siamrpn_r50_l234_dwxcorr/
exp_path=/media/tsingqguo/data/AttackTracker/experiments/siamrpn_$1_dwxcorr/
method_type=$2
attack_type1='UA'
attack_type2='TA'

#if [ ! -n "$4" ]; then
#   name_suffix=''
#else
#   name_suffix=$4$5
#fi

echo "Processing $method_type"

cd $exp_path

#export PYTHONPATH=/home/vil/Desktop/Object_Tracking/AttackTracker:$PYTHONPATH
export PYTHONPATH=/media/tsingqguo/data/AttackTracker:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$5

if [ $6 = "eval" ];
then
    echo $exp_path
    model_name=model_OIM_TA_noFLow_noADAPT30_APTS2_abc
#    python ../../tools/ua_eval.py 	 \
#        --tracker_path ./results \
#        --dataset $3       \
#        --num 1 		 \
#        --tracker_prefix $model_name \
#        --pert_type P

    python ../../tools/ta_eval.py 	 \
        --tracker_path ./results \
        --dataset $3       \
        --num 1 		 \
        --tracker_prefix $model_name \
        --pert_type MAP

else

    python -u ../../tools/attack_oim.py \
           --snapshot model.pth \
           --dataset $3 \
           --config config.yaml \
           --interval 30 \
           --apts \
           --att_method $method_type \
           --att_type $attack_type1 \
           --reg_type $4
fi


#cd $exp_path
#
##export PYTHONPATH=/home/vil/Desktop/Object_Tracking/AttackTracker:$PYTHONPATH
#export PYTHONPATH=/media/tsingqguo/data/AttackTracker:$PYTHONPATH
#
#python -u ../../tools/attack_oim.py \
#       --snapshot model.pth \
#       --dataset $3 \
#       --config config.yaml \
#       --interval 30 \
#       --apts \
#       --att_method $method_type \
#       --att_type $attack_type2 \
#       --eplison $5 \
#       --name_suffix $name_suffix




