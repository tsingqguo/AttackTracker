#!/usr/bin/env bash
exp_path=/media/tsingqguo/data/AttackTracker/experiments/siamrpn_$4_dwxcorr/
export PYTHONPATH=/media/tsingqguo/data/AttackTracker:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$5
cd $exp_path
# FGSM

if [ $6 = "eval" ]
then
#    python ../../tools/ua_eval.py 	 \
#        --tracker_path ./results \
#        --dataset $1        \
#        --num 1 		 \
#        --tracker_prefix model_FGSM_UA_noFLow_noADAPT30_APTS1_noSPERT_REGL21_NORML_1 \
#        --pert_type MAP

    python ../../tools/ua_eval.py 	 \
        --tracker_path ./results \
        --dataset $1        \
        --num 1 		 \
        --tracker_prefix model_MIFGSM_UA_noFLow_noADAPT30_APTS2_noSPERT_REGL21_NORML_11000.0 \
        --pert_type MAP
#
#    python ../../tools/ua_eval.py 	 \
#        --tracker_path ./results \
#        --dataset $1        \
#        --num 1 		 \
#        --tracker_prefix model_MIFGSM_UA_noFLow_noADAPT30_APTS2_noSPERT_REGL21_NORML_1 \
#        --pert_type MAP

else

#    python -u ../../tools/attack_oim.py \
#        --snapshot model.pth \
#        --dataset $1 \
#        --config config.yaml \
#        --interval 30 \
#        --apts \
#        --att_method 'FGSM' \
#        --norm_type 'L_1' \
#        --eplison $3 \
#        --att_type $2
#
#    python -u ../../tools/attack_oim.py \
#        --snapshot model.pth \
#        --dataset $1 \
#        --config config.yaml \
#        --interval 30 \
#        --apts \
#        --att_method 'BIM' \
#        --norm_type 'L_1' \
#        --eplison $3 \
#        --att_type $2

    python -u ../../tools/attack_oim.py \
        --snapshot model.pth \
        --dataset $1 \
        --config config.yaml \
        --interval 30 \
        --apts \
        --att_method 'MIFGSM' \
        --norm_type 'L_1' \
        --eplison $3 \
        --name_suffix $3 \
        --att_type $2
fi
#python
# -u ../../tools/attack_oim.py \
#	--snapshot model.pth \
#	--dataset $1 \
#	--config config.yaml \
#	--interval 30 \
#	--apts \
#	--att_method 'CW-L2' \
#	--att_type $2
#
#python -u ../../tools/attack_oim.py \
#	--snapshot model.pth \
#	--dataset $1 \
#	--config config.yaml \
#	--interval 30 \
#	--apts \
#	--att_method 'SAP' \
#	--att_type $2


