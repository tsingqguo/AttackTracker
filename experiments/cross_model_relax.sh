#!/usr/bin/env bash
if [ "$2" = "alex" ];
then
    exp_path=/home/guoqing/projects/AttackTrack/experiments/siamrpn_alex_dwxcorr/
elif [ "$2" = "mobile" ];
then
    exp_path=/home/guoqing/projects/AttackTrack/experiments/siamrpn_mobilev2_l234_dwxcorr/
elif [ "$2" = "r50" ];
then
    exp_path=/home/guoqing/projects/AttackTrack/experiments/siamrpn_r50_l234_dwxcorr/
elif [ "$2" = "alex_online" ];
then
    exp_path=/home/guoqing/projects/AttackTrack/experiments/dsiamrpn_alex_dwxcorr/
elif [ "$2" = "mobile_online" ];
then
    exp_path=/home/guoqing/projects/AttackTrack/experiments/dsiamrpn_mobilev2_l234_dwxcorr/
elif [ "$2" = "r50_online" ];
then
    exp_path=/home/guoqing/projects/AttackTrack/experiments/dsiamrpn_r50_l234_dwxcorr/
elif [ "$2" = "eco" ];
then
    exp_path=/home/guoqing/projects/AttackTrack/experiments/eco/
elif [ "$2" = "kcf" ];
then
    exp_path=/home/guoqing/projects/AttackTrack/experiments/kcf/
elif [ "$2" = "mosse" ];
then
    exp_path=/home/guoqing/projects/AttackTrack/experiments/mosse/
elif [ "$2" = "strcf" ];
then
    exp_path=/home/guoqing/projects/AttackTrack/experiments/strcf/
elif [ "$2" = "bacf" ];
then
    exp_path=/home/guoqing/projects/AttackTrack/experiments/bacf/
fi

attackmodel=$1
echo $exp_path
echo "ATTACKing $2 with $1"
export CUDA_VISIBLE_DEVICES=$4
if [ "$3" = "ua" ];
then
    cd $exp_path
    export PYTHONPATH=/home/guoqing/projects/AttackTrack:$PYTHONPATH

    if [ "$5" = "eval" ];
    then
        python ../../tools/ua_eval.py 	 \
              --tracker_path ./results \
              --dataset OTB100 \
              --num 1 \
              --tracker_prefix model_OIM_UA_noFLow_noADAPT30_APTS2_noSPERT_REGL21_NORML_inf_relax_eplison_$6_$1 \
              --pert_type MAP

    elif [ "$5" = "nogtbox" ];
    then
        python -u ../../tools/attack_relax.py \
           --snapshot model.pth \
           --dataset OTB100 \
           --config config.yaml \
           --interval 30 \
           --apts \
           --att_method OIM \
           --att_type UA \
           --eplison 2.0 \
           --configC $attackmodel \
           --name_suffix nogtbox

    elif [ "$5" = "nogtbox_detect" ];
    then
        python -u ../../tools/attack_relax.py \
           --snapshot model.pth \
           --dataset OTB100 \
           --config config.yaml \
           --interval 30 \
           --apts \
           --att_method OIM \
           --att_type UA \
           --eplison 2.0 \
           --configC $attackmodel \
           --name_suffix nogtbox_detect

    elif [ "$5" = "nogtbox_eval" ];
    then
        python ../../tools/ua_eval.py 	 \
              --tracker_path ./results \
              --dataset OTB100 \
              --num 1 \
              --tracker_prefix model_OIM_UA_noFLow_noADAPT30_APTS2_noSPERT_REGL21_NORML_inf_nogtbox \
              --pert_type MAP

    elif [ "$5" = "cf" ];
    then
        python -u ../../tools/attack_relax.py \
           --snapshot model.pth \
           --dataset OTB100 \
           --config config.yaml \
           --interval 30 \
           --apts \
           --att_method OIM \
           --att_type UA \
           --eplison $6 \
           --configC $attackmodel \
           --name_suffix relax_eplison_$6
    else
        python -u ../../tools/attack_relax.py \
               --snapshot model.pth \
               --dataset OTB100 \
               --config config.yaml \
               --interval 30 \
               --apts \
               --att_method OIM \
               --att_type UA \
               --eplison 0.6 \
               --configC $attackmodel \
               --name_suffix relax_eplison_06
    fi

elif [ "$3" = "ta" ];
then
    cd $exp_path
    export PYTHONPATH=/home/guoqing/projects/AttackTrack:$PYTHONPATH
    if [ "$5" = "eval" ];
    then
        python ../../tools/ta_eval.py 	 \
              --tracker_path ./results \
              --dataset OTB100 \
              --num 1 \
              --tracker_prefix model_OIM_TA_noFLow_noADAPT30_APTS2_noSPERT_REGL21_NORML_inf_relax_eplison_1_$1 \
              --pert_type MAP
    elif [ "$5" = "nogtbox_eval" ];
    then
        python ../../tools/ta_eval.py 	 \
              --tracker_path ./results \
              --dataset OTB100 \
              --num 1 \
              --tracker_prefix model_OIM_TA_noFLow_noADAPT30_APTS2_noSPERT_REGL21_NORML_inf_nogtbox \
              --pert_type MAP
    else
        python -u ../../tools/attack_relax.py \
               --snapshot model.pth \
               --dataset OTB100 \
               --config config.yaml \
               --interval 30 \
               --apts \
               --att_method OIM \
               --att_type TA \
               --eplison 0.6 \
               --configC $attackmodel \
               --name_suffix relax_eplison_06
    fi

elif [ "$3" = "uata" ];
then
    cd $exp_path
    export PYTHONPATH=/home/guoqing/projects/AttackTrack:$PYTHONPATH

    if [ "$5" = "eval" ];
    then
        python ../../tools/ua_eval.py 	 \
          --tracker_path ./results \
          --dataset OTB100 \
          --num 1 \
          --tracker_prefix model_OIM_UA_noFLow_noADAPT30_APTS2_noSPERT_REGL21_NORML_inf_relax_eplison_1_$1 \
          --pert_type MAP

        python ../../tools/ta_eval.py 	 \
          --tracker_path ./results \
          --dataset OTB100 \
          --num 1 \
          --tracker_prefix model_OIM_TA_noFLow_noADAPT30_APTS2_noSPERT_REGL21_NORML_inf_relax_eplison_1_$1 \
          --pert_type MAP

    elif [ "$5" = "nogtbox" ];
    then
        python -u ../../tools/attack_relax.py \
           --snapshot model.pth \
           --dataset OTB100 \
           --config config.yaml \
           --interval 30 \
           --apts \
           --att_method OIM \
           --att_type UA \
           --eplison 1.0 \
           --configC $attackmodel \
           --name_suffix nogtbox_e1

        python -u ../../tools/attack_relax.py \
           --snapshot model.pth \
           --dataset OTB100 \
           --config config.yaml \
           --interval 30 \
           --apts \
           --att_method OIM \
           --att_type TA \
           --eplison 1.0 \
           --configC $attackmodel \
           --name_suffix nogtbox_e1

    elif [ "$5" = "nogtbox_detect" ];
    then
        python -u ../../tools/attack_relax.py \
           --snapshot model.pth \
           --dataset OTB100 \
           --config config.yaml \
           --interval 30 \
           --apts \
           --att_method OIM \
           --att_type UA \
           --eplison 10 \
           --configC $attackmodel \
           --name_suffix nogtbox_detect_e1

        python -u ../../tools/attack_relax.py \
           --snapshot model.pth \
           --dataset OTB100 \
           --config config.yaml \
           --interval 30 \
           --apts \
           --att_method OIM \
           --att_type TA \
           --eplison 1.0 \
           --configC $attackmodel \
           --name_suffix nogtbox_detect_e1

    else
        python -u ../../tools/attack_relax.py \
               --snapshot model.pth \
               --dataset OTB100 \
               --config config.yaml \
               --interval 30 \
               --apts \
               --att_method OIM \
               --att_type UA \
               --eplison 0.6 \
               --configC $attackmodel \
               --name_suffix relax_eplison_06

        python -u ../../tools/attack_relax.py \
               --snapshot model.pth \
               --dataset OTB100 \
               --config config.yaml \
               --interval 30 \
               --apts \
               --att_method OIM \
               --att_type TA \
               --eplison 0.6 \
               --configC $attackmodel \
               --name_suffix relax_eplison_06
    fi
fi