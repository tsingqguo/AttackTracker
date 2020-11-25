
# conclution: when eplison=2e-1, MAP tends to be reasonable
# $1 for model $2 for dataset $3 for gpu id
echo "Processing method $1 $2 with gpu $3"

sh comparison_study_oim.sh $1 OIM $2 acc30frames 3e-1 $3

sh eval.sh $1 $2 model_OIM_UA_noFLow_noADAPT30_APTS2_noSPERT_REGL21_NORML_infacc30frames3e-1 > ./results/OIA_$2/ua_infacc30frames3e-1_$1.txt
sh eval.sh $1 $2 model_OIM_TA_noFLow_noADAPT30_APTS2_noSPERT_REGL21_NORML_infacc30frames3e-1 > ./results/OIA_$2/ta_infacc30frames3e-1_$1.txt



