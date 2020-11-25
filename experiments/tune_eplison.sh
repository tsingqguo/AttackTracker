
# conclution: when eplison=2e-1, MAP tends to be reasonable

echo "Processing method acc30frames 5e-2"

#sh comparison_study_oim.sh mobilev2_l234 OIM VOT2018 acc30frames 5e-2 0

sh eval.sh mobilev2_l234 VOT2018 model_OIM_TA_noFLow_noADAPT30_APTS2_noSPERT_REGL21_NORML_infacc30frames5e-2 > infacc30frames5e-2_res.txt

echo "Processing method acc30frames 1e-1"

#sh comparison_study_oim.sh mobilev2_l234 OIM VOT2018 acc30frames 1e-1 0

sh eval.sh mobilev2_l234 VOT2018 model_OIM_TA_noFLow_noADAPT30_APTS2_noSPERT_REGL21_NORML_infacc30frames1e-1 > infacc30frames1e-1_res.txt

echo "Processing method acc30frames 2e-1"

#sh comparison_study_oim.sh mobilev2_l234 OIM VOT2018 acc30frames 2e-1 0

sh eval.sh mobilev2_l234 VOT2018 model_OIM_TA_noFLow_noADAPT30_APTS2_noSPERT_REGL21_NORML_infacc30frames2e-1 > infacc30frames2e-1_res.txt

echo "Processing method acc30frames 3e-1"

#sh comparison_study_oim.sh mobilev2_l234 OIM VOT2018 acc30frames 3e-1 0

sh eval.sh mobilev2_l234 VOT2018 model_OIM_TA_noFLow_noADAPT30_APTS2_noSPERT_REGL21_NORML_infacc30frames3e-1 > infacc30frames3e-1_res.txt

