echo "Processing method accframes 30"

sh comparison_study_oim_tuneaccframes.sh alex OIM OTB100 accframes 30 0

sh eval.sh alex OTB100 model_OIM_UA_noFLow_noADAPT30_APTS2_noSPERT_REGL21_NORML_infaccframes > ./results/ta.txt

echo "Processing method accframes 25"

sh comparison_study_oim_tuneaccframes.sh alex OIM OTB100 accframes 25 0

echo "Processing method accframes 20"

sh comparison_study_oim_tuneaccframes.sh alex OIM OTB100 accframes 20 0

echo "Processing method accframes 15"

sh comparison_study_oim_tuneaccframes.sh alex OIM OTB100 accframes 15 0

echo "Processing method accframes 10"

sh comparison_study_oim_tuneaccframes.sh alex OIM OTB100 accframes 10 0

echo "Processing method accframes 5"

sh comparison_study_oim_tuneaccframes.sh alex OIM OTB100 accframes 5 0

echo "Processing method accframes 0"

sh comparison_study_oim_tuneaccframes.sh alex OIM OTB100 accframes 0 0

sh eval.sh alex VOT2018 model_OIM_TA_noFLow_noADAPT30_APTS2_noSPERT_REGL21_NORML_infacc30frames5e-2 > infacc30frames5e-2_res.txt