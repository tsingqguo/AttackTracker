# conclution: when eplison=2e-1, MAP tends to be reasonable
# $1 : gpu id
echo "Processing method ECO, Dataset OTB100, with gpu $1"

bash white_model_test.sh siamdwres22 siamdwres22 ua $1

bash eval.sh OTB100 siamdwres22 CIResNet22 org
bash eval.sh OTB100 siamdwres22 model_OIMSIAMDW_UA_noFLow_noADAPT30_APTS2_noSPERT_REGL21_NORML_inf ua

bash white_model_test.sh siamdwnext22 siamdwnext22 ua $1
bash eval.sh OTB100 siamdwnext22 CIRNext22 org
bash eval.sh OTB100 siamdwnext22 model_OIMSIAMDW_UA_noFLow_noADAPT30_APTS2_noSPERT_REGL21_NORML_inf ua


bash white_model_test.sh siamdwincep22 siamdwincep22 ua $1
bash eval.sh OTB100 siamdwincep22 CIRIncep22 org
bash eval.sh OTB100 siamdwincep22 model_OIMSIAMDW_UA_noFLow_noADAPT30_APTS2_noSPERT_REGL21_NORML_inf ua
