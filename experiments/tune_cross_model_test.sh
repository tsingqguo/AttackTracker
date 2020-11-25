#!/usr/bin/env bash
#for eps in `seq 0.4 0.2 1.0`
#do
eps=$1
#bash cross_model_test.sh mobile alex ua $eps $2 $3 $4
#bash cross_model_test.sh alex mobile ua $eps $2 $3 $4
#bash cross_model_test.sh mobile r50 ua $eps $2 $3 $4
bash cross_model_test.sh r50 mobile ua $eps $2 $3 $4
#done