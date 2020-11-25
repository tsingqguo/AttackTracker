
#
method=siamrpn_r50_l234_dwxcorr #siamrpn_mobilev2_l234_dwxcorr #siamdwnext22 #siamdwres22 #
model=model.pth #CIRNext22.pth #CIResNet22.pth #
#
#method=siamdwincep22 #siamdwnext22 #siamdwres22 #
#model=CIRIncep22.pth #CIRNext22.pth #CIResNet22.pth #
#
exp_path=/media/tsingqguo/data/AttackTracker/experiments/$method/
export PYTHONPATH=/media/tsingqguo/data/AttackTracker:$PYTHONPATH

cd $exp_path

python -u ../../tools/test_us.py \
       --snapshot $model \
       --dataset OTB100 \
       --config config.yaml