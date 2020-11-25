exp_path1=/home/vil/Desktop/Object_Tracking/AttackTracker/experiments/siamrpn_alex_dwxcorr/
exp_path2=/home/vil/Desktop/Object_Tracking/AttackTracker/experiments/siamrpn_mobilev2_l234_dwxcorr/
exp_path3=/home/vil/Desktop/Object_Tracking/AttackTracker/experiments/siamrpn_r50_l234_dwxcorr/
method_type='OIM' # 'OIM'
attack_type='TA'

cd $exp_path1
export PYTHONPATH=/home/vil/Desktop/Object_Tracking/AttackTracker:$PYTHONPATH

python -u ../../tools/attack_oim.py \
	--snapshot model.pth \
	--dataset OTB100 \
	--config config.yaml \
	--interval 30 \
	--apts \
	--att_method $method_type \
	--att_type $attack_type


cd $exp_path2
export PYTHONPATH=/home/vil/Desktop/Object_Tracking/AttackTracker:$PYTHONPATH

python -u ../../tools/attack_oim.py \
	--snapshot model.pth \
	--dataset OTB100 \
	--config config.yaml \
	--interval 30 \
	--apts \
	--att_method $method_type \
	--att_type $attack_type


cd $exp_path3
export PYTHONPATH=/home/vil/Desktop/Object_Tracking/AttackTracker:$PYTHONPATH

python -u ../../tools/attack_oim.py \
	--snapshot model.pth \
	--dataset OTB100 \
	--config config.yaml \
	--interval 30 \
	--apts \
	--att_method $method_type \
	--att_type $attack_type






