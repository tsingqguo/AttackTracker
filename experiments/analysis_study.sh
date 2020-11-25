export PYTHONPATH=/home/vil/Desktop/Object_Tracking/AttackTracker:$PYTHONPATH
exp_path=/home/vil/Desktop/Object_Tracking/AttackTracker/experiments/siamrpn_alex_dwxcorr/
cd $exp_path

echo "Running: BA-E"
python -u ../../tools/attack_analysis.py \
	--snapshot model.pth \
	--dataset OTB100 \
	--config config.yaml \
	--interval 1 \
	--att_method $1 \
	--att_type $2 \
	--reg_type None \
	--name_suffix analysis-BA-E

echo "Running: BA-R1"
python -u ../../tools/attack_analysis.py \
	--snapshot model.pth \
	--dataset OTB100 \
	--config config.yaml \
	--interval 10 \
	--att_method $1 \
	--att_type $2 \
	--reg_type None \
	--name_suffix analysis-BA-R1

echo "Running: BA-2-Random"
python -u ../../tools/attack_analysis.py \
	--snapshot model.pth \
	--dataset OTB100 \
	--config config.yaml \
	--interval -2 \
	--att_method $1 \
	--att_type $2 \
	--reg_type None \
	--name_suffix analysis

echo "Running: BA-3"
python -u ../../tools/attack_analysis.py \
	--snapshot model.pth \
	--dataset OTB100 \
	--config config.yaml \
	--interval 10 \
	--apts \
	--apts_num 4 \
	--att_method $1 \
	--att_type $2 \
	--reg_type None \
	--name_suffix analysis


