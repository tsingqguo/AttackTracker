exp_path=/media/tsingqguo/data/AttackTracker/experiments/siamrpn_$1_dwxcorr/
attack_type1='UA'
attack_type2='TA'

echo "Processing OIM"

cd $exp_path

#export PYTHONPATH=/home/vil/Desktop/Object_Tracking/AttackTracker:$PYTHONPATH
export PYTHONPATH=/media/tsingqguo/data/AttackTracker:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$4

python -u ../../tools/attack_oim.py \
       --snapshot model.pth \
       --dataset $2 \
       --config config.yaml \
       --interval 30 \
       --apts \
       --att_method OIM \
       --norm_type $3 \
       --att_type $attack_type1


cd $exp_path

#export PYTHONPATH=/home/vil/Desktop/Object_Tracking/AttackTracker:$PYTHONPATH
export PYTHONPATH=/media/tsingqguo/data/AttackTracker:$PYTHONPATH

python -u ../../tools/attack_oim.py \
       --snapshot model.pth \
       --dataset $2 \
       --config config.yaml \
       --interval 30 \
       --apts \
       --att_method OIM \
       --norm_type $3 \
       --att_type $attack_type2




