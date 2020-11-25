#exp_path=/media/tsingqguo/data/AttackTracker/experiments/siamrpn_r50_l234_dwxcorr/
exp_path=/media/tsingqguo/data/AttackTracker/experiments/siamrpn_$1_dwxcorr/
method_type=$2
attack_type1='UA'
attack_type2='TA'

if [ ! -n "$4" ]; then
   name_suffix=''
else
   name_suffix=$4$5
fi

echo "Processing $method_type with $name_suffix"

cd $exp_path

#export PYTHONPATH=/home/vil/Desktop/Object_Tracking/AttackTracker:$PYTHONPATH
export PYTHONPATH=/media/tsingqguo/data/AttackTracker:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$6

python -u ../../tools/attack_oim.py \
       --snapshot model.pth \
       --dataset $3 \
       --config config.yaml \
       --interval 30 \
       --apts \
       --att_method $method_type \
       --att_type $attack_type1 \
       --accframes $5 \
       --name_suffix $name_suffix


cd $exp_path

#export PYTHONPATH=/home/vil/Desktop/Object_Tracking/AttackTracker:$PYTHONPATH
export PYTHONPATH=/media/tsingqguo/data/AttackTracker:$PYTHONPATH

python -u ../../tools/attack_oim.py \
       --snapshot model.pth \
       --dataset $3 \
       --config config.yaml \
       --interval 30 \
       --apts \
       --att_method $method_type \
       --att_type $attack_type2 \
       --accframes $5 \
       --name_suffix $name_suffix




