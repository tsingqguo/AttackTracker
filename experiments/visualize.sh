#exp_path=/media/tsingqguo/data/AttackTracker/experiments/siamrpn_$1_dwxcorr/
exp_path=/home/vil/Desktop/Object_Tracking/AttackTracker/experiments/siamrpn_$1_dwxcorr/

# visulalize 'OIM' 'BIM' 'MIFGSM'

echo "Processing $method_type"

cd $exp_path

export PYTHONPATH=/home/vil/Desktop/Object_Tracking/AttackTracker:$PYTHONPATH
#export PYTHONPATH=/media/tsingqguo/data/AttackTracker:$PYTHONPATH
#export CUDA_VISIBLE_DEVICES=$4

python -u ../../tools/attack_vis.py \
       --snapshot model.pth \
       --dataset $2 \
       --video $3 \
       --config config.yaml \
       --interval 30 \
       --apts \
       --att_method MIFGSM \
       --traj_type 2 \
       --att_type TA

python -u ../../tools/attack_vis.py \
       --snapshot model.pth \
       --dataset $2 \
       --video $3 \
       --config config.yaml \
       --interval 30 \
       --apts \
       --att_method BIM \
       --traj_type 2 \
       --att_type TA

python -u ../../tools/attack_vis.py \
       --snapshot model.pth \
       --dataset $2 \
       --video $3 \
       --config config.yaml \
       --interval 30 \
       --apts \
       --att_method OIM \
       --traj_type 2 \
       --att_type TA

#results_paths=''
#python ../../tools/visualize.py 	 \
#	--dataset $2 \
#	--result_path_config visual_results_pathes.txt \
#	--video $3 \
#	--att_type TA
