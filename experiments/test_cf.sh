#!/usr/bin/env bash
exp_path=/home/guoqing/projects/AttackTrack/experiments/$1/
export PYTHONPATH=/home/guoqing/projects/AttackTrack:$PYTHONPATH

cd $exp_path

python -u ../../tools/test_us.py \
       --dataset OTB100 \
       --config config.yaml

python -u ../../tools/test_us.py \
       --dataset VOT2018 \
       --config config.yaml