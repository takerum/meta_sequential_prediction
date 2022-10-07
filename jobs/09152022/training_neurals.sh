#!/bin/bash
LOGDIR_ROOT='/mnt/research_logs/logs/09152022'
DATASET_ROOT='/home/TakeruMiyato/datasets'

for seed in 1 2 3; do
    for dataset_name in 3dshapes smallNORB; do
        for model_name in neuralM neural_trans; do
            python run.py --log_dir=${LOGDIR_ROOT}/${dataset_name}-${model_name}-seed${seed}/ \
            --config_path=./configs/${dataset_name}/lstsq/${model_name}.yml \
            --attr seed=${seed}  train_data.args.root=${DATASET_ROOT}
        done
    done 
done