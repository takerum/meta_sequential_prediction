#!/bin/bash
LOGDIR_ROOT=$1
DATADIR_ROOT=$2

# Training on MNIST, MNIST-bg w/ digit 4, 3DShapes and smallNORB
for seed in 1 2 3; do
    for dataset_name in mnist mnist_bg 3dshapes smallNORB; do
        for model_name in lstsq lstsq_multi lstsq_rec neuralM neural_trans; do
            python run.py --log_dir=${LOGDIR_ROOT}/${dataset_name}-${model_name}-seed${seed}/ \
            --config_path=./configs/${dataset_name}/lstsq/${model_name}.yml \
            --attr seed=${seed}  train_data.args.root=${DATADIR_ROOT}
        done
    done 
done

# Training on MNIST-bg w/ all digits
for seed in 1 2 3; do
    for dataset_name in mnist_bg; do
        for model_name in lstsq lstsq_multi lstsq_rec; do
            python run.py --log_dir=${LOGDIR_ROOT}/${dataset_name}_full-${model_name}-seed${seed}/ \
            --config_path=./configs/${dataset_name}/lstsq/${model_name}.yml \
            --attr seed=${seed} train_data.args.root=${DATADIR_ROOT} train_data.args.only_use_digit4=False max_iteration=200000 training_loop.args.lr_decay_iter=160000
        done
        for model_name in neuralM neural_trans; do
            python run.py --log_dir=${LOGDIR_ROOT}/${dataset_name}_full-${model_name}-seed${seed}/ \
            --config_path=./configs/${dataset_name}/lstsq/${model_name}.yml \
            --attr seed=${seed}  train_data.args.root=${DATADIR_ROOT} train_data.args.only_use_digit4=False max_iteration=200000 training_loop.args.lr_decay_iter=160000 training_loop.args.reconst_iter=200000
        done
    done 
done

# Training on Accelerated Sequential MNIST
for seed in 1 2 3; do
    for dataset_name in mnist_accl; do
        for model_name in lstsq holstsq neural_trans; do
            python run.py --log_dir=${LOGDIR_ROOT}/${dataset_name}-${model_name}-seed${seed}/ \
            --config_path=./configs/${dataset_name}/lstsq/${model_name}.yml \
            --attr seed=${seed}  train_data.args.root=${DATADIR_ROOT}
        done
    done 
done

