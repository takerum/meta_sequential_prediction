#!/bin/bash
LOGDIR_ROOT='/mnt/research_logs/logs/09232022'
DATASET_ROOT='/home/TakeruMiyato/datasets'

for seed in 1 2 3; do
    for dataset_name in mnist_bg; do
        for model_name in neuralM_latentpred; do
            for loss_latent_coeff in 0.0001 0.001 0.01; do
                for loss_reconst_coeff in 0.1 1.0 10; do
                    python run.py --log_dir=${LOGDIR_ROOT}/${dataset_name}-${model_name}-lrc${loss_reconst_coeff}-llc${loss_latent_coeff}-seed${seed}/ \
                    --config_path=./configs/${dataset_name}/lstsq/${model_name}.yml \
                    --attr seed=${seed}  train_data.args.root=${DATASET_ROOT} \
                    model.args.loss_latent_coeff=${loss_latent_coeff} model.args.loss_reconst_coeff=${loss_reconst_coeff}
                done
            done
        done
    done 
done