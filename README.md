# Meta-sequential prediction (MSP)
<p align="center">
<img width="650" alt="image" src="https://user-images.githubusercontent.com/11573649/195465919-53ad444e-c2dc-40c8-a4d5-99e7b072f5b2.png">
</p>



This repository contains the code for the NeurIPS2022 paper: Unsupervised Learning of Equivariant Structure on Sequences.
A simple encoder-decoder model trained with *meta-sequential prediction* captures the hidden disentangled structure underlying the datasets.

[[OpenReview]](https://openreview.net/forum?id=7b7iGkuVqlZ)

## The implementation of MSP and simultaneous block diagonalization (SBD)
- `SeqAELSTSQ` in `./models/seqae.py` is the implementation of *meta-sequential prediction*.
- `tracenorm_of_normalized_laplacian` in `./utils/laplacian.py` is used to calculate the block diagonalization loss in our paper.

# Setup
## Prerequisite
python3.7, CUDA11.2, cuDNN

Please install additional python libraries by:
```
pip install -r requirements.txt
```

## Download datasets 
Download the compressed dataset files from the following link:

https://drive.google.com/drive/folders/1_uXjx06U48to9OSyGY1ezqipAbbuT0vq?usp=sharing

This is an example script to download and decompress the files:
```
# If gdown is not installed:
pip install gdown

export DATADIR_ROOT=/tmp/path/to/datadir/
cd /tmp
gdown --folder https://drive.google.com/drive/folders/1_uXjx06U48to9OSyGY1ezqipAbbuT0vq?usp=sharing 
mv datasets/* ${DATADIR_ROOT}; rm datasets -r 
cd -
tar xzf  ${DATADIR_ROOT}/MNIST.tar.gz -C $DATADIR_ROOT
tar xzf  ${DATADIR_ROOT}/3dshapes.tar.gz -C $DATADIR_ROOT
tar xzf  ${DATADIR_ROOT}/smallNORB.tar.gz -C $DATADIR_ROOT
```

## Training with MSP
1. Select a config file for the dataset on which you want to train the model:
```
# Sequential MNIST
export CONFIG=configs/mnist/lstsq/lstsq.yml
# Sequential MNIST-bg with digit 4
export CONFIG=configs/mnist_bg/lstsq/lstsq.yml
# 3DShapes
export CONFIG=configs/3dshapes/lstsq/lstsq.yml
# SmallNORB
export CONFIG=configs/smallNORB/lstsq/lstsq.yml
# Accelerated sequential MNIST
export CONFIG=configs/mnist_accl/lstsq/holstsq.yml
```

2. Run:
```
export LOGDIR=/tmp/path/to/logdir
export DATADIR_ROOT=/tmp/path/to/datadir
python run.py --config_path=$CONFIG --log_dir=$LOGDIR --attr train_data.args.root=$DATADIR_ROOT
```

## Training all the methods we tested in our NeurIPS paper
```
export LOGDIR=/tmp/path/to/logdir
export DATADIR_ROOT=/tmp/path/to/datadir
bash training_allmodels.sh $LOGDIR $DATADIR_ROOT
```

## Evaluations
- Generated images: `gen_images.ipynb`
- Equivariance errors: `equivariance_error.ipynb`
- Prediction errors: `extrp.ipynb`
- Simultaneous block diagoanlization: `block_diagonalization.ipynb`
