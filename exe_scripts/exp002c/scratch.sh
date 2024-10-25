#!/bin/bash

#$-l rt_F=16
#$-l h_rt=96:00:00
#$-l USE_SSH=1
#$-j y
#$-o output/exp002c/
#$-cwd

source /etc/profile.d/modules.sh
module load python/3.10/3.10.14
module load cuda/12.1/12.1.1
module load cudnn/9.0/9.0.0
module load nccl/2.17/2.17.1-1
module load hpcx/2.12
source .venv/bin/activate

# environment variable which is the IP address of the machine in rank 0 (need only for multiple nodes)
# for ABCI, default ssh port is 2222
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))
echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
if [[ "$SGE_RESOURCE_TYPE" == "rt_F" ]]; then
  export NUM_GPU_PER_NODE=4
  NODE_TYPE="v100"
elif [[ "$SGE_RESOURCE_TYPE" == "rt_AF" ]]; then
  export NUM_GPU_PER_NODE=8
  NODE_TYPE="a100"
else
  echo "Unrecognized SGE_RESOURCE_TYPE: $SGE_RESOURCE_TYPE"
fi
NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))
mkdir -p ./hostfile
HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r line; do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done <"$SGE_JOB_HOSTLIST" >"$HOSTFILE_NAME"


# ======== parameter for pre-trained model ========
# exp name
EXP_NAME=exp002c
# model size
MODEL=base
# initial learning rate for pre-train
# PRE_LR=$LR
# name of dataset for pre-train
# PRE_DATA_NAME=$DATA_NAME
# num of classes for pre-train
# PRE_CLASSES=$CLASSES
# path to checkpoint of pre-trained model
# CP_PATH=./output/pretrain/${EXP_NAME}/pretrain_deit_${MODEL}_${PRE_DATA_NAME}${PRE_CLASSES}_${PRE_LR}/model_best.pth.tar

# ======== parameter for fine-tuning ========
# output dir path
OUT_DIR=./output/finetune/${EXP_NAME}
# path to fine-tune dataset
SOURCE_DATASET_DIR=/groups/gag51404/dataset/ImageNet1k/.cache_timm
# name of dataset
DATA_NAME=ImageNet1k
# initial learning rate
LR=1.0e-3
# num of classes
CLASSES=1000
# num of epochs
EPOCHS=180
# num of GPUs
NGPUS=$NUM_GPUS
# num of processes per node
NPERNODE=$NUM_GPU_PER_NODE
# local mini-batch size (global mini-batch size = NGPUS × LOCAL_BS)
LOCAL_BS=64

# environment variable which is the IP address of the machine in rank 0 (need only for multiple nodes)
# MASTER_ADDR="192.168.1.1"

mpirun -npernode $NPERNODE -np $NGPUS \
    -hostfile $HOSTFILE_NAME \
    -x MASTER_ADDR=$MASTER_ADDR \
    -x MASTER_PORT=$MASTER_PORT \
    python3 finetune.py ${SOURCE_DATASET_DIR} \
    --dataset hfds/ILSVRC/imagenet-1k \
    --model deit_${MODEL}_patch16_224 --experiment scratch_deit_${MODEL}_${DATA_NAME}${CLASSES}_${LR} \
    --input-size 3 224 224 --num-classes ${CLASSES} \
    --sched cosine_iter --epochs ${EPOCHS} --lr ${LR} --weight-decay 0.05 \
    --batch-size ${LOCAL_BS} --opt adamw \
    --warmup-epochs 5 --cooldown-epochs 0 \
    --smoothing 0.1 --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug --mixup 0.8 --cutmix 1.0 \
    --drop-path 0.1 --reprob 0.25 -j 16 \
    --output ${OUT_DIR} \
    --log-wandb
    
# --pretrained-path ${CP_PATH}