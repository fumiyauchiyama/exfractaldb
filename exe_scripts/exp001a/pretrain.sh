#!/bin/bash

#$-l rt_F=4
#$-l h_rt=24:00:00
#$-l USE_SSH=1
#$-j y
#$-o output/exp001a/
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

# hyperparameters
# model size
MODEL=tiny
# initial learning rate
LR=1.0e-3
# name of dataset
DATA_NAME=ExFractalDB
# num of classes
CLASSES=1000
# num of epochs
EPOCHS=2
# path to train dataset
SOURCE_DATASET=/groups/gag51404/user/fumiyau/fdsl_language/libraries/exfractaldb/dataset/MVFractalDB-1000/images
# output dir path
OUT_DIR=./output/pretrain/exp001a
# num of GPUs
NGPUS=$NUM_GPUS
# num of processes per node
NPERNODE=$NUM_GPU_PER_NODE
# local mini-batch size (global mini-batch size = NGPUS × LOCAL_BS)
LOCAL_BS=64

# execution
mpirun -npernode $NPERNODE -np $NGPUS \
    -hostfile $HOSTFILE_NAME \
    -x MASTER_ADDR=$MASTER_ADDR \
    -x MASTER_PORT=$MASTER_PORT \
    python3 pretrain.py ${SOURCE_DATASET} \
    --model deit_${MODEL}_patch16_224 --experiment pretrain_deit_${MODEL}_${DATA_NAME}${CLASSES}_${LR} \
    --input-size 3 224 224 \
    --sched cosine_iter --epochs ${EPOCHS} --lr ${LR} --weight-decay 0.05 \
    --batch-size ${LOCAL_BS} --opt adamw --num-classes ${CLASSES} \
    --warmup-epochs 5 --cooldown-epochs 0 \
    --smoothing 0.1 --drop-path 0.1 --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --remode pixel --interpolation bicubic --hflip 0.0 \
    -j 16 --eval-metric loss \
    --interval-saved-epochs 10 --output ${OUT_DIR} \
    --log-wandb
