#!/bin/sh
SOURCE=$1
TARGET=$2
ARCH=$3
SEED=$4

if [ $# -ne 4 ]
  then
    echo "Arguments error: <SOURCE> <TARGET> <ARCH> <SEED>"
    exit 1
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/source_pretrain.py -ds ${SOURCE} -dt ${TARGET} -a ${ARCH} --seed ${SEED} --margin 0.0 \
	--num-instances 4 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 100 --epochs 80 --eval-step 40 \
	--logs-dir logs/${SOURCE}TO${TARGET}/${ARCH}-pretrain-${SEED}
