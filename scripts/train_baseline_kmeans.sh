#!/bin/sh
SOURCE=$1
TARGET=$2
ARCH=$3
CLUSTER=$4

if [ $# -ne 4 ]
  then
    echo "Arguments error: <SOURCE> <TARGET> <ARCH> <CLUSTER NUM>"
    exit 1
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/base_train_kmeans.py -dt ${TARGET} -a ${ARCH} --num-clusters ${CLUSTER} \
	--num-instances 4 --lr 0.00035 --iters 400 -b 64 --epochs 40 --dropout 0 \
	--init logs/${SOURCE}TO${TARGET}/${ARCH}-pretrain-1/model_best.pth.tar \
	--logs-dir logs/${SOURCE}TO${TARGET}/${ARCH}-baseline-${CLUSTER}
