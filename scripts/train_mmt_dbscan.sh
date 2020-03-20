#!/bin/sh
SOURCE=$1
TARGET=$2
ARCH=$3

if [ $# -ne 3 ]
  then
    echo "Arguments error: <SOURCE> <TARGET> <ARCH>"
    exit 1
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/mmt_train_dbscan.py -ds ${SOURCE} -dt ${TARGET} -a ${ARCH} \
	--num-instances 4 --lr 0.00035 --iters 400 -b 64 --epochs 40 \
	--soft-ce-weight 0.5 --soft-tri-weight 0.8 --dropout 0 --lambda-value 0 \
	--init-1 logs/${SOURCE}TO${TARGET}/${ARCH}-pretrain-1/model_best.pth.tar \
	--init-2 logs/${SOURCE}TO${TARGET}/${ARCH}-pretrain-2/model_best.pth.tar \
	--logs-dir logs/${SOURCE}TO${TARGET}/${ARCH}-MMT-DBSCAN
	# --rr-gpu
