#!/bin/sh
TARGET=$1
ARCH=$2
MODEL=$3

if [ $# -ne 3 ]
  then
    echo "Arguments error: <TARGET> <ARCH> <MODEL PATH>"
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 \
python examples/test_model.py -b 256 -j 8 \
	--dataset-target ${TARGET} -a ${ARCH} --resume ${MODEL}
