set -xe

CUDA_VISIBLE_DEVICES=1 python train.py \
    -p bdd100k \
    -c 3 \
    -b 4