set -xe

CUDA_VISIBLE_DEVICES=0 python train.py \
    -p bdd100kcar \
    -c 3 \
    -b 8 \
    --lr 1e-4