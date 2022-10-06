set -xe

CUDA_VISIBLE_DEVICES=0 python train.py \
    -p bdd100k_single \
    -c 3 \
    -w weights/hybridnets.pth \
    -b 8 \
    --lr 1e-5