set -xe

CUDA_VISIBLE_DEVICES=0 python train.py \
    -p bdd100k \
    -c 3 \
    -w weights/hybridnets.pth \
    -b 8 \
    --cal_map False \
    --lr 1e-5