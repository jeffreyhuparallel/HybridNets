set -xe

CUDA_VISIBLE_DEVICES=1 python train.py \
    -p bdd100kcar_finetune \
    -c 3 \
    -w weights/hybridnets.pth \
    -b 4 \
    --lr 1e-5