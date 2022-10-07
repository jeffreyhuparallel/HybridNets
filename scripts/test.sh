set -xe

CUDA_VISIBLE_DEVICES=1 python test.py -p bdd100kcar -w weights/hybridnets.pth --conf_thres 0.5
