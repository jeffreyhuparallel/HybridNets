set -xe

# CUDA_VISIBLE_DEVICES=1 python val.py -w weights/hybridnets.pth --conf_thres 0.5
CUDA_VISIBLE_DEVICES=1 python val.py -w checkpoints/bdd100k/hybridnets-d3_0_6000.pth --conf_thres 0.5