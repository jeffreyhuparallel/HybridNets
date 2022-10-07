# CUDA_VISIBLE_DEVICES=1 python hybridnets_test.py -p bdd100k_single -w weights/hybridnets.pth --source demo/image --output demo_result --imshow False --imwrite True
CUDA_VISIBLE_DEVICES=1 python hybridnets_test.py -p bdd100k -w checkpoints/bdd100k/hybridnets-d3_0_4500.pth --source demo/image --output demo_result --imshow False --imwrite True
