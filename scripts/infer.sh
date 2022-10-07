# CUDA_VISIBLE_DEVICES=1 python hybridnets_test.py -p bdd100kcar -w weights/hybridnets.pth --source demo/image --output demo_result --imshow False --imwrite True
CUDA_VISIBLE_DEVICES=1 python hybridnets_test.py -p bdd100k -w output/bdd100k/checkpoints/hybridnets-d3_0_5000.pth --source demo/image --output demo_result --imshow False --imwrite True
