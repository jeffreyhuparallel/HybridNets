from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.VERSION = 1


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DETECTION_ON = False
_C.MODEL.SEGMENTATION_ON = False
_C.MODEL.TRACK_ON = False
_C.MODEL.CLASSIFICATION_ON = False
_C.MODEL.META_ARCHITECTURE = "hydranet"

_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = "yolov4"
_C.MODEL.BACKBONE.COMPOUND_COEF = 3
_C.MODEL.BACKBONE.FREEZE = False

_C.MODEL.DETECTION_HEAD = CN()
_C.MODEL.DETECTION_HEAD.FREEZE = False
_C.MODEL.DETECTION_HEAD.NUM_CLASSES = 80
_C.MODEL.DETECTION_HEAD.DETECTIONS_PER_IMAGE = 100
_C.MODEL.DETECTION_HEAD.ANCHORS_SCALES = [1, 1.52, 2.14]
_C.MODEL.DETECTION_HEAD.ANCHORS_RATIOS = [(1.0, 1.0), (0.57, 1.82), (1.82, 0.57)]

_C.MODEL.SEGMENTATION_HEAD = CN()
_C.MODEL.SEGMENTATION_HEAD.FREEZE = False
_C.MODEL.SEGMENTATION_HEAD.NUM_CLASSES = 19

_C.MODEL.TRACK_HEAD = CN()
_C.MODEL.TRACK_HEAD.FREEZE = False
_C.MODEL.TRACK_HEAD.LENGTH = 100  # Length of predicted track in meters

_C.MODEL.CLASSIFICATION_HEAD = CN()
_C.MODEL.CLASSIFICATION_HEAD.NUM_CLASSES = 7

# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------
_C.LOSSES = CN()

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.SIZE = (608, 416)
_C.INPUT.HORIZONTAL_FLIP_PROB = 0.0
_C.INPUT.VERTICAL_FLIP_PROB = 0.0
_C.INPUT.RANDOM_CROP_MIN_SCALE = 1.0
_C.INPUT.RANDOM_CROP_MAX_SCALE = 1.0
_C.INPUT.RANDOM_ROTATION_MAX_THETA = 0.0

# ColorJitter
_C.INPUT.BRIGHTNESS = 0.1
_C.INPUT.CONTRAST = 0.1
_C.INPUT.SATURATION = 0.1
_C.INPUT.HUE = 0.1


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training. Must be registered
# Samples from these datasets will be merged and used as one dataset.
_C.DATASETS.TRAIN = ("rail10k",)
# List of the dataset names for validation. Must be registered
_C.DATASETS.VAL = ("rail10k",)
# List of the dataset names for testing. Must be registered
_C.DATASETS.TEST = ("rail10k",)
# List of the dataset names for prediction. Must be registered
_C.DATASETS.PREDICT = ("union_pacific",)


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.BATCH_SIZE = 8


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.BASE_LR = 0.001


# ---------------------------------------------------------------------------- #
# Reconstruction
# ---------------------------------------------------------------------------- #
_C.RECONSTRUCTION = CN()
_C.RECONSTRUCTION.SPARSE_BACKEND = "colmap"
_C.RECONSTRUCTION.DENSE_BACKEND = "nerf"
_C.RECONSTRUCTION.SIZE = (1620, 1080)

# ---------------------------------------------------------------------------- #
# Export the PyTorch model to ONNX and TensorRT
# ---------------------------------------------------------------------------- #
_C.EXPORT = CN()
# Batch size of the ONNX model. The generated TensorRT engine should also run
# inference with this batch size. Currently we have 2 sources (2 cameras). So the
# batch size is set to 2.
_C.EXPORT.ONNX_BATCH_SIZE = 2
# This is the version of the onnx opset. (https://github.com/onnx/onnx/blob/main/docs/Operators.md)
_C.EXPORT.ONNX_OPSET = 11
# This is the property for the EfficientNMS_TRT Plugin. The maximum number boxes
# after Non-maxmimum Suppression. (https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin)
_C.EXPORT.NMS_MAX_OUTPUT_BOXES = 50
# This is the precision of the TensorRT engine (currently only support fp16 and fp32)
_C.EXPORT.PRECISION = "fp16"

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Directory where output files are written
_C.OUTPUT_DIR = "./output"
# Set seed to negative to fully randomize everything.
# Set seed to positive to use a fixed seed. Note that a fixed seed increases
# reproducibility but does not guarantee fully deterministic behavior.
# Disabling all parallelism further increases reproducibility.
_C.SEED = -1

# Maximum epochs to train
_C.MAX_EPOCHS = 180
# Save top k model checkpoints
_C.SAVE_TOP_K = 3