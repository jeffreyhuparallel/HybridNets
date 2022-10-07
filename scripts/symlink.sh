
MOUNT_DIR="$HOME/mount/nas"
SCRATCH_DIR="$MOUNT_DIR/scratch/$USER/"

ln -snf $SCRATCH_DIR/hybridnet_symlinks/datasets datasets
ln -snf $SCRATCH_DIR/hybridnet_symlinks/output output
