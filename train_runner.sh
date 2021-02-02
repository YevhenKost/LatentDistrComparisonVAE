docker run --gpus all --rm -it -v ~/tmp:/tmp trainer python3 /src/train_grid.py -dist_type $1 -device "cuda:$2"
