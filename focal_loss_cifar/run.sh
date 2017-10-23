set -x
set -e
export CUDA_VISIBLE_DEVICES=$1
LOSS_TYPE=$2

DATASET=$3 #cifar-100
DATA_DIR=~/dataset/${DATASET}
TRAIN_DIR=~/models/${DATASET}/${LOSS_TYPE}

if [ ! -d "$TRAIN_DIR" ]; then
    mkdir -p $TRAIN_DIR
fi

python cifar_train.py \
	--loss_type=$LOSS_TYPE\
	--data_dir=$DATA_DIR\
	--dataset=$DATASET\
	--train_dir=$TRAIN_DIR 2>&1 | tee -a  $TRAIN_DIR/log_train.log &

python cifar_eval.py\
	--loss_type=$LOSS_TYPE\
	--data_dir=$DATA_DIR\
	--dataset=$DATASET\
	--checkpoint_dir=$TRAIN_DIR\
	--eval_dir=$TRAIN_DIR/eval
	
