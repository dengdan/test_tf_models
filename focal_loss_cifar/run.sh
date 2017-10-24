set -x
set -e
export CUDA_VISIBLE_DEVICES=$1
LOSS_TYPE=focal_loss
DATASET=$2 #cifar-100
FOCAL_LOSS_ALPHA=$3


DATA_DIR=~/dataset/${DATASET}
TRAIN_DIR=~/models/${DATASET}/${LOSS_TYPE}_alpha_${FOCAL_LOSS_ALPHA}

if [ ! -d "$TRAIN_DIR" ]; then
    mkdir -p $TRAIN_DIR
fi

python cifar_train.py \
	--loss_type=$LOSS_TYPE\
	--data_dir=$DATA_DIR\
	--dataset=$DATASET\
	--focal_loss_alpha=${FOCAL_LOSS_ALPHA}\
	--train_dir=$TRAIN_DIR 2>&1 | tee -a  $TRAIN_DIR/log_train.log &

python cifar_eval.py\
	--loss_type=$LOSS_TYPE\
	--data_dir=$DATA_DIR\
	--dataset=$DATASET\
	--checkpoint_dir=$TRAIN_DIR\
	--eval_dir=$TRAIN_DIR/eval
	
