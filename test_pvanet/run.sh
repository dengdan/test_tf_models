set -x
set -e
export CUDA_VISIBLE_DEVICES=$1
BATCH_SIZE=$2
DATASET=cifar-100


DATA_DIR=~/dataset/${DATASET}
TRAIN_DIR=~/temp/models/${DATASET}/pvanet_no_bn

if [ ! -d "$TRAIN_DIR" ]; then
    mkdir -p $TRAIN_DIR
fi

python cifar_eval.py\
	--batch_size=${BATCH_SIZE}\
	--data_dir=$DATA_DIR\
	--dataset=$DATASET\
	--checkpoint_dir=$TRAIN_DIR\
	--eval_dir=$TRAIN_DIR/eval

python cifar_train.py \
	--batch_size=${BATCH_SIZE}\
	--data_dir=$DATA_DIR\
	--dataset=$DATASET\
	--train_dir=$TRAIN_DIR #2>&1 | tee -a  $TRAIN_DIR/log_train.log 

	
