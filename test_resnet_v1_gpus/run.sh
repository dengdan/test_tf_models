set -x
set -e
export CUDA_VISIBLE_DEVICES=$1
BATCH_SIZE=$2
ACT=$3
DATASET=cifar-10
BATCH_NORM=0

# get the number of gpus
OLD_IFS="$IFS" 
IFS="," 
gpus=($CUDA_VISIBLE_DEVICES) 
IFS="$OLD_IFS"
NUM_GPUS=${#gpus[@]}


DATA_DIR=~/dataset/${DATASET}
TRAIN_DIR=~/temp/models/${DATASET}/resnet_batch_size_${BATCH_SIZE}_gpus_${NUM_GPUS}_cnn

if [ ! -d "$TRAIN_DIR" ]; then
    mkdir -p $TRAIN_DIR
fi


	
if [ $ACT == 'eval' ]
then
	python cifar_eval.py\
		--batch_size=100\
		--data_dir=$DATA_DIR\
		--dataset=$DATASET\
		--apply_batch_norm=${BATCH_NORM}\
		--checkpoint_dir=$TRAIN_DIR\
		--eval_dir=$TRAIN_DIR/eval #2>&1 | tee -a  $TRAIN_DIR/log_eval.log
else
	python cifar_train.py \
		--batch_size=${BATCH_SIZE}\
		--num_gpus=$NUM_GPUS\
		--apply_batch_norm=${BATCH_NORM}\
		--data_dir=$DATA_DIR\
		--dataset=$DATASET\
		--train_dir=$TRAIN_DIR #2>&1 | tee -a  $TRAIN_DIR/log_train.log 
fi
	
