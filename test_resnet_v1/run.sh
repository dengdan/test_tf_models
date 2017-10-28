set -x
set -e
export CUDA_VISIBLE_DEVICES=$1
BATCH_SIZE=$2
ACT=$3
DATASET=cifar-10

DATA_DIR=~/dataset/${DATASET}
TRAIN_DIR=~/temp/models/${DATASET}/resnet_batch_size_${BATCH_SIZE}_no_ema

if [ ! -d "$TRAIN_DIR" ]; then
    mkdir -p $TRAIN_DIR
fi


	
if [ $ACT == 'eval' ]
then
	python cifar_eval.py\
		--batch_size=${BATCH_SIZE}\
		--data_dir=$DATA_DIR\
		--dataset=$DATASET\
		--checkpoint_dir=$TRAIN_DIR\
		--eval_dir=$TRAIN_DIR/eval #2>&1 | tee -a  $TRAIN_DIR/log_eval.log
else
	python cifar_train.py \
		--batch_size=${BATCH_SIZE}\
		--data_dir=$DATA_DIR\
		--dataset=$DATASET\
		--train_dir=$TRAIN_DIR #2>&1 | tee -a  $TRAIN_DIR/log_train.log 
fi
	