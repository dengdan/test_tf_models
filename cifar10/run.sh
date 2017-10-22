set -x
set -e
export CUDA_VISIBLE_DEVICES=$1
LOSS_TYPE=$2

DATA_DIR=~/dataset/cifar10
TRAIN_DIR=~/models/cifar10/${LOSS_TYPE}
python cifar10_train.py \
	--loss_type=$LOSS_TYPE\
	--train_dir=$TRAIN_DIR &

python cifar10_eval.py\
	--checkpoint_path=$TRAIN_DIR\
	--eval_path=$TRAIN_DIR/eval
	
