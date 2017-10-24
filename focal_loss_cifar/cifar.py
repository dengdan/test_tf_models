# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
import sys
import tarfile
import tensorflow as tf
from six.moves import urllib
import tensorflow as tf
import util
import cifar_input
slim = tf.contrib.slim
parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--batch_size', type=int, default=128,
                                        help='Number of images to process in a batch.')
parser.add_argument('--weight_decay', type = float, default = 0.0001)
parser.add_argument('--data_dir', type=str,
                                        help='Path to the CIFAR data directory.')
parser.add_argument('--dataset', type=str,
                                        help='cifar10 or 100.')

# Global constants describing the CIFAR data set.
IMAGE_SIZE = cifar_input.IMAGE_SIZE
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999         # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0            # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1    # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1             # Initial learning rate.
# 0.01, 0.00001 for ce_loss of cifar-10 and 100


def get_data_url():
        FLAGS = parser.parse_args()
        if FLAGS.dataset == 'cifar-10':
                return 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
        elif FLAGS.dataset == 'cifar-100':
                return 'https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz'
        else:
                raise ValueError('Unknow dataset:', FLAGS.dataset)

def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
        x: Tensor
    Returns:
        nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                                                                             tf.nn.zero_fraction(x))



def distorted_inputs(is_cifar10):
    """Construct distorted input for CIFAR training using the Reader ops.

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.

    Raises:
        ValueError: If no data_dir
    """
    FLAGS = parser.parse_args()
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, '%s-binary'%(FLAGS.dataset))
    images, labels = cifar_input.distorted_inputs(data_dir=data_dir,
                                                                                                    batch_size=FLAGS.batch_size, is_cifar10 = is_cifar10)
    return images, labels


def inputs(eval_data, is_cifar10):
    """Construct input for CIFAR evaluation using the Reader ops.

    Args:
        eval_data: bool, indicating if one should use the train or eval data set.

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.

    Raises:
        ValueError: If no data_dir
    """
    FLAGS = parser.parse_args()
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, '%s-binary'%(FLAGS.dataset))
    images, labels = cifar_input.inputs(eval_data=eval_data,
                                                                                data_dir=data_dir,
                                                                                batch_size=FLAGS.batch_size,
                                                                                is_cifar10 = is_cifar10)
    return images, labels


def inference(images):
    """Build the CIFAR-10 model.

    Args:
        images: Images returned from distorted_inputs() or inputs().

    Returns:
        Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    FLAGS = parser.parse_args()
    # conv1
    with slim.arg_scope([slim.conv2d], weights_regularizer = slim.l2_regularizer(FLAGS.weight_decay)):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME'):
                # Block1
                net = slim.conv2d(images, 16, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')

                # Block 2.
                net = slim.conv2d(net, 32, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')

                # Block 3.
                net = slim.conv2d(net, 64, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')

                # Block 4.
                net = slim.conv2d(net, 128, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                
        weight_init, bias_init = util.tf.focal_loss_layer_initializer()
        logits = slim.conv2d(net, 1, [2, 2], scope = 'score', activation_fn = None,
                                                weights_initializer = weight_init, 
                                                biases_initializer = bias_init,
                                                padding = 'VALID'
        )
#         if FLAGS.loss_type == 'focal_loss':
#         else:
#                 logits = slim.conv2d(net, 1, [2, 2], scope = 'score', activation_fn = None,
#                                                            padding = 'VALID'
#         )
    return logits


def focal_loss(labels, logits, gamma, alpha, normalize = True):
    labels = tf.where(labels > 0, tf.ones_like(labels), tf.zeros_like(labels))
    labels = tf.cast(labels, tf.float32)
    probs = tf.sigmoid(logits)
    CE = tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits = logits)

    alpha_t = tf.ones_like(logits) * alpha
    alpha_t = tf.where(labels > 0, alpha_t, 1.0 - alpha_t)
    probs_t = tf.where(labels > 0, probs, 1.0 - probs)

    focal_matrix = alpha_t * tf.pow((1.0 - probs_t), gamma)
    fl = focal_matrix * CE

    fl = tf.reduce_sum(fl)
    if normalize:
        #n_pos = tf.reduce_sum(labels)
        #fl = fl / tf.cast(n_pos, tf.float32)
        total_weights = tf.stop_gradient(tf.reduce_sum(focal_matrix))
        fl = fl / total_weights
    return fl

def loss(logits, labels):
    FLAGS = parser.parse_args()
    logits = logits[:, 0, 0, 0]
    labels = tf.equal(labels, 1)
    labels = tf.cast(labels, tf.float32)
    
    if FLAGS.loss_type == 'focal_loss':
            loss = focal_loss(labels = labels, logits = logits, 
                                      gamma = FLAGS.focal_loss_gamma, alpha = FLAGS.focal_loss_alpha)
    elif FLAGS.loss_type == 'ce_loss':
            ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels = labels, logits = logits)
            loss = tf.reduce_sum(ce_loss) / tf.cast(tf.reduce_prod(tf.shape(labels)), tf.float32)
    elif FLAGS.loss_type == 'cls_balance':
            ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels = labels, logits = logits)
            pos_weight = tf.cast(tf.equal(labels, 1), tf.float32)
            neg_weight = 1 - pos_weight
            
            n_pos = tf.reduce_sum(pos_weight)
            n_neg = tf.reduce_sum(neg_weight)
            
            def has_pos():
                    return tf.reduce_sum(ce_loss * pos_weight) / n_pos
            def has_neg(): 
                    return tf.reduce_sum(ce_loss * neg_weight) / n_neg
            def no():
                    return tf.constant(0.0)
            pos_loss = tf.cond(n_pos > 0, has_pos, no)
            neg_loss = tf.cond(n_neg > 0, has_neg, no)
            loss = (pos_loss + neg_loss) / 2.0
            
    elif FLAGS.loss_type == 'ohem':
            ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels = labels, logits = logits)
            pos_weight = tf.cast(tf.equal(labels, 1), tf.float32)
            neg_weight = 1 - pos_weight
            n_pos = tf.reduce_sum(pos_weight)
            n_neg = tf.reduce_sum(neg_weight)
            
            scores = util.tf.sigmoid(logits)
            neg_mask = tf.equal(labels, 0)
            neg_scores = tf.where(neg_mask, scores, tf.zeros_like(scores))
            
            # find the most wrongly classified negative examples:
            n_selected = tf.minimum(n_pos * 3, n_neg)
            n_selected = tf.cast(tf.maximum(n_selected, 1), tf.int32)
            vals, _ = tf.nn.top_k(neg_scores, k = n_selected)
            th = vals[-1]
            selected_neg_mask = tf.logical_and(scores >= th, neg_mask)
            neg_weight = tf.cast(selected_neg_mask, tf.float32)
            
            loss_weight = pos_weight + neg_weight
            loss = tf.reduce_sum(ce_loss * loss_weight) / tf.reduce_sum(loss_weight)
    elif FLAGS.loss_type == 'ohem_all':
            ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels = labels, logits = logits)
            scores = util.tf.sigmoid(logits)
            hardness = tf.where(labels > 0, 1 - scores, scores)
            # find the most wrongly classified examples:
            num_examples = tf.reduce_prod(labels.shape)
            n_selected = tf.cast(num_examples / 2, tf.int32)
            vals, _ = tf.nn.top_k(hardness, k = n_selected)
            th = vals[-1]
            selected_mask = hardness >= th
            loss_weight = tf.cast(selected_mask, tf.float32) 
            loss = tf.reduce_sum(ce_loss * loss_weight) / tf.reduce_sum(loss_weight)
    else:
            raise ValueError('Unknow loss_type:', FLAGS.loss_type)
            
                
                    
    tf.add_to_collection('losses', loss)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
        total_loss: Total loss from loss().
    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    """Train CIFAR model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps
            processed.
    Returns:
        train_op: op for training.
    """
    # Variables that affect learning rate.
    FLAGS = parser.parse_args()
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = 200000

    # Decay the learning rate exponentially based on the number of steps.
    lr = INITIAL_LEARNING_RATE
    lr = tf.train.exponential_decay(lr,
                                                            global_step,
                                                            decay_steps,
                                                            LEARNING_RATE_DECAY_FACTOR,
                                                            staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    FLAGS = parser.parse_args()
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
                os.makedirs(dest_directory)
    DATA_URL = get_data_url()    
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                        float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, '%s-binary'%(FLAGS.dataset))
    if not os.path.exists(extracted_dir_path):
                if not util.str.contains(extracted_dir_path, 'cifar-100'):
                        tarfile.open(filepath, 'r:gz').extractall(extracted_dir_path)
                        cmd ='mv {0}/{1}/* {0}/;rm -rf {0}/{1}'.format(extracted_dir_path, 'cifar-10-batches-bin')
                        print(cmd)
                        print(util.cmd.cmd(cmd))
                else:
                        tarfile.open(filepath, 'r:gz').extractall(dest_directory)