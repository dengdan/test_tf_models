# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

import util
import cifar
parser = cifar.parser
parser.add_argument('--train_dir', type=str, default= util.io.get_absolute_path('~/models/cifar/'),
                    help='Directory where to write event logs and checkpoint.')
parser.add_argument('--num_gpus', type=int, default = 1, help = 'number of gpus for trainings')
parser.add_argument('--max_steps', type=int, default=1000000,
                    help='Number of batches to run.')
parser.add_argument('--log_device_placement', type=bool, default=False,
                    help='Whether to log device placement.')

parser.add_argument('--log_frequency', type=int, default=100,
                    help='How often to log results to the console.')

# Constants describing the training process.
LEARNING_RATE_DECAY_FACTOR = 0.1    # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1             # Initial learning rate.
DECAY_STEPS = 200000 # decay rate for lr 


def build_loss(logits, labels):
    """
    OHEM loss
    """
    FLAGS = parser.parse_args()
    
    ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels = labels, logits = logits)
    # find the most wrongly classified examples:
    num_examples = tf.reduce_prod(labels.shape)
    n_selected = tf.cast(num_examples / 2, tf.int32)
    vals, _ = tf.nn.top_k(ce_loss, k = n_selected)
    th = vals[-1]
    selected_mask = ce_loss >= th
    loss_weight = tf.cast(selected_mask, tf.float32) 
    loss = tf.reduce_sum(ce_loss * loss_weight) / tf.reduce_sum(loss_weight)
    tf.add_to_collection('losses', loss)
    return loss


def train():
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Decay the learning rate exponentially based on the number of steps.
    decay_steps = DECAY_STEPS
    lr = INITIAL_LEARNING_RATE
    lr = tf.train.exponential_decay(lr,
                    global_step,
                    decay_steps,
                    LEARNING_RATE_DECAY_FACTOR,
                    staircase=True)
    
    lr = tf.constant(INITIAL_LEARNING_RATE)
    optimizer = tf.train.MomentumOptimizer(lr, momentum = FLAGS.momentum)
#     optimizer = tf.train.GradientDescentOptimizer(lr)

    tf.summary.scalar('learning_rate', lr)
    
    total_loss = 0
    import util
    gradients = []
    reuse = False
    for clone_idx, gpu in enumerate(util.tf.get_available_gpus(num_gpus = FLAGS.num_gpus)):
        with tf.device('/cpu:0'):
            images, labels = cifar.distorted_inputs(is_cifar10 = FLAGS.dataset == 'cifar-10')
        clone_scope = 'clone_%d'%(clone_idx)
        with tf.variable_scope(tf.get_variable_scope(), reuse = reuse):
            with tf.name_scope(clone_scope) as clone_scope:
                reuse = True
                with tf.device(gpu):
                    logits = cifar.inference(images, is_training = True)
                    build_loss(logits, labels)
                    losses = tf.get_collection(tf.GraphKeys.LOSSES, clone_scope)
                    total_clone_loss = tf.add_n(losses) / FLAGS.num_gpus
                    total_loss += total_clone_loss
                    # gather regularization total_loss and add to clone_0 only
                    if clone_idx == 0:
                        regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                        total_clone_loss = total_clone_loss + regularization_loss
                        total_loss += regularization_loss
                        # compute clone gradients
                    clone_gradients = optimizer.compute_gradients(total_clone_loss)
                    gradients.append(clone_gradients)

    grads = util.tf.sum_gradients(gradients)        
    
    train_op = []
    batch_norm_update_op = util.tf.get_update_op()
    if batch_norm_update_op is not None:
        train_op.append(batch_norm_update_op)
        
    # Apply gradients.
    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
    train_op.append(apply_gradient_op)
    
    # Track the moving averages of all trainable variables.
    if FLAGS.using_moving_average:
        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())
        train_op.append(variable_averages_op)
        
        with tf.control_dependencies([apply_gradient_op]):# ema after updating
                train_op.append(tf.group(variable_averages_op))
            
    train_op = control_flow_ops.with_dependencies(train_op, total_loss, name='train_op')
    
    class _LoggerHook(tf.train.SessionRunHook):
      """Logs total_loss and runtime."""

      def begin(self):
        self._start_time = time.time()

      def before_run(self, run_context):
        return tf.train.SessionRunArgs([total_loss, global_step])  # Asks for total_loss value.

      def after_run(self, run_context, run_values):
        loss_value, step = run_values.results
        if step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, total_loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), step, loss_value,
                               examples_per_sec, sec_per_batch))
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement, allow_soft_placement = True)

    util.tf.gpu_config(config = config, allow_growth = True)
    
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        save_checkpoint_secs = 30,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(total_loss),
               _LoggerHook()], config=config) as mon_sess:
        while not mon_sess.should_stop():
            mon_sess.run(train_op)

def main(argv=None):  # pylint: disable=unused-argument
    cifar.maybe_download_and_extract()
    train()


if __name__ == '__main__':
  FLAGS = parser.parse_args()
  util.proc.set_proc_name('train_on_%s'%(FLAGS.dataset))
  tf.app.run()
