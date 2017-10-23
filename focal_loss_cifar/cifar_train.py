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

import util
import cifar
parser = cifar.parser
parser.add_argument('--train_dir', type=str, default= util.io.get_absolute_path('~/models/cifar/origin_code'),
                    help='Directory where to write event logs and checkpoint.')
parser.add_argument('--loss_type', type=str, default = 'focal_loss', help = 'loss type')
parser.add_argument('--max_steps', type=int, default=1000000,
                    help='Number of batches to run.')
parser.add_argument('--log_device_placement', type=bool, default=False,
                    help='Whether to log device placement.')

parser.add_argument('--log_frequency', type=int, default=100,
                    help='How often to log results to the console.')


def train():
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()
    
    # Get images and labels for CIFAR.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      images, labels = cifar.distorted_inputs(is_cifar10 = FLAGS.dataset == 'cifar-10')

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar.inference(images)

    # Calculate loss.
    loss = cifar.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar.train(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    util.tf.gpu_config(config = config, allow_growth = True)
    
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        save_checkpoint_secs = 30,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()], config=config) as mon_sess:
        while not mon_sess.should_stop():
            mon_sess.run(train_op)

def main(argv=None):  # pylint: disable=unused-argument
    cifar.maybe_download_and_extract()
    train()


if __name__ == '__main__':
  FLAGS = parser.parse_args()
  util.proc.set_proc_name('train_on_%s_%s'%(FLAGS.dataset, FLAGS.loss_type))
  tf.app.run()
