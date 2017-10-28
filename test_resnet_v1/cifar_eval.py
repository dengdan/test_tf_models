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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import cifar

import util
parser = cifar.parser

parser.add_argument('--loss_type', type=str, default='focal_loss',
                                        help='loss type')

parser.add_argument('--eval_dir', type=str,
                                        help='Directory where to write event logs.')

parser.add_argument('--eval_data', type=str, default='test',
                                        help='Either `test` or `train_eval`.')

parser.add_argument('--checkpoint_dir', type=str,
                                        help='Directory where to read model checkpoints.')


parser.add_argument('--num_examples', type=int, default=10000,
                                        help='Number of examples to run.')

parser.add_argument('--run_once', type=bool, default=False,
                                        help='Whether to run eval only once.')


def eval_once(saver, summary_writer, corrected, summary_op):
    """Run Eval once.

    Args:
        saver: Saver.
        summary_writer: Summary writer.
        corrects: Top K op.
        summary_op: Summary op.
    """

    with tf.Session(config=util.tf.gpu_config(allow_growth=True)) as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #     /my-favorite-path/cifar_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return
        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                                                 start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            correct_count = 0    # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                corrected_val = sess.run([corrected])
                correct_count += np.sum(corrected_val) 
                step += 1
                
            precision = correct_count * 1.0 / total_sample_count
            print('step %r in %s on resnet, precision = %f ' % (int(global_step), FLAGS.dataset, precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:    # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Eval CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR.
        eval_data = FLAGS.eval_data == 'test'
        images, labels = cifar.inputs(eval_data=eval_data, is_cifar10=FLAGS.dataset == 'cifar-10')

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cifar.inference(images, is_training=False)

        scores = util.tf.softmax(logits)
        predicted = tf.arg_max(scores, dimension=-1)
        corrects = tf.cast(tf.equal(predicted, tf.cast(labels, tf.int64)), tf.int32)
        
        # Restore the moving average version of the learned variables for eval.
#         variable_averages = tf.train.ExponentialMovingAverage(
#                 cifar.MOVING_AVERAGE_DECAY)
#         variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver()

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        while True:
                for _ in util.tf.wait_for_checkpoint(FLAGS.checkpoint_dir):
                        eval_once(saver, summary_writer, corrects, summary_op)


def main(argv=None):    # pylint: disable=unused-argument
    util.io.mkdir(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    util.proc.set_proc_name('eval_resnet_%s' % (FLAGS.dataset))
    tf.app.run()
