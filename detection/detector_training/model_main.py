# This script is taken from https://github.com/tensorflow/models/blob/master/research/object_detection/model_main.py

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Binary to run train and evaluation on object detection model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

from object_detection import model_hparams
from object_detection import model_lib

print('model_main.py, tensorflow version: {}'.format(tf.__version__))

# substitute the AML data reference mount points for relevant parts in the pipeline.config
actual_path_artifacts = os.environ.get('AZUREML_DATAREFERENCE_artifacts')
actual_path_tfrecords = os.environ.get('AZUREML_DATAREFERENCE_tfrecords_mdv4_1')
print('model_main.py, AZUREML_DATAREFERENCE_artifacts is {}'.format(actual_path_artifacts))
print('model_main.py, AZUREML_DATAREFERENCE_tfrecords is {}'.format(actual_path_tfrecords))
print('Listing artifacts:')
artifacts_content = os.listdir(actual_path_artifacts)
for i in artifacts_content:
    print(i)
tfrecords_content = os.listdir(actual_path_tfrecords)
print('len of tfrecords:', len(tfrecords_content))
if len(tfrecords_content) > 0:
    print(tfrecords_content[0])
    print(tfrecords_content[-1])

print('end of mount point checking...')

flags.DEFINE_string(
    'model_dir', None, 'Path to output model directory '
                       'where event and checkpoint files will be written.')
flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config '
                                                  'file.')
flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
flags.DEFINE_boolean('eval_training_data', False,
                     'If training data should be evaluated for this job. Note '
                     'that one call only use this in eval-only mode, and '
                     '`checkpoint_dir` must be supplied.')
flags.DEFINE_integer('sample_1_of_n_eval_examples', 1, 'Will sample one of '
                                                       'every n eval input examples, where n is provided.')
flags.DEFINE_integer('sample_1_of_n_eval_on_train_examples', 5, 'Will sample '
                                                                'one of every n train input examples for evaluation, '
                                                                'where n is provided. This is only used if '
                                                                '`eval_training_data` is True.')
flags.DEFINE_string(
    'hparams_overrides', None, 'Hyperparameter overrides, '
                               'represented as a string containing comma-separated '
                               'hparam_name=value pairs.')
flags.DEFINE_string(
    'checkpoint_dir', None, 'Path to directory holding a checkpoint.  If '
                            '`checkpoint_dir` is provided, this binary operates in eval-only mode, '
                            'writing resulting metrics to `model_dir`.')
flags.DEFINE_boolean(
    'run_once', False, 'If running in eval-only mode, whether to run just '
                       'one round of eval vs running continuously (default).'
)
FLAGS = flags.FLAGS


def main(unused_argv):
    flags.mark_flag_as_required('model_dir')
    flags.mark_flag_as_required('pipeline_config_path')

    # substitute the AML data reference mount points for relevant parts in the pipeline.config and overwrite
    with open(FLAGS.pipeline_config_path) as f:
        config_file = f.read()
    new_config_file = config_file.replace(
        '$AZUREML_DATAREFERENCE_tfrecords', actual_path_tfrecords).replace(
        '$AZUREML_DATAREFERENCE_artifacts', actual_path_artifacts)
    with open(FLAGS.pipeline_config_path, 'w') as f:
        f.write(new_config_file)
    print('model_main.py, main(), finished substituting mount points.')

    config = tf.estimator.RunConfig(
        model_dir=FLAGS.model_dir,
        save_checkpoints_steps= 104012  # save less often than default - 1/5 of an epoch
    )

    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=config,
        hparams=model_hparams.create_hparams(FLAGS.hparams_overrides),
        pipeline_config_path=FLAGS.pipeline_config_path,
        train_steps=FLAGS.num_train_steps,
        sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
        sample_1_of_n_eval_on_train_examples=(
            FLAGS.sample_1_of_n_eval_on_train_examples))
    estimator = train_and_eval_dict['estimator']
    train_input_fn = train_and_eval_dict['train_input_fn']
    eval_input_fns = train_and_eval_dict['eval_input_fns']
    eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
    predict_input_fn = train_and_eval_dict['predict_input_fn']
    train_steps = train_and_eval_dict['train_steps']

    if FLAGS.checkpoint_dir:
        if FLAGS.eval_training_data:
            name = 'training_data'
            input_fn = eval_on_train_input_fn
        else:
            name = 'validation_data'
            # The first eval input will be evaluated.
            input_fn = eval_input_fns[0]
        if FLAGS.run_once:
            estimator.evaluate(input_fn,
                               steps=None,
                               checkpoint_path=tf.train.latest_checkpoint(
                                   FLAGS.checkpoint_dir))
        else:
            model_lib.continuous_eval(estimator, FLAGS.checkpoint_dir, input_fn,
                                      train_steps, name)
    else:
        train_spec, eval_specs = model_lib.create_train_and_eval_specs(
            train_input_fn,
            eval_input_fns,
            eval_on_train_input_fn,
            predict_input_fn,
            train_steps,
            eval_on_train_data=False)

        # Currently only a single Eval Spec is allowed.

        # throttle_secs is not documented in eval.proto. This replaces eval_interval_secs somewhat
        throttle_secs = 60 * 60  # every 60 min

        eval_spec = eval_specs[0]

        my_eval_spec = tf.estimator.EvalSpec(
            name=eval_spec.name,
            input_fn=eval_spec.input_fn,
            steps=None,
            exporters=eval_spec.exporters,
            start_delay_secs=1800,  # 30 minutes - does not seem to be respected...
            throttle_secs=throttle_secs)

        print('=========== my_eval_spec')
        print(my_eval_spec)
        print('=========================')

        tf.estimator.train_and_evaluate(estimator, train_spec, my_eval_spec)


if __name__ == '__main__':
    tf.app.run()
