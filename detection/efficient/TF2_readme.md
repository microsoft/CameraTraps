### 1. Working directory
1. Keep your working directory as `CamerTraps/detection/efficient/tf2_workspace/`. And let's address it as `{project_dir}`.
2. cd `{project_dir}`


### 1. Setup & Install requirements
   1. Add the TF2 Object detection repo by `git clone https://github.com/gitlost-murali/models.git`
   2. Setup conda environment

      ```
      conda env create --file tf2environment-efficient.yml

      conda activate tf2_camtrap

      cd models/research/

      protoc object_detection/protos/*.proto --python_out=.
      # depending on the CUDA stack installed in your environment, you might have to specify a particular version of tensorflow in the range given here:

      cp object_detection/packages/tf2/setup.py .

      python -m pip install .
      ```
### Generate Pipeline Config

This step is optional.

You can either edit the config file directly at `{project_dir}/models/research/deploy/ssd_efficientdet_d0_512x512_coco17_tpu-8.config`.

OR 

`python pipeline_setup.py {model_variant}` will generate a config file in `{project_dir}/models/research/deploy/pipeline_file.config` for the selected model variant. `{model_variant}` can be d0, d1, d2, d3, ..., d7.

__Note:__ If you are using this method, _update the eval batch_size to 1_.

### Training

```
cd {project_dir}

python models/research/object_detection/model_main_tf2.py \
     --pipeline_config_path=models/research/deploy/pipeline_file.config \
     --model_dir=./training \
     --num_train_steps=40000 \
     --num_steps_btwn_eval=10000 \
     --sample_1_of_n_eval_examples=10 \
     --alsologtostderr
```
### Tensorboard usage
```
cd {project_dir}
tensorboard --logdir 'training/' --host 0.0.0.0
```
### Evaluation
```
Just add `checkpoint-dir` to the training command 

python models/research/object_detection/model_main_tf2.py \
     --pipeline_config_path=models/research/deploy/pipeline_file.config \
     --model_dir=./training \
     --num_train_steps=40000 \
     --num_steps_btwn_eval=10000 \
     --sample_1_of_n_eval_examples=10 \
     --alsologtostderr
     --checkpoint_dir=./training
```


## Changes done to the actual TF2 Object Detection repo
```
diff --git a/research/object_detection/model_lib_v2.py b/research/object_detection/model_lib_v2.py
index 68a4b302..d41ef2ba 100644
--- a/research/object_detection/model_lib_v2.py
+++ b/research/object_detection/model_lib_v2.py
@@ -267,11 +267,11 @@ def eager_train_step(detection_model,
     gradients, _ = tf.clip_by_global_norm(gradients, clip_gradients_value)
   optimizer.apply_gradients(zip(gradients, trainable_variables))
   tf.compat.v2.summary.scalar('learning_rate', learning_rate, step=global_step)
-  tf.compat.v2.summary.image(
-      name='train_input_images',
-      step=global_step,
-      data=features[fields.InputDataFields.image],
-      max_outputs=3)
+#  tf.compat.v2.summary.image(
+#      name='train_input_images',
+#      step=global_step,
+#      data=features[fields.InputDataFields.image],
+#      max_outputs=3)
   return total_loss
 
 
@@ -418,6 +418,11 @@ def train_loop(
     checkpoint_every_n=1000,
     checkpoint_max_to_keep=7,
     record_summaries=True,
+    sample_1_of_n_eval_examples=1,
+    sample_1_of_n_eval_on_train_examples=1,
+    override_eval_num_epochs=True,
+    postprocess_on_cpu=False,
+    eval_frequency=500,
     **kwargs):
   """Trains a model using eager + functions.
 
@@ -464,12 +469,29 @@ def train_loop(
       'train_steps': train_steps,
       'use_bfloat16': configs['train_config'].use_bfloat16 and use_tpu
   })
+  kwargs.update({
+      'sample_1_of_n_eval_examples': sample_1_of_n_eval_examples,
+  })
   configs = merge_external_params_with_configs(
       configs, None, kwargs_dict=kwargs)
   model_config = configs['model']
   train_config = configs['train_config']
   train_input_config = configs['train_input_config']
 
+  # Reference: https://github.com/tensorflow/models/issues/8876
+  eval_config = configs['eval_config']
+  eval_input_configs = configs['eval_input_configs']
+  eval_on_train_input_config = copy.deepcopy(train_input_config)
+  eval_on_train_input_config.sample_1_of_n_examples = (
+    sample_1_of_n_eval_on_train_examples)
+  if override_eval_num_epochs and eval_on_train_input_config.num_epochs != 1:
+    tf.logging.warning('Expected number of evaluation epochs is 1, but '
+                       'instead encountered `eval_on_train_input_config'
+                       '.num_epochs` = '
+                       '{}. Overwriting `num_epochs` to 1.'.format(
+      eval_on_train_input_config.num_epochs))
+    eval_on_train_input_config.num_epochs = 1
+
   unpad_groundtruth_tensors = train_config.unpad_groundtruth_tensors
   add_regularization_loss = train_config.add_regularization_loss
   clip_gradients_value = None
@@ -517,6 +539,16 @@ def train_loop(
     train_input = strategy.experimental_distribute_datasets_from_function(
         train_dataset_fn)
 
+    # Reference: https://github.com/tensorflow/models/issues/8876
+    # Create eval inputs.
+    eval_inputs = []
+    for eval_input_config in eval_input_configs:
+      next_eval_input = inputs.eval_input(
+        eval_config=eval_config,
+        eval_input_config=eval_input_config,
+        model_config=model_config,
+        model=detection_model)
+      eval_inputs.append((eval_input_config.name, next_eval_input))
 
     global_step = tf.Variable(
         0, trainable=False, dtype=tf.compat.v2.dtypes.int64, name='global_step',
@@ -616,6 +648,19 @@ def train_loop(
 
           return _sample_and_train(strategy, train_step_fn, data_iterator)
 
+        # Reference: https://github.com/tensorflow/models/issues/8876
+        def eval_step_fn():
+          for eval_name, eval_input in eval_inputs:
+            summary_writer = tf.compat.v2.summary.create_file_writer(
+              os.path.join(model_dir, 'eval', eval_name))
+            with summary_writer.as_default():
+              eager_eval_loop(detection_model,
+                              configs,
+                              eval_input,
+                              use_tpu=use_tpu,
+                              postprocess_on_cpu=postprocess_on_cpu,
+                              global_step=global_step)
+
         train_input_iter = iter(train_input)
 
         if int(global_step.value()) == 0:
@@ -625,7 +670,7 @@ def train_loop(
         logged_step = global_step.value()
 
         last_step_time = time.time()
-        for _ in range(global_step.value(), train_steps,
+        for stepnum in range(global_step.value(), train_steps,
                        num_steps_per_iteration):
 
           loss = _dist_train_step(train_input_iter)
@@ -649,6 +694,15 @@ def train_loop(
             manager.save()
             checkpointed_step = int(global_step.value())
 
+          if stepnum%eval_frequency==0:
+            tf.logging.info('Running Evaluation for step {}...'.format(global_step.value()))
+            eval_step_fn()
+
+          # Reference: https://github.com/tensorflow/models/issues/8876
+          # latest_checkpoint_tmp = tf.train.latest_checkpoint(model_dir)
+          # if latest_checkpoint_tmp != latest_checkpoint:
+          # latest_checkpoint = latest_checkpoint_tmp
+
   # Remove the checkpoint directories of the non-chief workers that
   # MultiWorkerMirroredStrategy forces us to save during sync distributed
   # training.
diff --git a/research/object_detection/model_main_tf2.py b/research/object_detection/model_main_tf2.py
index 0cf05303..706d61a7 100644
--- a/research/object_detection/model_main_tf2.py
+++ b/research/object_detection/model_main_tf2.py
@@ -29,6 +29,8 @@ python model_main_tf2.py -- \
 """
 from absl import flags
 import tensorflow.compat.v2 as tf
+import sys
+sys.path.append('/home/azureuser/tf_cameratrap/models/research/')
 from object_detection import model_lib_v2
 
 flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config '
@@ -53,6 +55,9 @@ flags.DEFINE_string(
 flags.DEFINE_integer('eval_timeout', 3600, 'Number of seconds to wait for an'
                      'evaluation checkpoint before exiting.')
 
+flags.DEFINE_integer('num_steps_btwn_eval', 1000, 'Number of steps to wait for an'
+                     'evaluation to take place.')
+
 flags.DEFINE_bool('use_tpu', False, 'Whether the job is executing on a TPU.')
 flags.DEFINE_string(
     'tpu_name',
@@ -98,7 +103,8 @@ def main(unused_argv):
     elif FLAGS.num_workers > 1:
       strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
     else:
-      strategy = tf.compat.v2.distribute.MirroredStrategy()
+      strategy = tf.compat.v2.distribute.OneDeviceStrategy(device="/gpu:0")
+      # Reference: https://github.com/tensorflow/models/issues/8876
 
     with strategy.scope():
       model_lib_v2.train_loop(
@@ -107,7 +113,9 @@ def main(unused_argv):
           train_steps=FLAGS.num_train_steps,
           use_tpu=FLAGS.use_tpu,
           checkpoint_every_n=FLAGS.checkpoint_every_n,
-          record_summaries=FLAGS.record_summaries)
+          record_summaries=FLAGS.record_summaries, 
+          sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
+          eval_frequency=FLAGS.num_steps_btwn_eval)
 
 if __name__ == '__main__':
   tf.compat.v1.app.run()
```