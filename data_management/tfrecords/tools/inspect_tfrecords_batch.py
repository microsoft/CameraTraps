
# coding: utf-8

# In[1]:


import glob

import PIL
import tensorflow as tf

coord = None


# In[2]:


TFRECORDS_PATH = list(glob.glob('/data/lila/nacti/cropped_tfrecords/t*'))



# In[ ]:


def analyze_record(TFRECORDS_PATH):
    def read_and_decode(filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        feat_keys = { 'image/encoded': tf.FixedLenFeature([], tf.string),
                      'image/filename': tf.FixedLenFeature([], tf.string),
                      'image/class/label': tf.FixedLenFeature([], tf.int64),
                      'image/class/text': tf.FixedLenFeature([], tf.string),
                      'image/height': tf.FixedLenFeature([], tf.int64),
                      'image/width': tf.FixedLenFeature([], tf.int64)
                     }
        features = tf.parse_single_example(
                        serialized_example,
                        # Defaults are not specified since both keys are required.
                        features=feat_keys)
        output = {k:features[k] for k,v in feat_keys.items() if k != 'image/encoded'}
        output['image/encoded'] = tf.decode_raw(features['image/encoded'], tf.uint8)
        return output

    with tf.Session() as sess:

        #image, label, height, width

        filename_queue = tf.train.string_input_producer([TFRECORDS_PATH])
        output = read_and_decode(filename_queue)
        image = tf.reshape(output['image/encoded'], [output['image/height'], output['image/width'], 3])
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        #print(sess.run([output['image/filename'], output['image/class/text'], output['image/class/label'], output['image/height'], output['image/width']]))

        finished = set()
        while True:
            path, h, w = sess.run([output['image/filename'], output['image/height'], output['image/width']])
            if path in finished:
                break
            if not PIL.Image.open(path).size == (w,h):
                print(path, h, w, 'is invalid')
                exit()
            finished.add(path)

        coord.request_stop()
        coord.join(threads)
    tf.reset_default_graph()
from multiprocessing import Pool

p = Pool(maxtasksperchild=1)
#tasks = list(glob.glob('/data/lila/nacti/cropped_tfrecords/t*'))
#for _ in tqdm.tqdm(p.imap(analyze_record, tasks, chunksize=10), total=len(tasks)):
#    pass
p.map(analyze_record, list(glob.glob('/data/lila/nacti/cropped_tfrecords/t*')), chunksize=10000)
#Parallel(n_jobs=8)(delayed(analyze_record)(path) for path in list(glob.glob('/data/lila/nacti/cropped_tfrecords/t*')))
