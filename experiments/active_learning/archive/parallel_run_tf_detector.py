######
#
# run_tf_detector.py
#
# Functions to load a TensorFlow detection model, run inference,
# and render bounding boxes on images.
#
# See the "test driver" cell for example invocation.
#
######

#%% Constants, imports, environment

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import os
import glob
from PIL import Image
from multiprocessing import Process


#%% Core detection functions

def load_model(checkpoint):
    """
    Load a detection model (i.e., create a graph) from a .pb file
    """

    print('Creating Graph...')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(checkpoint, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    print('...done')

    return detection_graph


def generate_detections(pid, detection_graph,image_paths):
    """
    boxes,scores,classes,images = generate_detections(detection_graph,images)

    Run an already-loaded detector network on a set of images.

    [images] can be a list of numpy arrays or a list of filenames.  Non-list inputs will be
    wrapped into a list.

    Boxes are returned in relative coordinates as (top, left, bottom, right); x,y origin is the upper-left.
    """

    os.environ["CUDA_VISIBLE_DEVICES"]=str(pid//2)
    imagenames = image_paths.copy()
    images=[]
    # Load images if they're not already numpy arrays
    for iImage,image in enumerate(imagenames):
        images.append(Image.open(imagenames[iImage]))

    boxes = []
    scores = []
    classes = []
    
    # nImages = len(images)
    to_remove=[]
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            box = detection_graph.get_tensor_by_name('detection_boxes:0')
            score = detection_graph.get_tensor_by_name('detection_scores:0')
            clss = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0') 
            for iImage,imageNP in enumerate(images):
                try:
                  imageNP_expanded = np.expand_dims(imageNP, axis=0)
               
                  # Actual detection
                  (boxv, scorev, clssv, num_detectionsv) = sess.run(
                        [box, score, clss, num_detections],
                        feed_dict={image_tensor: imageNP_expanded})
                  boxes.append(boxv)
                  scores.append(scorev)
                  classes.append(clssv)
                except:
                  to_remove.append(iImage)
                  pass
    for index in sorted(to_remove, reverse=True):
      del images[index]
      del imagenames[index]
    boxes = np.squeeze(np.array(boxes))
    scores = np.squeeze(np.array(scores))
    classes = np.squeeze(np.array(classes)).astype(int)

    return scores,boxes,imagenames


def divide(t,n,i):
    length=t/(n+0.0)
    #print length,(i-1)*length,i*length
    return int(round(i*length)),int(round((i+1)*length))

def do_chunk(pid,model,filelist):

  #print(len(filelist))
  out =open("detections_"+str(pid)+".csv", "w")
  for i in range(0,len(filelist),500):
    print(pid,i)
    TARGET_IMAGES= filelist[i:min(i+500,len(filelist))]
    scores, boxes, imagenames= generate_detections(pid,model,TARGET_IMAGES)
    for score,box,imagename in zip(scores,boxes,imagenames):
      for s,b in zip(score,box):
        if s>=0.9:
          print("%s,%.6f,%.6f,%.6f,%.6f,%.6f"%(imagename,s,b[0],b[1],b[2],b[3]),file=out) 
    out.flush()
  out.close()


#%% Test driver
  
MODEL_FILE = r'mnt/models/object_detection/faster_rcnn_inception_resnet_v2_atrous/train_on_eccv_18_and_imerit_2/frozen_inference_graph.pb'
allfiles=[]
subfolders= glob.glob('/datadrive0/dataD/snapshot/S3/')
print(subfolders)
for subf in subfolders:
  for path, subdirs, files in os.walk(subf):
    for f in files:
      if f.endswith(".JPG") or f.endswith(".jpg"):
        allfiles.append(os.path.join(path,f))
        
# Load and run detector on target images
print(len(allfiles))
total_records=len(allfiles)
total_processors=6
print(total_records)
detection_graph = load_model(MODEL_FILE)
for i in range(0,total_processors):
  st,ln=divide(total_records,total_processors,i)
  print(st,ln,"mio mio")
  p1 = Process(target=do_chunk, args=(i,detection_graph,allfiles[st:ln]))
  p1.start()
