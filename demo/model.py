
import tensorflow as tf
import numpy as np

import os
import urllib.request as urlopen
from io import BytesIO
from PIL import Image

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker


class Model:
    def __init__(self, checkpoint):
        self.confidenceThreshold = 0.9
        print('Creating Graph...')
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(checkpoint, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        print('...done')

    def generate_image_detections(self, image):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # img_file = BytesIO(urlopen.urlopen(url).read())
                # image = Image.open(img_file).convert('RGB')

                print('image shape', image.size)
                # image = mpimg.imread(url)
                imageNP_expanded = np.expand_dims(image, axis=0)
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                box = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                score = self.detection_graph.get_tensor_by_name('detection_scores:0')
                clss = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                
                rbboxes = []
                # Actual detection
                try:
                    (obox, oscore, oclss, num_detections) = sess.run(
                            [box, score, clss, num_detections],
                            feed_dict={image_tensor: imageNP_expanded})

                    bboxes, scores, classes = obox[0], oscore[0], oclss[0]
                    
                    
                    #calculate the size
                    iBox = 0; 
                    box = bboxes[iBox]
                    dpi = 100
                    s = image.size   
                    imageHeight = s[1]
                    imageWidth = s[0]

                    for iBox,box in enumerate(bboxes):
                        iScore = scores[iBox]
                        if iScore < self.confidenceThreshold:
                            continue

                        topRel = box[0]
                        leftRel = box[1]
                        bottomRel = box[2]
                        rightRel = box[3]
                        
                        x = leftRel * imageWidth
                        y = topRel * imageHeight
                        w = (rightRel-leftRel) * imageWidth
                        h = (bottomRel-topRel) * imageHeight


                        rbboxes.append({
                                'x': x,
                                'y': y,
                                'w': w,
                                'h': h,
                                'score': str(iScore),
                                'class': str(classes[iBox])
                            })
                
                except Exception as e:
                    rbboxes = str(e)
                    
                return rbboxes

    def draw_bounding_box(self, bboxes, image, outputFileName, confidenceThreshold=0.9):
        number_of_det = 0
        # img_file = BytesIO(urlopen.urlopen(inputFileName).read())
        # image = Image.open(img_file).convert('RGB')

        s = image.size;         
        imageHeight = s[1]; 
        imageWidth = s[0]

        dpi = 100
        figsize = imageWidth / float(dpi), imageHeight / float(dpi)
        fig = plt.figure(figsize=figsize)
        ax = plt.axes([0,0,1,1])

        ax.imshow(image)
        ax.set_axis_off()

        for iBox,box in enumerate(bboxes):
            score = float(box.get('score'))
            if score < confidenceThreshold:
                continue


            iLeft = box.get('x')
            iBottom = box.get('y')
            w = box.get('w')
            h = box.get('h')
            rect = patches.Rectangle((iLeft,iBottom),w,h,linewidth=4,edgecolor='r',facecolor='none')
            
            # Add the patch to the Axes
            ax.add_patch(rect)
            number_of_det += 1

        
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        ax.xaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.axis('tight')
        ax.set(xlim=[0,imageWidth],ylim=[imageHeight,0],aspect=1)
        plt.axis('off')                

        # plt.savefig(outputFileName, bbox_inches='tight', pad_inches=0.0, dpi=dpi, transparent=True)
        plt.savefig(outputFileName, dpi=dpi, transparent=True) 

        return number_of_det

            
            



#model configuration
checkpoint = 'checkpoint/frozen_inference_graph.pb'

model = Model(checkpoint)
