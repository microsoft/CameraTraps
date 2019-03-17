################
#
# iwildcam_dataset.py
#
# Loader for the iWildCam detection data set.
#
################

import os
import xml.etree.ElementTree as ET
import numpy as np
import json
from .util import read_image
import PIL


def load_taxonomy(ann_data, tax_levels, classes):

    # loads the taxonomy data and converts to ints
    taxonomy = {}

    if 'categories' in ann_data.keys():
        num_classes = len(ann_data['categories'])
        for tt in tax_levels:
            tax_data = [aa[tt] for aa in ann_data['categories']]
            _, tax_id = np.unique(tax_data, return_inverse=True)
            taxonomy[tt] = dict(zip(range(num_classes), list(tax_id)))
    else:
        # set up dummy data
        for tt in tax_levels:
            taxonomy[tt] = dict(zip([0], [0]))

    # create a dictionary of lists containing taxonomic labels
    classes_taxonomic = {}
    for cc in classes:
        tax_ids = [0]*len(tax_levels)
        for ii, tt in enumerate(tax_levels):
            tax_ids[ii] = taxonomy[tt][cc]
        classes_taxonomic[cc] = tax_ids

    return taxonomy, classes_taxonomic


class IWildCamBboxDataset:
    """Bounding box dataset for iWildCam 2018.

    The index corresponds to each image.

    When queried by an index, if :obj:`return_difficult == False`,
    this dataset returns a corresponding
    :obj:`img, bbox, label`, a tuple of an image, bounding boxes and labels.
    This is the default behaviour.
    If :obj:`return_difficult == True`, this dataset returns corresponding
    :obj:`img, bbox, label, difficult`. :obj:`difficult` is a boolean array
    that indicates whether bounding boxes are labeled as difficult or not.

    The bounding boxes are packed into a two dimensional tensor of shape
    :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the
    four attributes are coordinates of the top left and the bottom right
    vertices.

    The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
    :math:`R` is the number of bounding boxes in the image.
    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`VOC_BBOX_LABEL_NAMES`.

    The array :obj:`difficult` is a one dimensional boolean array of shape
    :math:`(R,)`. :math:`R` is the number of bounding boxes in the image.
    If :obj:`use_difficult` is :obj:`False`, this array is
    a boolean array with all :obj:`False`.

    The type of the image, the bounding boxes and the labels are as follows.

    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`
    * :obj:`difficult.dtype == numpy.bool`

    Args:
        data_dir (string): Path to the root of the training data. 
            i.e. "/data/image/voc/VOCdevkit/VOC2007/"
        root (string): path to folder, which contains train_val_images
        ann_file (string): path to file, which contains the bounding box 
            annotations
    """
    
    def __init__(self, root, ann_file, max_images=None):

        self.root = root
        print('Loading annotations from: ' + os.path.basename(ann_file))
        with open(ann_file) as data_file:
            ann_data = json.load(data_file)

        # set up the filenames and annotations
        self.impaths = np.array([aa['file_name'] for aa in ann_data['images']])
        self.image_ids = np.array([aa['id'] for aa in ann_data['images']])
        
        # This loop reads the bboxes and corresponding labels and assigns them
        # the correct image. Kind of slow at the moment...
        self.bboxes = [[] for _ in self.image_ids]
        self.labels = [[] for _ in self.image_ids]
        # To speed up the loop, creating mapping for image_id to list index 
        image_id_to_idx = {id:idx for idx, id in enumerate(self.image_ids)}
        for ann in ann_data['annotations']:
            idx = image_id_to_idx[ann['image_id']]
            #check that the image contains an animal, if not, don't append a box or label to the
            #image list
            if 'bbox' in ann:

                # Bboxes should have ('ymin', 'xmin', 'ymax', 'xmax') format
                self.bboxes[idx].append([ann['bbox'][1], ann['bbox'][0],
                                          ann['bbox'][1] + ann['bbox'][3],
                                          ann['bbox'][0] + ann['bbox'][2]])
                # Currently we take the label from the annotation file, non-consecutive-
                # label-support would be great
                self.labels[idx].append(ann['category_id'])
            else:
                #self.bboxes[idx].append([-1.,-1.,0.,0.])
                #self.labels[idx].append(30)
         
                self.bboxes[idx].append([])
                self.labels[idx].append(30)

        # load classes
        self.classes = np.unique(sorted([cat['id'] for cat in ann_data['categories']]))
        
        self.class_names = ['' for _ in range(self.get_class_count())]
        
        for cat in ann_data['categories']:    
            self.class_names[cat['id']] = '{}'.format(cat['name'])
        
        # print out some stats
        print("The dataset has {} images containing {} classes".format(
                  len(self.image_ids),
                  len(self.classes)))
        
        if max_images is not None:
            print('Selecting a subset of {} images from training and validation'.
                format(max_images))
            self.impaths = self.impaths[:max_images]
            self.image_ids = self.image_ids[:max_images]
            self.bboxes = self.bboxes[:max_images]
            self.labels = self.labels[:max_images]
                                      
        # To make sure we loaded the bboxes correctly:        
        # self.validate_bboxes()
            

    def __len__(self):

        return len(self.image_ids)
        

    def validate_bboxes(self):

        import ipdb      
        import traceback
        import sys
        from tqdm import tqdm
        try:
            for idx in tqdm(range(len(self.image_ids))):
                img_file = os.path.join(self.root, self.impaths[idx])
                width,height = PIL.Image.open(img_file).size
                for bbox in self.bboxes[idx]:
                    assert bbox[1] <= width 
                    assert bbox[3] <= width
                    assert bbox[0] <= height
                    assert bbox[2] <= height
                    assert bbox[3] > bbox[1]
                    assert bbox[2] > bbox[0]
                    # Make sure all are greater equal 0
                    assert np.all(np.array(bbox) >= 0)
        except:
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            ipdb.post_mortem(tb)
            

    def get_class_count(self):

        # We have to add 1 as the framework assumes that labels start from 0
        return np.max(self.classes).tolist() + 1

    def find_files(self, str):
        return range(0,len(self.impaths)), self.impaths.tolist()

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image in CHW format, bounding boxes in 
            ('ymin', 'xmin', 'ymax', 'xmax')  format, label as int32
            starting from 0 and difficult_flag, which is usually zero.
        """

        img_file = os.path.join(self.root, self.impaths[i])
        img = read_image(img_file, color=True)
        
        bboxes = self.bboxes[i]
        labels = np.asarray(self.labels[i])
        difficulties = [0 for _ in labels]
        image_id = [self.image_ids[i]]
        #print(bboxes)

        img = np.transpose(img,[1, 2, 0])
        img = img/255

        return img, bboxes, labels, difficulties, image_id, img_file, self.impaths[i]

    def get_files(self):
        return self.impaths
    
    __getitem__ = get_example
    

    def get_class_names(self):

        return self.class_names
