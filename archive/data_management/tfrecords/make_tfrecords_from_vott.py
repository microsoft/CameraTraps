#
# create_tfrecords_format.py
#
# This script creates a tfrecords file from a dataset in VOTT format.

#%% Imports and environment

import os
from PIL import Image
from tqdm import tqdm
import glob
import numpy as np
import argparse
import sys
if sys.version_info.major >= 3:
    from utils.create_tfrecords_py3 import create
else:
    from utils.create_tfrecords import create


class VottBboxDataset:
    """Bounding box dataset loader for VOTT / CNTK Faster RCNN format

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

    The type of the image, the bounding boxes and the labels are as follows.

    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`

    Args:
        root (string): path to folder, which contains the images
        class_names (list of strings): the classnames, which are used to derive
        the class ID

    """
    
    def __init__(self, root, class_names = [], store_empty_images=False):
        self.root = root
        print('Loading images from folder ' + root)
        # set up the filenames and annotations
        old_dir = os.getcwd()
        os.chdir(root)
        self.impaths = sorted([fi for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.PNG', '*.png'] for fi in glob.glob(ext)])
        print('Found {} images'.format(len(self.impaths)))
        self.image_ids = list(range(len(self.impaths)))
        
        # This loop reads the bboxes and corresponding labels and assigns them
        # the correct image. Kind of slow at the moment...
        self.bboxes = [[] for _ in self.image_ids]
        self.labels = [[] for _ in self.image_ids]
        self.class_names = class_names
        empty_images = []
        for image_id, impath in enumerate(self.impaths):
            with open(os.path.splitext(impath)[0] + '.bboxes.labels.tsv', 'rt') as labelfile:
                bbox_labels = labelfile.read().splitlines()
                # If needed: merging all classes
                #bbox_labels = ['Animal' for _ in bbox_labels]
            # BBox coords are stored in the format
            # x_min (of width axis) y_min (of height axis), x_max, y_max
            # Coordinate system starts in top left corner
            bbox_coords = np.loadtxt(os.path.splitext(impath)[0] + '.bboxes.tsv', dtype=np.int32)
            if len(bbox_coords.shape) == 1 and bbox_coords.size > 0:
                bbox_coords = bbox_coords[None,:]
            assert len(bbox_coords) == len(bbox_labels)
            width,height = Image.open(self.impaths[image_id]).size
            for i in range(len(bbox_coords)):
                if bbox_labels[i] not in self.class_names:
                    self.class_names.append(bbox_labels[i])
                bb = bbox_coords[i]
                if np.all(bb >= 0) and bb[0] <= width and bb[2] <= width and bb[1] <= height and bb[3] <= height and bb[0] < bb[2] and bb[1] < bb[3]:
                    # In this framework, we need ('ymin', 'xmin', 'ymax', 'xmax') format
                    self.bboxes[image_id].append([bb[1],bb[0],bb[3],bb[2]])
                    self.labels[image_id].append(self.class_names.index(bbox_labels[i]))
            if len(self.bboxes[image_id]) == 0:
                empty_images.append(image_id)
        if not store_empty_images:
            for empty_image_id in empty_images[::-1]:
                print("Deleting image {} as all bounding boxes are outside".format(empty_image_id) + \
                         "of the image or no bounding boxes are provided")
                del self.impaths[empty_image_id]
                del self.image_ids[empty_image_id]
                del self.bboxes[empty_image_id]
                del self.labels[empty_image_id]
        
        self.classes = list(range(len(self.class_names)))
        # print out some stats
        print("The dataset has {} images containing {} classes".format(
                  len(self.image_ids),
                  len(self.classes)))
        os.chdir(old_dir)
        
        # To make sure we loaded the bboxes correctly:        
        #self.validate_bboxes()
        print("All checks passed")
            

    def __len__(self):
        return len(self.image_ids)
        
    def validate_bboxes(self):
        import traceback
        import sys
        from tqdm import tqdm
        try:
            # For each image in the data set...
            for idx in tqdm(range(len(self.image_ids))):
                img_file = os.path.join(self.root, self.impaths[idx])
                width,height = Image.open(img_file).size
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
            import ipdb
            ipdb.post_mortem(tb)
            
    def get_class_count(self):
        return np.max(self.classes).tolist() + 1

    def get_class_names(self):
        return self.class_names


#%% Main tfrecord generation function

def create_tfrecords_format(root, class_names, output_tfrecords_folder, dataset_name,
                                                   num_threads=5, ims_per_record=200,
                                                   store_empty_images=False):
    dataset = VottBboxDataset(root, class_names, store_empty_images=store_empty_images)

    vis_data = []
    for i in tqdm(range(len(dataset))):
        img_file = os.path.join(dataset.root, dataset.impaths[i])

        bboxes = np.array(dataset.bboxes[i])
        labels = np.array(dataset.labels[i])
        image_id = np.array([dataset.image_ids[i]])

        image_data = {}
        image_data['filename'] = img_file
        image_data['id'] = image_id

        # Propagate optional metadata to tfrecords
        im_w, im_h = Image.open(image_data['filename']).size

        image_data['height'] = im_h
        image_data['width'] = im_w

        image_data['object'] = {}
        image_data['object']['count'] = 0
        image_data['object']['id'] = []
        image_data['object']['bbox'] = {}
        image_data['object']['bbox']['xmin'] = []
        image_data['object']['bbox']['xmax'] = []
        image_data['object']['bbox']['ymin'] = []
        image_data['object']['bbox']['ymax'] = []
        image_data['object']['bbox']['label'] = []
        image_data['object']['bbox']['text'] = []

        for bbox_id in range(len(bboxes)):
            image_data['object']['count'] += 1
            image_data['object']['id'].append(bbox_id)
            image_data['object']['bbox']['label'].append(labels[bbox_id])
            image_data['object']['bbox']['text'].append(class_names[labels[bbox_id]])

            xmin = bboxes[bbox_id][1] / float(im_w)
            xmax = bboxes[bbox_id][3] / float(im_w)
            ymin = bboxes[bbox_id][0] / float(im_h)
            ymax = bboxes[bbox_id][2] / float(im_h)

            try:
                assert np.all(np.array([xmin,ymin,xmax,ymax]) <= 1)
                assert np.all(np.array([xmin,ymin,xmax,ymax]) >= 0)
                assert xmax > xmin
                assert ymax > ymin
            except:
                import ipdb; ipdb.set_trace()

            image_data['object']['bbox']['xmin'].append(xmin)
            image_data['object']['bbox']['xmax'].append(xmax)
            image_data['object']['bbox']['ymin'].append(ymin)
            image_data['object']['bbox']['ymax'].append(ymax)

        # endfor each annotation for the current image
        vis_data.append(image_data)
    # endfor each image

    print('Creating tfrecords for {} from {} images'.format(dataset_name,len(dataset)))

    # Calculate number of shards to get the desired number of images per record,
    # ensure it is evenly divisible by the number of threads
    num_shards = int(np.ceil(float(len(dataset))/ims_per_record))
    while num_shards % num_threads:
        num_shards += 1
    print('Number of shards: ' + str(num_shards))

    failed_images = create(
      dataset=vis_data,
      dataset_name=dataset_name,
      output_directory=output_tfrecords_folder,
      num_shards=num_shards,
      num_threads=num_threads,
      store_images=True
    )

    return failed_images, dataset.get_class_names()

#%% Command-line driver

def parse_args():
    parser = argparse.ArgumentParser(description = 'Make tfrecords from a VOTT style detection dataset.')
    parser.add_argument('--train_dataset_root', help='Root dir of VOTT dataset used for training, ' + \
                         ' should contain images without any subdirectories.', type=str, required=True)
    parser.add_argument('--test_dataset_root', help='Root dir of VOTT dataset used for testing, ' + \
                         ' should contain images without any subdirectories.', type=str, required=True)
    parser.add_argument('--output_tfrecords_folder', dest='output_tfrecords_folder',
                         help='Path to folder to save tfrecords in',
                         type=str, required=True)
    parser.add_argument('--num_threads', dest='num_threads',
                         help='Number of threads to use while creating tfrecords',
                         type=int, default=5)
    parser.add_argument('--ims_per_record', dest='ims_per_record',
                         help='Number of images to store in each tfrecord file',
                         type=int, default=200)
    parser.add_argument('--store_empty_images', action='store_true',
                         help='If set, empty images will be stores as well.')
    args = parser.parse_args()

    return args


#%% Driver
if __name__ == '__main__':
    args = parse_args()

    try:
        os.makedirs(args.output_tfrecords_folder)
    except Exception as e:
        if os.path.isdir(args.output_tfrecords_folder):
            raise Exception('Directory {} already exists, '.format(args.output_tfrecords_folder) + \
                            'please remove any existing content')
        else:
            raise e

    failed_train, classnames = create_tfrecords_format(args.train_dataset_root,
                                                       ['__background__'],
                                                       args.output_tfrecords_folder,
                                                       'train',
                                                       args.num_threads,
                                                       args.ims_per_record,
                                                       args.store_empty_images)

    failed_test, final_classnames = create_tfrecords_format(args.test_dataset_root,
                                                       classnames,
                                                       args.output_tfrecords_folder,
                                                       'test',
                                                       args.num_threads,
                                                       args.ims_per_record,
                                                       args.store_empty_images)

    label_map = []
    for idx, cname in list(enumerate(final_classnames))[1:]:
        label_map += ['item {{name: "{}" id: {}}}\n'.format(cname, idx)]
    with open(os.path.join(args.output_tfrecords_folder, 'label_map.pbtxt'), 'w') as f:
        f.write(''.join(label_map))

    print('Finished with {} failed images'.format(len(failed_train) + len(failed_test)))
