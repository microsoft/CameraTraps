""" Script for preparing annotation files"""
import json
import argparse
from pathlib import Path
from glob import glob # For listing filenames
import os, sys # For arguments
from tqdm import tqdm # For checking progress
from cv2 import imread # For finding dimensions
from typing import Text, Union, Dict, Sequence

class Processor:
    """
    Takes the json file and converts it according
    to the template (categories) given and produces
    annotation files.
    """
    def __init__(self, split_type_: Text, json_file_: Text, dataset_directory_:Text, \
                 id2categories_: Dict[int,str], categories2id_: Dict[str,int],
                 coco_camtrap_:Dict):

        if not os.path.exists(dataset_directory/'annotations/'):
            os.mkdir(dataset_directory/'annotations/')
        if os.path.exists(dataset_directory/f'annotations/instances_{split_type}.json'):
            sys.exit("File already exists...\nQuitting the program....")

        self.split_type = split_type_
        self.json_file = json_file_
        self.dataset_directory = dataset_directory_
        self.id2categories = id2categories_
        self.categories2id = categories2id_
        self.coco_camtrap = coco_camtrap_
        self.split_filenames = [os.path.basename(imgname) \
                                for imgname in glob(str(dataset_directory/f"{split_type}/*"))]

        self.images = []
        self.imageid = 1
        self.annotations = []
        self.annotationid = 1

    def process(self):

        """
        Go through each instance in bboxes_inc_empty_20200325.json.
        Keep a counter for index count.
        filename = download_id + .png
        --------------------------------------------------------
        image = {"id":1,"file_name":f".","height":250,"width":500}
        images.append(image)

        annotation = {"id":1,"image_id":94,"category_id":2,
                     "bbox":[0.695,0.227,0.288,0.455],"iscrowd":0}
        annotations.append(annotation)
        annotation = {"id":2,"image_id":94,"category_id":1,
                     "bbox":[0.385,0.27,0.288,0.455],"iscrowd":0}
        annotations.append(annotation)

        coco_camtrap['images'] = images
        coco_camtrap['annotations'] = annotations
        """

        with open(f"{self.json_file}", 'r') as fhandler:
            instances = json.load(fhandler)

        for instance in tqdm(instances, desc='Progress:'):
            filename = instance['download_id'] + '.jpg'
            if not filename in self.split_filenames:
                continue # Skip if the image is from another split type
            try:
                img = imread(str(self.dataset_directory/f'{split_type}/{filename}'))
                if img is None: continue
                height, width, _ = img.shape
            except Exception as e:
                print(f'{filename} is skipped due to ', e)
                continue

            image = {"id":self.imageid, "file_name":f"{filename}", "height":height, "width":width}
            labels = instance['bbox']
            for label in labels:
                category = label['category']
                category_id = self.categories2id[category]
                bbox = [label['bbox'][0]*width, label['bbox'][1]*height,\
                        label['bbox'][2]*width, label['bbox'][3]*height]
                area = bbox[2] * bbox[3]
                annotation = {"id":self.annotationid, "image_id":self.imageid,
                              "category_id":category_id, "bbox":bbox,
                              "area":area, "iscrowd": 0}
                self.annotations.append(annotation)
                self.annotationid += 1

            self.images.append(image)
            self.imageid += 1

        self.coco_camtrap['images'] = self.images
        self.coco_camtrap['annotations'] = self.annotations

        with open(self.dataset_directory/f'annotations/instances_{self.split_type}.json', "w") as fhandler:
            json.dump(self.coco_camtrap, fhandler)

def get_args() -> Union[bytes]:
    """
    Argument parser
    """
    parser = argparse.ArgumentParser("Create JSON splits from mdv4 labelling file")
    parser.add_argument('-f', '--file', type=str,
                        default='datasets/camtrap/bboxes_inc_empty_20200325.json',
                        help = 'Look for bboxes_inc_empty_20200325.json')
    parser.add_argument('-s', '--split', type=str, default='train',
                        help='It will go to folder_of_{file}/{split}/ for reading images.'\
                             'For ex: datasets/camtrap/train/')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    OPT = get_args()
    split_type = OPT.split # train / val / test
    json_file = OPT.file # path of `bboxes_inc_empty_20200325.json`
    dataset_directory = Path(json_file).resolve().parent
    id2categories = {1:'animal', 2:'person', 3:'group', 4:'vehicle'}
    categories2id = dict((v, k) for (k, v) in id2categories.items())
    coco_camtrap = {"info": \
                   {"description": "", "url": "", "version": "", "year": 2020,\
                   "contributor": "", "date_created": "2020-04-14 01:45:18.567988"},\
                   "licenses": [{"id": 1, "name": 'null', "url": 'null'}], \
                   "categories": [{"id": 1, "name": f"{id2categories[1]}", "supercategory": "None"}, \
                                  {"id": 2, "name": f"{id2categories[2]}", "supercategory": "None"}, \
                                  {"id": 3, "name": f"{id2categories[3]}", "supercategory": "None"}, \
                                  {"id": 4, "name": f"{id2categories[4]}", "supercategory": "None"}]}

    processor = Processor(split_type, json_file, dataset_directory, id2categories, categories2id, coco_camtrap)
    processor.process()