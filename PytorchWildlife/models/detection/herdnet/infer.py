__copyright__ = \
    """
    Copyright (C) 2024 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the MIT License.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: March 18, 2024
    """
__author__ = "Alexandre Delplanque"
__license__ = "MIT License"
__version__ = "0.2.1"


import argparse
import torch
import os
import pandas
import warnings
import numpy
import PIL
import sys

# Get the path to the 'models' directory # Removed relative import from BaseDetector. TODO: Will this work in all cases ?
current_dir = os.path.dirname(os.path.abspath(__file__))  
parent_dir = os.path.dirname(current_dir)  
sys.path.insert(0, parent_dir)  

import albumentations as A

from torch.utils.data import DataLoader
from PIL import Image

from animaloc.data.transforms import DownSample, Rotate90
from animaloc.models import LossWrapper
from animaloc.models import HerdNet as HerdNetModel
from animaloc.eval import HerdNetStitcher, HerdNetEvaluator
from animaloc.eval.metrics import PointsMetrics
from animaloc.datasets import CSVDataset
from animaloc.utils.useful_funcs import mkdir, current_date
from animaloc.vizual import draw_points, draw_text

from herdnet import HerdNet

warnings.filterwarnings('ignore')
PIL.Image.MAX_IMAGE_PIXELS = None


parser = argparse.ArgumentParser(
    prog='inference', 
    description='Collects the detections of a pretrained HerdNet model on a set of images '
    )

parser.add_argument('root', type=str,
    help='path to the JPG images folder (str)')
parser.add_argument('pth', type=str,
    help='path to PTH file containing your model parameters (str)')  
parser.add_argument('-size', type=int, default=512,
    help='patch size use for stitching. Defaults to 512.')
parser.add_argument('-over', type=int, default=160,
    help='overlap for stitching. Defaults to 160.')
parser.add_argument('-device', type=str, default='cuda',
    help='device on which model and images will be allocated (str). \
        Possible values are \'cpu\' or \'cuda\'. Defaults to \'cuda\'.')
parser.add_argument('-ts', type=int, default=256,
    help='thumbnail size. Defaults to 256.')
parser.add_argument('-pf', type=int, default=10,
    help='print frequence. Defaults to 10.')
parser.add_argument('-rot', type=int, default=0,
    help='number of times to rotate by 90 degrees. Defaults to 0.')

args = parser.parse_args()

def main():

    # Create destination folder
    curr_date = current_date()
    dest = os.path.join(args.root, f"{curr_date}_HerdNet_results")
    mkdir(dest)
    
    # Read info from PTH file
    map_location = torch.device('cpu')
    if torch.cuda.is_available():
        map_location = torch.device('cuda')

    checkpoint = torch.load(args.pth, map_location=map_location)
    classes = checkpoint['classes']
    num_classes = len(classes) + 1
    img_mean = checkpoint['mean']
    img_std = checkpoint['std']
    
    # Prepare dataset and dataloader
    img_names = [i for i in os.listdir(args.root) 
            if i.endswith(('.JPG','.jpg','.JPEG','.jpeg'))]
    n = len(img_names)
    df = pandas.DataFrame(data={'images': img_names, 'x': [0]*n, 'y': [0]*n, 'labels': [1]*n})
    
    end_transforms = []
    if args.rot != 0:
        end_transforms.append(Rotate90(k=args.rot))
    end_transforms.append(DownSample(down_ratio = 2, anno_type = 'point'))
    
    albu_transforms = [A.Normalize(mean=img_mean, std=img_std)]
    
    dataset = CSVDataset(
        csv_file = df,
        root_dir = args.root,
        albu_transforms = albu_transforms,
        end_transforms = end_transforms
        )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
        sampler=torch.utils.data.SequentialSampler(dataset))
    
    # Build the trained model
    print('Building the model ...')
    device = torch.device(args.device)
    model_arch = HerdNetModel(num_classes=num_classes, pretrained=False)
    model = HerdNet(weights=args.pth, device=device, model=model_arch)
    model = LossWrapper(model, [])
    #model.load_state_dict(checkpoint['model_state_dict']) # TODO: This is now done in init, check where to do it.

    # Build the evaluator
    stitcher = HerdNetStitcher(
            model = model,
            size = (args.size,args.size),
            overlap = args.over,
            down_ratio = 2,
            up = True, 
            reduction = 'mean',
            device_name = device
            ) 

    metrics = PointsMetrics(5, num_classes = num_classes)
    evaluator = HerdNetEvaluator(
        model = model,
        dataloader = dataloader,
        metrics = metrics,
        lmds_kwargs = dict(kernel_size=(3,3), adapt_ts=0.2, neg_ts=0.1),
        device_name = device,
        print_freq = args.pf,
        stitcher = stitcher,
        work_dir=dest,
        header = '[INFERENCE]'
        )

    # Start inference
    print('Starting inference ...')
    out = evaluator.evaluate(wandb_flag=False, viz=False, log_meters=False)

    # Save the detections
    print('Saving the detections ...')
    detections = evaluator.detections
    detections.dropna(inplace=True)
    detections['species'] = detections['labels'].map(classes)
    detections.to_csv(os.path.join(dest, f'{curr_date}_detections.csv'), index=False)

    # Draw detections on images and create thumbnails
    print('Exporting plots and thumbnails ...')
    dest_plots = os.path.join(dest, 'plots')
    mkdir(dest_plots)
    dest_thumb = os.path.join(dest, 'thumbnails')
    mkdir(dest_thumb)
    img_names = numpy.unique(detections['images'].values).tolist()
    for img_name in img_names:
        img = Image.open(os.path.join(args.root, img_name))
        if args.rot != 0:
            rot = args.rot * 90
            img = img.rotate(rot, expand=True)
        img_cpy = img.copy()
        pts = list(detections[detections['images']==img_name][['y','x']].to_records(index=False))
        pts = [(y, x) for y, x in pts]
        output = draw_points(img, pts, color='red', size=10)
        output.save(os.path.join(dest_plots, img_name), quality=95)

        # Create and export thumbnails
        sp_score = list(detections[detections['images']==img_name][['species','scores']].to_records(index=False))
        for i, ((y, x), (sp, score)) in enumerate(zip(pts, sp_score)):
            off = args.ts//2
            coords = (x - off, y - off, x + off, y + off)
            thumbnail = img_cpy.crop(coords)
            score = round(score * 100, 0)
            thumbnail = draw_text(thumbnail, f"{sp} | {score}%", position=(10,5), font_size=int(0.08*args.ts))
            thumbnail.save(os.path.join(dest_thumb, img_name[:-4] + f'_{i}.JPG'))

if __name__ == '__main__':
    main()