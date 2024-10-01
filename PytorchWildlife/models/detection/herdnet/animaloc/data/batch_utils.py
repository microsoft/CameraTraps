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


from torchvision.utils import make_grid
import matplotlib.pyplot as plt 
import torch
import numpy as np

# Some helper functions for batching
# ---
def cat_list(images):
    ''' Create a batch of tensor from a list of tensors '''
    batched_images = images[0].unsqueeze(0)
    for im in images[1:]:
        batched_images = torch.cat((batched_images, im.unsqueeze(0)), dim=0)

    return batched_images

def collate_fn(batch):
    ''' Collate batch samples '''
    images , targets = list(zip(*batch))
    batched_images = cat_list(images)
    return batched_images , targets

def to_xywh(bbox):
    ''' Bbox from [x_min,y_min,x_max,y_max] to [x,y,width,height] '''
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return [bbox[0],bbox[1],width,height]

def show_batch(sample_batched, denormalize=True):
    ''' Show image with annotations for a batch of samples (tuple) '''
    imgs_batch , targets_batch = sample_batched

    batch_size = len(imgs_batch)
    img_width = imgs_batch.size(3)
    grid_border_size = 2

    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    grid = make_grid(imgs_batch)
    grid = grid.numpy().transpose((1,2,0))
    grid = std * grid + mean
    grid = np.clip(grid, 0, 1)

    if denormalize is not True:
        grid = make_grid(imgs_batch)
        grid = grid.numpy().transpose((1,2,0))

    ax = plt.gca()
    ax.imshow(grid)

    for i in range(batch_size):
        bboxes = targets_batch[i]['annotations']
        for bbox in bboxes:
            adjusted_bbox = to_xywh([
                bbox[0].numpy() + i * img_width + (i + 1) * grid_border_size,
                bbox[1],
                bbox[2].numpy() + i * img_width + (i + 1) * grid_border_size,
                bbox[3]
            ])

            rect = plt.Rectangle(
                adjusted_bbox[:2], adjusted_bbox[2], adjusted_bbox[3],
                edgecolor='red', fill=False)
            
            ax.add_patch(rect)
    
    ax.set_title(f'Batch of {batch_size} images')
    ax.axis('off')