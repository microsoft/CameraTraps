#
# Script for generating a two-class dataset in COCO format for training an obscured image classifier
#
# Requires Python >= 3.6 because of the glob ** expression
#

import glob
import PIL.Image
import json
import os

OUTPUT_DIR = '/data2/idfg/obscured_dataset'

# Collect images and labels
# Labels: clean = 0, obscured = 1
images = sorted(list(glob.glob('/data2/idfg/obscured_dataset/coco-style/obscured_images/*JPG')))
images = images[:len(images)//2]
labels = [1] * len(images)
images += list(glob.glob('/data2/idfg/obscured_dataset/coco-style/sampled_clean_train_images/**/*JPG', recursive=True))
labels += [0] * (len(images) - len(labels))

j = dict(images=[], annotations=[],
        categories=[dict(id=0, name='clean', supercategory='e'),
                    dict(id=1, name='obscured', supercategory='e'),
                    dict(id=2, name='notused_a', supercategory='e'),
                    dict(id=3, name='notused_b', supercategory='e'),
                    dict(id=4, name='notused_c', supercategory='e')
                   ]
        )
for idx, im in enumerate(images):
    w, h = PIL.Image.open(im).size
    j['images'].append(dict(id=idx, file_name=im, width=w, height=h))
    j['annotations'].append(dict(id=idx, image_id=idx, category_id=labels[idx]))
with open(os.path.join(OUTPUT_DIR, 'train.json'), 'wt') as fi:
    json.dump(j, fi)

images = sorted(list(glob.glob('/data2/idfg/obscured_dataset/coco-style/obscured_images/*JPG')))
images = images[len(images)//2:]
labels = [1] * len(images)
images += list(glob.glob('/data2/idfg/obscured_dataset/coco-style/sampled_clean_test_images_large/**/*JPG', recursive=True))
labels += [0] * (len(images) - len(labels))

j = dict(images=[], annotations=[],
        categories=[dict(id=0, name='clean', supercategory='e'),
                    dict(id=1, name='obscured', supercategory='e'),
                    dict(id=2, name='notused_a', supercategory='e'),
                    dict(id=3, name='notused_b', supercategory='e'),
                    dict(id=4, name='notused_c', supercategory='e')
                   ]
        )
for idx, im in enumerate(images):
    w, h = PIL.Image.open(im).size
    j['images'].append(dict(id=idx, file_name=im, width=w, height=h))
    j['annotations'].append(dict(id=idx, image_id=idx, category_id=labels[idx]))
with open(os.path.join(OUTPUT_DIR, 'test.json'), 'wt') as fi:
    json.dump(j, fi)
