#
# helena_to_cct.py
#
# Convert the Helena Detections data set to a COCO-camera-traps .json file
#

#%% Constants and environment

import os
import json
import uuid
import time
import humanfriendly
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd

from path_utils import find_images

base_directory = r'/mnt/blobfuse/wildlifeblobssc/'
output_directory = r'/home/gramener'
output_file = os.path.join(output_directory,'rspb.json')
input_file = os.path.join(base_directory, 'StHelena_Detections.xlsx')
image_directory = os.path.join(base_directory, 'StHELENA_images')

assert(os.path.isdir(image_directory))

#%% Create Filenames and timestamps mapping CSV

image_full_paths = find_images(image_directory, bRecursive=True)
print(len(image_full_paths))
map_list = []
for img_ in image_full_paths:
    try:
        date_cr = Image.open(img_)._getexif()[306]
        date_ = date_cr.split(" ")[0]
        time_ = date_cr.split(" ")[1]
        _tmp = {}
        img_path = img_.replace(image_directory, "")
        img_folder = img_path.split("/")[1]
        _tmp["image_name"] = img_path
        _tmp["folder"] = img_folder.replace("Fortnight", "")
        _tmp["mapping_name"] = "-".join(date_.split(":")[:-1])
        map_list.append(_tmp)
    except Exception as e:
        print(img_)
mapping_df = pd.DataFrame(map_list)
mapping_df.to_csv("mapping_names.csv", index=False)
# import pdb;pdb.set_trace()

