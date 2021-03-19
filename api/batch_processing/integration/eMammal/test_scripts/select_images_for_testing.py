#%% Imports

import os
import sys
import argparse
import json
import shutil

from tqdm import tqdm

#%% Main function

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='Input .json filename')
   
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args() 

    with open(args.input_file) as f:
        data = json.load(f)

    images = data['images'][:1000]
    data['images'] = images
    with open('output.json', 'w') as f:
        json.dump(data, f,  indent=1)

    select_images_folder = "select_images_folder"

    if os.path.exists(select_images_folder):
        shutil.rmtree(select_images_folder, ignore_errors=True)

    os.mkdir(select_images_folder)

    source = os.getcwd() + "\\SWWLF2019_R1_GMU1_F_9\\"
    destination = os.getcwd() + "\\" + select_images_folder + "\\"

    # if not os.path.exists(directory):
    #   os.makedirs(directory)

    for index, im in tqdm(enumerate(images), total=len(images)):
        fn=im['file']
        fn = fn.replace('/', '\\')
        print(os.path.dirname(fn))
        directory = destination + os.path.dirname(fn)
        if not os.path.exists(directory):
            os.makedirs(directory)
        dest = shutil.copyfile(source + fn, destination + fn)         

if __name__ == '__main__':
    main()
