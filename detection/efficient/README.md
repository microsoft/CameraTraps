### 1. Install requirements
    pip install numpy Cython
    pip install pycocotools opencv-python tqdm tensorboard tensorboardX pyyaml matplotlib webcolors
    pip install torch==1.4.0
    pip install torchvision==0.5.0

OR

    conda env create --file environment-efficient.yml
    conda activate efficientdet
    conda install gxx_linux-64 # To avoid Pycoco installation errors.
    pip install pycocotools

### 2. Working directory and submodule
1.  Keep your working directory as `CamerTraps/detection/efficient`.

2. Add the [submodule](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch.git) by `git submodule update --init --recursive`

### 3. Download and unzip dataset
    mkdir datasets
    #sample dataset
    wget https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.1/dataset_shape.tar.gz -O datasets/dataset_shape.tar.gz
    tar xzf datasets/dataset_shape.tar.gz

### 4. If you are using MDv4 Cameratrap data, prepare a coco-camtrap format dataset
1. Download the annotation file from the server. -> `wget http://storagesite/bboxes_inc_empty_20200325.json -O datasets/{projectname}/`.
    
2. Make sure your images are placed in datasets/{projectname}/{train/test/val}

3. Create COCO-format train/test/val split annotation files. `preprocessing/create_jsonsplits.py` converts the dataset to that of [COCO cameratrap format](https://github.com/Microsoft/CameraTraps/blob/master/data_management/README.md#coco-cameratraps-format). This includes changing `bbox` range from `0to1` to that of `pixel coordinates`(by multiplying with width and height).

    `python preprocessing/create_jsonsplits.py --file datasets/camtrap/bboxes_inc_empty_20200325.json --split train`

     `--split train` will create annotations/instances_train.json
     `--split val` will create annotations/instances_val.json

### 5. Download pretrained weights
    mkdir weights
    wget https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.0/efficientdet-d0.pth -O weights/efficientdet-d0.pth

### 6. Prepare project file projects/{projectname}.yml
    cat projects/{projectname}.yml

### 7. Run the training script

    python train.py -c 0 -p camtrap --batch_size 4 --lr 1e-5 --num_epochs 10  --load_weights weights/efficientdet-d0.pth --data_path datasets/


#### Commit histories

    22nd commit - Documentation: `train.py` & `utils_/calc_eval.py` : Added references & cleaned the files. 

    21st commit - Added `plot_images` function in `train.py`.

    20th commit - `environment-efficient.yml` is cleaned.

    19th commit - Wrapped `train.py` code into a class. Made separate functions for train and validate. Added Python `typing` hints for functions. README updated with instructions.

    18th commit - Project structure changed. Submodule only as dependency. Training scripts added to CameraTrap/.

    17th commit - Submodule updated to its latest commit (Submodule 13th commit).

    Submodule 13th commit - Validation related errors are solved. Previous commit of skipping negative values is wrong. Corrected it. Changes made in `datasets/camtrap/create_jsonsplits.py`.
    
    16th commit - Submodule updated to its latest commit (Submodule 12th commit).

    Submodule 12th commit - In `utils/calc_eval.py`, we skip the instances where there is a negative bbox prediction. Minor changes in `train.py`, which include changing param types from `int` to `float`.

    15th commit - Submodule updated to its latest commit (Submodule 11th commit).

    Submodule 11th commit - Added a feature to increase num of images visualized in `train.py`. Validation frequency parameters are added to `train.py`. Variable name changes to avoid confusion. `calc_eval` is updated to avoid path related issues.

    14th commit - Submodule updated to its latest commit (Submodule 10th commit).

    Submodule 10th commit - Added percentage of validation images to be validated in each epoch in `train.py`.

    13th commit - Submodule updated to its latest commit (submodule 9th commit) and conda yml file is updated along with its installation details in README.

    Submodule 9th commit - `datasets/camtrap/create_jsonsplits.py` is updated to access images from a different location.

    12th commit - Submodule updated to its latest commit (submodule 8th commit).

    Submodule 8th commit - Pylinting on some scripts: `train.py`, `utils/calc_eval.py`, `utils/utils.py`, `utils/tensorboard_logger.py`.

    11th commit - Submodule updated to its latest commit (submodule 7th commit).

    Submodule 7th commit - Since  the initial stages produce a lot of annotations, there's a computational overhead to compute mAP. So, kept a max limit on the predictions, which will be `opt.max_preds_toeval = len(val_generator)*opt.batch_size* 5`. Here, I averaged the #obj per image to 5 for computational efficacy. Tensorboard image visualization added to the evaluation dataset too. Original author updated `efficientdet/model.py` with comments.

    10th commit - Submodule updated to its latest commit (submodule 6th commit).

    Submodule 6th commit - Limiting the Validation metrics to `Average Precision  (AP) @[ IoU = 0.50      | area =    all | maxDets = 100 ]`. 

    9th commit - Use HTTPS instead of SSH for submodule. Updated this in .gitmodules

    8th commit - Submodule updated to its latest commit (submodule 5th commit).

    Submodule 5th commit - Validation metric is calculated for each category and added to Tensorboard. In utils/utils.py, replace nms with batched_nms to further improve mAP by 0.5~0.7. Updated changes in efficientnet/utils_extra.py: fix static padding. utils_extra is not used but still updated.

    7th commit - Submodule updated to its latest commit (submodule 4th commit). Minor README changes and Numpy is downgraded to 1.17.0 to avoid `pycocotools` error. `environment-efficient.yml` is updated accordingly.

    Submodule 4th commit - COCO evaluation metric mAP added to training script. Currently, prints the mAP scores in the terminal. Will have to log them in TensorBoard. `Efficientdet/dataset.py` is changed to accomodate evaluation in training, mainly including image id, aspect ratios in dataloader.

    6th commit - Submodule updated to its latest commit (submodule 3rd commit). Minor README changes `environment-efficient.yml` updated with `webcolors` library.

    Submodule 3rd commit - Pulled `Yet-Another-Efficientdet` repo, which updated utils/utils.py and `efficientdet/` folder.

    5th commit - Submodule updated to its latest commit (submodule 2nd commit). Minor README changes.

    Submodule 2nd commit - Training works on the dataset without any errors. When tried on 1600 images, loss came down i.e model converged. Image visualization added to the Tensorboard. Restriction on Image visualization is that the number of images visualized will be equal to the batch size. This is done to minimize the code changes. `Utils/tensorboard_logger.py` is an extra file. Noticed that training speed is a little faster when image visualization is done from this file than from `train.py`. Maybe related to other issues of my laptop. Using `train.py` for visualizing images for minimal code changes. However, images are flipped and are looking ugly. Need to look into it. Also need to check the `anchor scales&ratios, mean&std` of the dataset. Currently, using the stats of COCO dataset.

    4th commit - Training done on the converted cameratrap dataset. Changes involve changing `bbox` range from `0to1` to that of `pixel coordinates`(by multiplying with width and height). Scripts for COCO camera trap format conversion are added. `submodule/datasets/camtrap/convert_annotations.py` will convert the whole `bboxes_inc_empty_20200325.json` to COCO format json file, whereas `submodule/datasets/camtrap/create_jsonsplits.py` will create `annotations/train.json` etc.  Project `submodule/projects/camtrap.yml` configuration file is setup for training.

    Submodule 1st commit - Scripts for camera trap are added. `convert_annotations.py` will convert the whole `bboxes_inc_empty_20200325.json` to COCO format json file, whereas `create_jsonsplits.py` will create `annotations/train.json` etc.  Project yml configuration is setup for training.

    3rd commit - Updating submodule `Yet-Another-EfficientDet-Pytorch` to latest.

    2nd commit - Environment called `efficiendet` is setup with required packages and exported as `environment-efficient.yml`. Ran training and inference script with a sample dataset. Used `Python-3.7` in the environment. Works fine with `Python-3.6` too. PyCocoTools had issues with Python 3.8 and numpy.

    1st commit - Folder structure setup i.e submodule added and empty files created.