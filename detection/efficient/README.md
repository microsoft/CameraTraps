### Commit histories

    1st commit - Folder structure setup i.e submodule added and empty files created.
    2nd commit - Environment called `efficiendet` is setup with required packages and exported as `environment-efficient.yml`. Ran training and inference script with a sample dataset. Used `Python-3.7` in the environment. Works fine with `Python-3.6` too. PyCocoTools had issues with Python 3.8 and numpy.
    3rd commit - Updating submodule `Yet-Another-EfficientDet-Pytorch` to latest.

    Submodule 1st commit - Scripts for camera trap are added. `convert_annotations.py` will convert the whole `bboxes_inc_empty_20200325.json` to COCO format json file, whereas `create_jsonsplits.py` will create `annotations/train.json` etc.  Project yml configuration is setup for training.


    4th commit - Training done on the converted cameratrap dataset. Changes involve changing `bbox` range from `0to1` to that of `pixel coordinates`(by multiplying with width and height). Scripts for COCO camera trap format conversion are added. `submodule/datasets/camtrap/convert_annotations.py` will convert the whole `bboxes_inc_empty_20200325.json` to COCO format json file, whereas `submodule/datasets/camtrap/create_jsonsplits.py` will create `annotations/train.json` etc.  Project `submodule/projects/camtrap.yml` configuration file is setup for training.

    Submodule 2nd commit - Training works on the dataset without any errors. When tried on 1600 images, loss came down i.e model converged. Image visualization added to the Tensorboard. Restriction on Image visualization is that the number of images visualized will be equal to the batch size. This is done to minimize the code changes. `Utils/tensorboard_logger.py` is an extra file. Noticed that training speed is a little faster when image visualization is done from this file than from `train.py`. Maybe related to other issues of my laptop. Using `train.py` for visualizing images for minimal code changes. However, images are flipped and are looking ugly. Need to look into it. Also need to check the `anchor scales&ratios, mean&std` of the dataset. Currently, using the stats of COCO dataset.

    5th commit - Submodule updated to its latest commit (submodule 2nd commit). Minor README changes.

    Submodule 3rd commit - Pulled `Yet-Another-Efficientdet` repo, which updated utils/utils.py and `efficientdet/` folder.
    
    6th commit - Submodule updated to its latest commit (submodule 3rd commit). Minor README changes `environment-efficient.yml` updated with `webcolors` library.

    Submodule 4th commit - COCO evaluation metric mAP added to training script. Currently, prints the mAP scores in the terminal. Will have to log them in TensorBoard. `Efficientdet/dataset.py` is changed to accomodate evaluation in training, mainly including image id, aspect ratios in dataloader.

    7th commit - Submodule updated to its latest commit (submodule 4th commit). Minor README changes and Numpy is downgraded to 1.17.0 to avoid `pycocotools` error. `environment-efficient.yml` is updated accordingly.

    Submodule 5th commit - Validation metric is calculated for each category and added to Tensorboard. In utils/utils.py, replace nms with batched_nms to further improve mAP by 0.5~0.7. Updated changes in efficientnet/utils_extra.py: fix static padding. utils_extra is not used but still updated.

    8th commit - Submodule updated to its latest commit (submodule 5th commit).

    9th commit - Use HTTPS instead of SSH for submodule. Updated this in .gitmodules

    Submodule 6th commit - Limiting the Validation metrics to `Average Precision  (AP) @[ IoU = 0.50      | area =    all | maxDets = 100 ]`. 

    10th commit - Submodule updated to its latest commit (submodule 6th commit).

#### install requirements
    pip install numpy Cython
    pip install pycocotools opencv-python tqdm tensorboard tensorboardX pyyaml matplotlib webcolors
    pip install torch==1.4.0
    pip install torchvision==0.5.0

OR
    conda install gxx_linux-64 # To avoid Pycoco installation errors.
    conda env create --file environment-efficient.yml

#### download and unzip dataset
    mkdir datasets
    wget https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.1/dataset_shape.tar.gz -O datasets/dataset_shape.tar.gz
    tar xzf datasets/dataset_shape.tar.gz

#### If you are using MDv4 camera trap data, prepare coco-camtrap dataset
    cd datasets/{projectname}/ && `python create_jsonsplits.py train`
    # `create_jsonsplits.py train` will create annotations/train.json
    # `create_jsonsplits.py val` will create annotations/val.json

#### download pretrained weights
    mkdir weights
    wget https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.0/efficientdet-d0.pth -O weights/efficientdet-d0.pth

#### prepare project file projects/{projectname}.yml
    cat projects/{projectname}.yml

#### run the training script
    python train.py -c 0 -p camtrap --batch_size 8 --lr 1e-5 --num_epochs 10 --load_weights weights/efficientdet-d0.pth --debug True

### Camera Trap progress:
1. Add the [submodule](https://github.com/gitlost-murali/Yet-Another-EfficientDet-Pytorch.git)
2. Get a small portion of the dataset from `marmot` and place it under `Yet-Another-EfficientDet-Pytorch/datasets/camtrap/`.
3. Convert the dataset to that of [COCO cameratrap format](https://github.com/Microsoft/CameraTraps/blob/master/data_management/README.md#coco-cameratraps-format). This includes changing `bbox` range from `0to1` to that of `pixel coordinates`(by multiplying with width and height).
4. Run the training script
`
python train.py -c 0 -p camtrap --batch_size 8 --lr 1e-5 --num_epochs 10 --load_weights weights/efficientdet-d0.pth
`
5. `debug` flag saves predictions on the images in `test` folder. Can modify this for visualizing images.
### ToDo: ->
6. Understand what different weights like efficientdet-d0,efficientdet-d1 stand for. Revise obj-detection concepts.
7. Focus on `anchor scales&ratios` of the dataset. Currently using COCO stats.