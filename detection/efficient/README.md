### Commit histories

    1st commit - Folder structure setup i.e submodule added and empty files created.
    2nd commit - Environment called `efficiendet` is setup with required packages and exported as `environment-efficient.yml`. Ran training and inference script with a sample dataset. Used `Python-3.7` in the environment. Works fine with `Python-3.6` too. PyCocoTools had issues with Python 3.8 and numpy.
    3rd commit - Updating submodule `Yet-Another-EfficientDet-Pytorch` to latest.
    

    Submodule 1st commit - Scripts for camera trap are added. `convert_annotations.py` will convert the whole `bboxes_inc_empty_20200325.json` to COCO format json file, whereas `create_jsonsplits.py` will create `annotations/train.json` etc.  Project yml configuration is setup for training.

#### install requirements
    pip install numpy Cython
    pip install pycocotools opencv-python tqdm tensorboard tensorboardX pyyaml matplotlib
    pip install torch==1.4.0
    pip install torchvision==0.5.0

OR

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
2. Get dataset from `marmot` and place it under `Yet-Another-EfficientDet-Pytorch/datasets/camtrap/`.
3. Convert the dataset to that of [COCO cameratrap format](https://github.com/Microsoft/CameraTraps/blob/master/data_management/README.md#coco-cameratraps-format). This includes changing `bbox` range from `0to1` to that of `pixel coordinates`(by multiplying with width and height).
4. Run the training script
`
python train.py -c 0 -p camtrap --batch_size 8 --lr 1e-5 --num_epochs 10 --load_weights weights/efficientdet-d0.pth
`
### ToDo: ->
5. Understand what different weights like efficientdet-d0,efficientdet-d1 stand for. Revise obj-detection concepts.