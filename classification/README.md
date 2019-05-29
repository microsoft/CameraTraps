# Species classification training

This directory contains a set of scripts for:

- Detecting animals in camera trap images with image-level annotations
- Cropping the detected animals, associating the image-level annotation with the crop
- Collecting all the cropped images as a [COCO Camera Traps](https://github.com/Microsoft/CameraTraps/blob/master/data_management/README.md#coco-cameratraps-format) dataset or as TFRecords
- Training an image classifier on the collected data using TensorFlow's slim library


## Preparing datasets

The scripts need a data set with image-level class annotations in [COCO Camera Traps](https://github.com/Microsoft/CameraTraps/blob/master/data_management/README.md#coco-cameratraps-format) format. We do not need or use bounding box annotations as the 
purpose of the scripts is to locate the animals using a detector. [This library](https://patrickwasp.com/create-your-own-coco-style-dataset/)
facilitates the creation of COCO-style data sets. 

You can check out [http://lila.science](lila.science) for example data sets. In addition to the standard format, we usually split the camera trap datasets
by locations, i.e. into training and testing locations. Hence it is advisable to have a field in your image annotation specifying the location 
as string or integer. This could look like:

    image{
      "id" : int, 
      "width" : int, 
      "height" : int, 
      "file_name" : str, 
      "location": str
    }
    
The corresponding category annotation should contain at least

    annotation{
      "id" : int, 
      "image_id" : int, 
      "category_id" : int
    }

    
## Preparing your environment

The scripts use the following libraries:
- TensorFlow
- TensorFlow object detection API
- pycocotools

All these dependencies should be satisfied if you follow the installation instructions for the [TFODAPI](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md). You can also use conda for TensorFlow and pycocotools:

     conda install tensorflow-gpu
     conda install -c conda-forge pycocotools
     
However, you still need to follow the remaining parts of the TFODAPI installation. In particular, keep in mind that you always need to add 
relevant paths to your PYTHONPATH variable using:

    # From tensorflow/models/research/
    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
    
...before running any of the scripts. 


## Animal detection and cropping

The detection, cropping, and dataset generation is done in `database_tools/make_classification_dataset.py`. You can run 
`python make_classification_dataset.py -h` for a description of all required parameters.

    usage: make_classification_dataset.py [-h]
                                          [--coco_style_output COCO_STYLE_OUTPUT]
                                          [--tfrecords_output TFRECORDS_OUTPUT]
                                          [--location_key location]
                                          [--exclude_categories EXCLUDE_CATEGORIES [EXCLUDE_CATEGORIES ...]]
                                          [--use_detection_file USE_DETECTION_FILE]
                                          [--padding_factor PADDING_FACTOR]
                                          [--test_fraction TEST_FRACTION]
                                          [--ims_per_record IMS_PER_RECORD]
                                          input_json image_dir frozen_graph

    positional arguments:
      input_json            COCO style dataset annotation
      image_dir             Root folder of the images, as used in the annotations
                            file
      frozen_graph          Frozen graph of detection network as create by
                            export_inference_graph.py of TFODAPI.

    optional arguments:
      -h, --help            show this help message and exit
      --coco_style_output COCO_STYLE_OUTPUT
                            Output directory for a dataset in COCO format.
      --tfrecords_output TFRECORDS_OUTPUT
                            Output directory for a dataset in TFRecords format.
      --location_key location
                            Key in the image-level annotations that specifies the
                            splitting criteria. Usually we split camera-trap
                            datasets by locations, i.e. training and testing
                            locations. In this case, you probably want to pass
                            something like `--location_key location`. The script
                            prints the annotation of a randomly selected image
                            which you can use for reference.
      --exclude_categories EXCLUDE_CATEGORIES [EXCLUDE_CATEGORIES ...]
                            Categories to ignore. We will not run detection on
                            images of that category and will not use them for the
                            classification dataset.
      --use_detection_file USE_DETECTION_FILE
                            Uses existing detections from a file generated by this
                            script. You can use this to continue a partially
                            processed dataset.
      --padding_factor PADDING_FACTOR
                            We will crop a tight square box around the animal
                            enlarged by this factor. Default is 1.3 * 1.3 = 1.69,
                            which accounts for the cropping at test time and for a
                            reasonable amount of context
      --test_fraction TEST_FRACTION
                            Proportion of the locations used for testing, should
                            be in [0,1]. Default: 0.2
      --ims_per_record IMS_PER_RECORD
                            Number of images to store in each tfrecord file

                            
A typical command will look like:
    python make_classification_dataset.py \
            /path/to/dataset.json \
            /path/to/image/root/ \
            /path/to/frozen/detection/graph.pb \
            --coco_style_output /path/to/cocostyle/output/ \
            --tfrecords_output /path/to/tfrecords/output/ \
            --location_key location \
            --exclude_categories human empty

It is generally advisable to generate both the COCO-style and TFRecords output, as the former allows to check the
detection results while the latter is used for classification training. The COCO-style output folder will also contain a 
file called `detections_final.pkl`, which will be used to store the complete detection output of all images. This file 
can be used as input to the `make_classification_dataset.py` script, which makes sense if you added new images to the 
dataset and want to re-use all the detection you have already. Images without any entry in the `detections_final.pkl`
file will be analyzed using the detector. 

The script will only add images to the output folders, if they:
- exist in the images folder and can be opened
- have at least one detection with confidence 0.5 or above
- are annotated with exactly one class label (the COCO annotation format allows multiple class labels per image)
- do not exist yet in the output folders (this can happen if you re-run the script with a `detections_final.pkl` file as show above
All other images will be ignored without warning. 

The default padding factor is fairly large and optimized for images with only one animal inside and TF-slim based classification. 
You might need to adjust it according to the type of data, but keep in mind that the script currently ignores all images 
with two or more detections. 

## Dataset statistics
The file `database_tools/cropped_camera_trap_dataset_statistics.py` can be used to get some statistics about the generated
datasets, in particular the number of images and classes. This information will be required later on. The input is 
the original json file of the camera-trap dataset as well as the `train.json` and `test.json` files, which are located
in the generated COCO-style output folder. 

The usage of the script is as follows:

    usage: Tools for getting dataset statistics. It is written for datasets generated with the make_classification_dataset.py script.
           [-h] [--classlist_output CLASSLIST_OUTPUT]
           [--location_key LOCATION_KEY]
           camera_trap_json train_json test_json

    positional arguments:
      camera_trap_json      Path to json file of the camera trap dataset from
                            LILA.
      train_json            Path to train.json generated by the
                            make_classification_dataset.py script
      test_json             Path to test.json generated by the
                            make_classification_dataset.py script

    optional arguments:
      -h, --help            show this help message and exit
      --classlist_output CLASSLIST_OUTPUT
                            Generates the list of classes that corresponds to the
                            outputs of a network trained with the train.json file
      --location_key LOCATION_KEY
                            Key in the camera trap json specifying the location
                            which was used for splitting the dataset.

This prints all statistics to stdout. You can save the output by redirecting it to a file:
    python cropped_camera_trap_dataset_statistics.py \
        /path/to/dataset.json \
        /path/to/cocostyle/output/train.json \
        /path/to/cocostyle/output/test.json \
        > stats.txt

It is also useful to save the list of classes, which allows for associating the output of the classification CNN later with
the classes. You can generate this class list by using the `--classlist_output` parameter.

Note: line 31 of `cropped_camera_trap_dataset_statistics.py` might need some adjustments depending on the dataset you
are using. In this line, we collect the list of all locations by getting the COCO-style annotaion for each image that we find
in `train.json` and `test.json`. For each image, we hence have to convert the field `file_name` of `train.json`/`test.json` 
to the corresponding key used in the COCO-style annotations.

## Classification training
Once the TFRecords output is generated by `make_classification_dataset.py`, we can prepare the classification training.
Unfortunately, Tensorflow slim requires code adjustments for every new dataset you want to use. Go to the folder
`classification/tf-slim/datasets/` and copy one of the existing camera-trap dataset descriptors, for example `wellington.py`. 
We will call the copied file `newdataset.py` and place it in the same folder. The only lines that need adjustment
are the ones specifying the number of training and testing images as well as the number of classes, i.e. line 20 and 22.
These lines look in `wellington.py` like 

    SPLITS_TO_SIZES = {'train': 112698, 'test': 24734}

    _NUM_CLASSES = 17

and should be adjusted to the new dataset. If you use the output of the script presented in the previous section, then
you want to use the total number of classes, not the number of non-empty classes. 

The second step is connecting the newly generated `newdataset.py` with the Tensorflow slim code. This is done by modifying
`classification/tf-slim/datasets/dataset_factory.py`. You first need to add an import statement `import newdataset` to the top of 
the file. Afterward, add an additional dictionary entry to `datasets_map` in line 29. Afterward, it should look similar 
to 

    datasets_map = {
        'cifar10': cifar10,
        'flowers': flowers,
        'imagenet': imagenet,
        'mnist': mnist,
        'cct': cct,
        'wellington': wellington,
        'new_dataset': new_dataset # This is the newly added line
    }

This concludes the code modifications. The training can be now started using the `train_image_classifier.py` file or one
of the scripts. The easiest way to get started is by copying one of the bash scripts, e.g. `train_well_inception_v4.sh`,
and name the copy according to your dataset, e.g. `train_newdataset_inception_v4.sh`. Now open the script and adjust all
the variables at the top. In particular, 

- Assign `DATASET_NAME` the name of the dataset as used in `classification/datasets/dataset_factory.py`, we called it 
`new_dataset`in this example.
- Set `DATASET_DIR` to the TFRecords directory created above (we named it `/path/to/tfrecords/output/` in the example)
- Set `TRAIN_DIR` to the log output directory you wish to use. Folders will be created automatically
- Assign `CHECKPOINT_PATH` the path to the pre-trained Inception V4 model. It is available at
`http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz`

Now you are ready to run the training. Go to `classification/tf-slim/` and execute your script with `bash ../train_newdataset_inception_v4`. 
The provided script trains first only the last layer, executes one evaluation run, then fine-tunes the whole
network, and then runs evaluation again. If everything goes well, the final top-1 and top-5 accuracy should be reported
at the end. 

NOTE: It appears that there is a bug in the tf-slim code, which manifests in a significantly lower accuracy, e.g., 10% lower than expected. If you experiences issues with low accuracy values, try the following fix. After the training is finished, locate the created log directory, change the variable `TRAIN_DIR` in `train_serengeti_inception_v4.sh` to this folder. Now re-run `bash ../train_serengeti_inception_v4.sh`. The tensorflow training will recognize the existing checkpoints, read the model, and write out a new bug-free model.

## Remarks and advanced adjustments of training parameters
The parameter `NUM_GPUS` in the training script is currently not used. The batch size and learning rates are optimized
for the Inception V4 architecture and should give good results without any change. However, you might need to adjust the
number of steps, i.e. `--max_number_of_steps=`. One step processed one batch of images, i.e. by default 32 images. 
Divide the number of images in your dataset by the batch size and you will obtain the number of steps required for one 
epoch, i.e. one pass over the complete training set. While it is enough to train the last layer only one or a few epochs,
fine-tuning the whole network should be done for at least 10 epochs, the more challenging and the larger the dataset,
the longer. 

























