# Visualization tools

This directory contains some stand-alone Python utility scripts that you can use to visualize various incoming and predicted labels on images.


## Environment setup

Please see the Installation section in the main [README](../README.md#installation) for instructions on setting up the environment to run these scripts in.


## Visualize detector output

`visualize_detector_output.py` draws the bounding boxes, their confidence level, and predicted category annotated on top of the original images, and saves the annotated images to another directory. The original images can be in a local directory or in Azure Blob Storage.

If you are not running this on the computer with the original images, the script can download them from Azure Blob Storage using a SAS key to the container (supplied as the `--images-dir` argument with the `--is-azure` flag, *please surround the SAS URL by double quotes*). It takes about 1.5 seconds per image, depending on your location and network speed. The SAS key looks like

```
https://storageaccountname.blob.core.windows.net/container-name?se=2019-04-06T23%3A38%3A00Z&sp=rl&sv=2018-03-28&sr=c&sig=A_LONG_STRING
```

You can choose to render a sample of `n` images by supplying the `--sample` argument.

Please run `python visualize_detector_output.py -h` to see the arguments that the script takes.


### Example invocations

It is best to call the script from the root dir of this repo so the path to the repo is on the `PYTHONPATH`.

Example invocation of the script, images stored locally:
```bash
python visualize_detector_output.py path_to/requestID_detections.json rendered_images_dir --confidence 0.9 --images_dir path_to_root_dir_of_original_images
```

Another example, for images stored in Azure Blob Storage and drawing a sample of 20 images:
```bash
python visualize_detector_output.py path_to/requestID_detections.json rendered_images_dir --confidence 0.9 --images-dir "https://storageaccountname.blob.core.windows.net/container-name?se=2019-04-06T23%3A38%3A00Z&sp=rl&sv=2018-03-28&sr=c&sig=A_LONG_STRING" --is-azure --sample 20
```

If you encounter an error where it complains about not finding the module `visualization_utils`, you need to append the absolute path to the current directory to your `PYTHONPATH`. At your terminal or command line:

```bash
export PYTHONPATH=$PYTHONPATH:/absolute_path/CameraTraps
```
