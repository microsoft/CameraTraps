# Visualization tools

This directory contains some stand-alone Python utility scripts that you can use to visualize various incoming and predicted labels on images. 


## Environment setup

If you are not very experienced in managing Python environments, we suggest that you start a conda virtual environment and use our visualization scripts within that environment. Conda is a package manager for Python (among other things).

You can install a lightweight distribution of conda (Miniconda) at https://docs.conda.io/en/latest/miniconda.html for your OS. 

At the terminal, issue this command to create a conda virtual environment (called 'cameratrap') with the required version of Python and packages:

```
conda create -n cameratrap python=3.5 pandas=0.23.4 pillow=5.3.0 azure-storage-blob=1.5.0 tqdm=4.31.1
```

When prompted, press 'y' to proceed with installing the packages and their dependencies. 

If you run into an error (e.g. 'ValueError... cannot be negative') while creating the environment, make sure to update conda to version 4.5.11 or above. Check the version of conda using `conda --version`; upgrade it using `conda update conda`. 

Once the environment is created, activate it using `conda activate cameratrap`. You can then run scripts such as `visualize_detector_output.py`.

To exit the conda environment, issue `conda deactivate cameratrap`.


## Visualize detector output

`visualize_detector_output.py` draws the bounding boxes, their confidence level and predicted category annotated on top of the original images, and saves the annotated images to another directory. The original images can be in a local directory or in Azure Blob Storage. 

Please see the top of `visualize_detector_output.py` for the arguments it requires. You can also change the size of the output image by changing the `viz_size` variable found there.

- If you are not running this on the computer with the original images, the script can download them from Azure Blob Storage using a SAS key to the container (supplied as the `--sas_url` argument). It takes about 1.5 seconds per image, depending on your location and network speed. The SAS key looks like

```
https://storageaccountname.blob.core.windows.net/container-name?se=2019-04-06T23%3A38%3A00Z&sp=rl&sv=2018-03-28&sr=c&sig=A_LONG_STRING
```

- You can choose to render a sample of `n` images by supplying the `--sample` argument.

Example invocation of the script, images stored locally:
```
python visualize_detector_output.py path_to/requestID_detections.json rendered_images_dir --confidence 0.9 --images_dir path_to_root_dir_of_original_images 
```

Another example, for images stored in the cloud and drawing a sample of 20 images:
```
python visualize_detector_output.py path_to/requestID_detections.json rendered_images_dir --confidence 0.9 --sas_url https://storageaccountname.blob.core.windows.net/container-name?se=2019-04-06T23%3A38%3A00Z&sp=rl&sv=2018-03-28&sr=c&sig=A_LONG_STRING 
```

If you encounter an error where it complains about not finding the module `visualization_utils`, you need to append the absolute path to the current directory `visualization` to your `PYTHONPATH`. At your terminal or command line:

```
export PYTHONPATH=$PYTHONPATH:/absolute_path/CameraTraps/visualization
```
