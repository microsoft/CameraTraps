# Detection fine-tuning

This repository focuses on training detection models for Pytorch-Wildlife based on ultralytics framework. This module is designed to help both programmers and biologists train a detection model for animal identification. The output weights of this training codebase can be easily integrated in our [Pytorch-Wildlife](https://github.com/microsoft/CameraTraps/) framework. Our goal is to make this tool accessible and easy to use, regardless of your background in machine learning or programming.

## Installation

Before you start, ensure you have Python installed on your machine. This project is developed using Python 3.10. If you're not sure how to install Python, you can find detailed instructions on the [official Python website](https://www.python.org/).

To install the required libraries and dependencies, follow these steps:

1. Make sure you are in the PW_FT_detection directory.

2. Install the required packages

### Using pip and `requirements.txt`

   ```bash
   pip install -r requirements.txt
   ```

### Using conda and `environment.yaml`

  Create and activate a Conda environment with Python 3.10:

   ```bash
   conda env create -f environment.yaml
   conda activate PW_Finetuning_Detection
   ```

## Data Preparation

### Data Structure

The codebase is optimized for ease of use by non-technical users. For the code to function correctly, data should be organized as follows. Inside data/, create a subfolder your_data/ including two subfolders: images/ and labels/. The images/ folder should have three subfolders named test/, train/, and val/ for storing test, training, and validation images respectively. Similarly, the labels/ folder should have subfolders test/, train/, and val/ for the corresponding annotation files in txt format. Place a configuration file named your_data.yaml file within the data/ folder, alongside the your_data/ folder.

Example directory structure:

```plaintext
PW_FT_detection/
│
└── data/
   ├── your data/
   |    ├── images/
   |    |    ├── test/
   |    │    ├── train/
   |    │    └── val/
   |    └── labels/ 
   |         ├── test/
   |         ├── train/
   |         └── val/
   └── your_data.yaml

```
`./data/data_example` folder shows an example of the structure.

### Data configuration file structure

To ensure the code works correctly, your_data.yaml file should be structured as follows. The path field should contain the relative path to your_data/ folder. The train, val, and test fields should point to the subdirectories within the images/ folder where the training, validation, and test images are stored, respectively. For the classes, the names field should be a list where each class is assigned to a unique identifier number. 

The `.data/data_example.yaml` file shows an example of the structure.

### Annotations structure

The .txt files inside each folder of `./data/labels/` must be structured containing each object on a separate line, following the format: class x_center y_center width height. The coordinates for the bounding box should be normalized in the xywh format, with values ranging from 0 to 1.

## Detection models available for Finetuning

Below you find the models that you can use for fine-tuning, along with their respective names to use in the configuration file.

|Model|Name|License|
|---|---|---|
|MegaDetectroV6-Ultralytics-YoloV9-Compact|MDV6-yolov9-c|AGPL-3.0|
|MegaDetectorV6-Ultralytics-YoloV9-Extra|MDV6-yolov9-e|AGPL-3.0|
|MegaDetectorV6-Ultralytics-YoloV10-Compact|MDV6-yolov10-c|AGPL-3.0|
|MegaDetectorV6-Ultralytics-YoloV10-Extra|MDV6-yolov10-e|AGPL-3.0|
|MegaDetectorV6-Ultralytics-RtDetr-Compact|MDV6-rtdetr-c|AGPL-3.0|

## Configuration

Before training your model, you need to configure the training and data parameters in the `config.yaml` file. Here's a brief explanation of the parameters to help both technical and non-technical users understand their purposes:

- **General Parameters:**  
  - `model`: The type of model used (YOLO or RTDETR). Default: YOLO  
  - `model_name`: The name of the model [see available names here](#detection-models-available-for-finetuning). Default: MDV6-yolov9-e.pt  
  - `data`: Path to the dataset configuration file. Default: ./data/data_example.yaml  
  - `test_data`: Path to the test data directory. Default: ./data/data_example/images/test  
  - `task`: The task to perform (train, validation or inference). Default: train  
  - `exp_name`: The name of the experiment. Default: MDV6-yolov9e  
  
- **Training Parameters:**  
  - `epochs`: The total number of training epochs. Default: 20  
  - `batch_size_train`: The batch size for training. Default: 16  
  - `imgsz`: The image size. Default: 640  
  - `device_train`: The device ID for training. Default: 0  
  - `workers`: The number of workers. Default: 8  
  - `optimizer`: The optimizer to use. Default: auto  
  - `lr0`: The initial learning rate. Default: 0.01  
  - `patience`: The number of epochs to wait before stopping without improvement. Default: 5  
  - `save_period`: The period for saving the model. Default: 1  
  - `val`: Boolean value indicating whether to perform validation while training. Default: True  
  - `resume`: Boolean value indicating whether to resume training from weights. Default: False  
  - `weights`: Path to the weights file to resume training. Default: None
  
- **Validation Parameters:**  
  - `save_json`: Boolean value indicating whether to save results as JSON. Default: True  
  - `plot`: Boolean value indicating whether to plot results. Default: True 
  - `device_val`: The device ID for validation. Default: 0  
  - `batch_size_val`: The batch size for validation. Default: 12 

## Usage

After configuring your `config.yaml` file, you can start training your model by running:

```bash
python main.py
```

This command will initiate the training process based on the parameters specified in `config.yaml`. Make sure to monitor the output for any errors or important information regarding the training progress.

**We have provided example images and annotation files in the `data/data_example` directory for code testing without needing to provide your own data.**

## Output

Once training is complete, the output weights will be saved in the `./runs/detect` directory according to the experiment name you configured in the `config.yaml`. These weights can be used to classify new images using [Pytorch-Wildlife](https://github.com/microsoft/CameraTraps/).

We are working on adding a feature in a future release to directly integrate the output weights with the Pytorch-Wildlife framework and the Gradio App.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Support

If you encounter any issues or have questions, please feel free to open an issue on the GitHub repository page. We aim to make this tool as accessible as possible and will gladly provide assistance.

Thank you!
