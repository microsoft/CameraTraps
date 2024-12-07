# Classification fine-tuning

This repository focuses on training classification models for Pytorch-Wildlife. This module is designed to help both programmers and biologists train a classification model for animal identification. The output weights of this training codebase can be easily integrated in our [Pytorch-Wildlife](https://github.com/microsoft/CameraTraps/) framework. Our goal is to make this tool accessible and easy to use, regardless of your background in machine learning or programming.

## Installation

Before you start, ensure you have Python installed on your machine. This project is developed using Python 3.9. If you're not sure how to install Python, you can find detailed instructions on the [official Python website](https://www.python.org/).

To install the required libraries and dependencies, follow these steps:

1. Make sure you are in the PW_FT_classification directory.

2. Install the required packages

### Using pip and `requirements.txt`

   ```bash
   pip install -r requirements.txt
   ```

### Using conda and `environment.yaml`

  Create and activate a Conda environment with Python 3.8:

   ```bash
   conda env create -f environment.yaml
   conda activate PT_Finetuning
   ```

## Data Preparation

### Data Structure

This codebase has been optimized to facilitate its use for non-technical users. To ensure the code works correctly, your images should be stored in a single directory with no nested directories. The `annotations.csv` file, containing image paths and their classification IDs and labels, should be placed outside of the images directory. Image paths in the CSV should be relative to the position of the `annotations.csv` file.

Example directory structure:

```plaintext
PW_FT_classification/
│
├── data/
│   ├── imgs/            # All images stored here
│   └── annotation_example.csv # Annotations file
│
└── configs/config.yaml            # Configuration file
```

### Annotation file structure

To ensure the code works correctly, your annotation file should contain the following columns:

1. path: relative path to the image file
2. classification: unique identifier for each class (e.g., 0, 1, 2, etc.)
3. label: name of the class (e.g., "cat", "dog", "bird", etc.)

### Data splitting

If you want to split your data into training, validation, and test sets, you can use the `split_path` and `split_data` parameters in the `config.yaml` file. This `split_path` should point to a CSV file containing the image paths and their corresponding classification IDs and labels, while the `split_data` parameter should be set to `True`.

Currently, pytorch-wildlife classification supports three types of data splitting: `random`, `location`, and `sequence`. Random splitting uses the class ID to randomly split the data into training, validation, and test sets while keeping a balanced class distribution. **Due to the nature of camera trap images, it is common to capture a burst of pictures when movement is detected. For this reason, using random splitting is not recommended. This is because similar-looking images of the same animal could end up in both training and validation sets, leading to overfitting.**

Location splitting requires an additional "Location" column in the data, and it splits the data based on the location of the images, making sure that all images from one location will be in a single split; this splitting method does not guarantee a balanced class distribution. Finally, sequence splitting requires a "Photo_time" column containing the shooting time of the picture, it should be in YYYY-MM-DD HH:MM:SS format. This method will group images within a 30 second period in a "sequence", and then split the data based on these sequences; this splitting method does not guarantee a balanced class distribution.

The CSV file should have the previously mentioned structure. The code will then split the data into training, validation, and test sets based on the proportions specified in the `config.py` file and the splitting type. [The annotation example](data/imgs/annotation_example.csv) shows how files should be annotated for each type of splitting.

If you don't require data splitting, you can set the `split_data` parameter to `False` in the `config.yaml` file.

## Configuration

Before training your model, you need to configure the training and data parameters in the `config.yaml` file. Here's a brief explanation of the parameters to help both technical and non-technical users understand their purposes:

- **Training Parameters:**
  - `conf_id`: A unique identifier for your training configuration.
  - `algorithm`: The training algorithm to use. Default is "Plain".
  - `log_dir`: Directory where training logs are saved.
  - `num_epochs`: Total number of training epochs.
  - `log_interval`: How often to log training information.
  - `parallel`: Set to 1 to enable parallel computing (if supported by your hardware).

- **Data Parameters:**
  - `dataset_root`: The root directory where your images are stored.
  - `dataset_name`: Name of the dataset. Custom_Crop is required for finetuning.
  - `annotation_dir`: Directory where annotation files are located.
  - `split_path`: Path to the single CSV file containing the annotations, it will be used for data splitting.
  - `test_size`: Proportion of data to use as test set.
  - `val_size`: Proportion of data to use as validation set.
  - `split_data`: Set to True if you want the code to split your data into training, validation, and test sets using the `split_path`.
  - `split_type`: Type of data splitting, it can be "random", "location" or "sequence".
  - `batch_size`: Number of images to process in a batch.
  - `num_workers`: Number of subprocesses to use for data loading.

- **Model Parameters:**
  - `num_classes`: The number of classes.
  - `model_name`: The name of the model architecture to use. The current version only supports PlainResNetClassifier.
  - `num_layers`: Number of layers in the resnet model. Currently only supports 18 and 50.
  - `weights_init`: Initial weights setting for the model. Currently only supports "ImageNet".

- **Optimization Parameters:**
  - `lr_feature`, `momentum_feature`, `weight_decay_feature`: Learning rate, momentum, and weight decay for feature extractor.
  - `lr_classifier`, `momentum_classifier`, `weight_decay_classifier`: Learning rate, momentum, and weight decay for classifier.
  - `step_size`, `gamma`: Parameters for learning rate scheduler.

## Usage

> [!NOTE]
> Currently, our support is limited to the ResNet architecture. You are encouraged to explore other architectures, but it's important to maintain consistency with our code structure (particularly an independent feature extractor and a classifier) for compatibility with the PyTorch-Wildlife framework

After configuring your `config.yaml` file, you can start training your model by running:

```bash
python main.py
```

This command will initiate the training process based on the parameters specified in `config.yaml`. Make sure to monitor the output for any errors or important information regarding the training progress.

**We have provided 10 example images and an annotation file in the `data` directory for code testing without needing to provide your own data.**

## Output

Once training is complete, the output weights will be saved in the `weights` directory. These weights can be used to classify new images using the [Pytorch-Wildlife](https://github.com/microsoft/CameraTraps/)

We are working on adding a feature in a future release to directly integrate the output weights with the Pytorch-Wildlife framework and the Gradio App.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Support

If you encounter any issues or have questions, please feel free to open an issue on the GitHub repository page. We aim to make this tool as accessible as possible and will gladly provide assistance.

Thank you!
