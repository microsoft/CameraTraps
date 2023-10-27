# PyTorchWildlife: An Animal Detection and Classification Package

PyTorchWildlife is a collaborative Deep Learning Framework for conservation that provides pre-trained models for animal detection and classification. This README will guide you through the steps to run a demo of the package using the Gradio interface.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Running the Demo](#running-the-demo)
4. [License](#license)

## Prerequisites

1. Python 3.x
2. NVIDIA GPU (for CUDA support, although the demo can run on CPU)

## Installation

### 1. Install through pip:
\```bash
pip install PyTorchWildlife
\```

## Running the Demo

Once the setup is complete, execute:

\```bash
python demo_gradio.py
\```

This will launch a Gradio interface where you can:

- Perform Single Image Detection: Upload an image and set a confidence threshold to get detections.
- Perform Batch Image Detection: Upload a zip file containing multiple images to get detections in a JSON format.
- Perform Video Detection: Upload a video and get a processed video with detected animals.

## License

This project is licensed under the MIT License. Refer to the LICENSE file for more details.

## Copyright

Copyright (c) Microsoft Corporation. All rights reserved.