# ruff: noqa: E402
import sys
import os

# Add the root directory to sys.path so Python can find PytorchWildlife
# from the root directory when running `uv run pytest -s tests`
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification
from PIL import Image
import numpy as np


def test_tamandua():
    # read tamandua image from the demo images
    img_path = "demo/demo_data/imgs/10050028_1.JPG"

    # Detection
    detection_model = (
        pw_detection.MegaDetectorV6()
    )  # Model weights are automatically downloaded.
    detection_result = detection_model.single_image_detection(img_path)

    first_detected_label = detection_result["labels"][0]
    assert first_detected_label.startswith("animal"), (
        "Animal not detected by MegaDetectorV6"
    )

    detection = detection_result.get("detections")  # apparently returns one detection?
    first_xyxy = detection.xyxy[0]
    first_detection_coordinates = (
        int(first_xyxy[0]),
        int(first_xyxy[1]),
        int(first_xyxy[2]),
        int(first_xyxy[3]),
    )

    assert first_detection_coordinates == (3082, 1866, 4357, 2509)

    # Classification
    classification_model = (
        pw_classification.AI4GAmazonRainforest_v2()
    )  # Model weights are automatically downloaded.

    img = Image.open(img_path)

    # crop the image using the coordinates and convert it to a numpy array
    img = img.crop(first_detection_coordinates)
    img = np.array(img)

    classification_results = classification_model.single_image_classification(img)
    assert classification_results["prediction"] == "Tamandua", (
        "Tamandua not detected by AI4GAmazonRainforest_v2"
    )
