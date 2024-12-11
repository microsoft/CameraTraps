# ruff: noqa: E402
import sys
import os

# Add the root directory to sys.path so Python can find PytorchWildlife during development
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification
from PIL import Image
import numpy as np


def detect_and_classify(img_path):
    # Detection
    detection_model = (
        pw_detection.MegaDetectorV6()
    )  # Model weights are automatically downloaded.
    detection_result = detection_model.single_image_detection(img_path)

    detection = detection_result.get("detections")
    first_xyxy = detection.xyxy[0]
    first_detection_coordinates = (
        int(first_xyxy[0]),
        int(first_xyxy[1]),
        int(first_xyxy[2]),
        int(first_xyxy[3]),
    )

    # Classification
    classification_model = (
        pw_classification.AI4GAmazonRainforest_v2()
    )  # Model weights are automatically downloaded.
    img = Image.open(img_path)

    # crop the image using the coordinates and convert it to a numpy array
    img = img.crop(first_detection_coordinates)
    img = np.array(img)

    classification_results = classification_model.single_image_classification(img)

    return classification_results["prediction"]


def main():
    images = []
    images.append({"path": "dev/data/tamandua.jpg", "expected": "Tamandua"})
    images.append({"path": "dev/data/onca.jpg", "expected": "Leopardus"})
    images.append({"path": "dev/data/sloth.jpg", "expected": "Bradypus"})
    images.append({"path": "dev/data/tapir.jpg", "expected": "Tapirus"})
    images.append({"path": "dev/data/tamandua2.jpg", "expected": "Tamandua"})
    images.append({"path": "dev/data/capybara.jpg", "expected": "Capybara"})

    for image in images:
        prediction = detect_and_classify(image["path"])
        print(
            f"Image: {image['path']}, Expected: {image['expected']}, Prediction: {prediction}"
        )


if __name__ == "__main__":
    main()
