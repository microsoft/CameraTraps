# ruff: noqa: E402
import sys
import os

# Add the root directory to sys.path so Python can find PytorchWildlife during development
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import pandas as pd
from datasets import load_dataset
from PIL import Image
import logging
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification
import numpy as np
import requests
from io import BytesIO
import sklearn.metrics as metrics

logging.basicConfig(level=logging.INFO)


def extract_url_genus_from_record(record, taxonomy):

    url = record["image"].replace(
        "lilablobssc.blob.core.windows.net/",
        "lilawildlife.blob.core.windows.net/lila-wildlife/",
    )

    # Let's do just the first detection for now
    genus_id = record["annotations"]["taxonomy"][0]["genus"]
    genus_id = int(genus_id) if genus_id is not None else None
    
    if genus_id is not None:
        genus_name = taxonomy["genus"].int2str(genus_id).lower()
    else:
        genus_name = "none"

    return url, genus_name

def get_animal_sample(n_records:int, dataset_name:str):
    cn2tax_df = pd.read_json(
        "https://huggingface.co/datasets/society-ethics/lila_camera_traps/raw/main/data/common_names_to_tax.json",
        lines=True,
    )
    cn2tax_df = cn2tax_df.set_index("common_name")
    

    dataset = load_dataset(path="society-ethics/lila_camera_traps",
                           name=dataset_name, 
                           split="train"
    )
    
    taxonomy = dataset.features["annotations"].feature["taxonomy"]

    # get n_records random items from the dataset
    sample = dataset.shuffle().select(range(n_records))    

    df_animals = pd.DataFrame(columns=["genus", "url"])

    # create a list of dictionaries with the genus and url of each record
    rows = []

    for record in sample:
        url, genus = extract_url_genus_from_record(record, taxonomy)
        rows.append({"genus": genus, "url": url})

    df_animals = pd.DataFrame(rows)
    return df_animals

def detect_and_classify(img_url):

    # Download the image with requests
    img = requests.get(img_url)
    
    pil_img = Image.open(BytesIO(img.content))
    img_array = np.array(pil_img)

    # Detection
    detection_model = (
        pw_detection.MegaDetectorV6()
    )  # Model weights are automatically downloaded.
    detection_result = detection_model.single_image_detection(img_array)

    detection = detection_result.get("detections")
    if detection is None:
        return "none"
    
    if len(detection.xyxy) == 0:
        return "none"

    pil_img.show()
    
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
    
    # crop the image using the coordinates and convert it to a numpy array
    cls_img = pil_img.crop(first_detection_coordinates)
    cls_img = np.array(cls_img)

    classification_results = classification_model.single_image_classification(cls_img)

    genus = classification_results["prediction"]
    if isinstance(genus, str):
        genus = genus.lower()
    else:
        genus = "none"
    
    return genus


def main():
    n_samples = 30
    df_animals = get_animal_sample(n_samples, "Orinoquia Camera Traps")

    results = []
    for index, row in df_animals.iterrows():
        prediction = detect_and_classify(row["url"])
        results.append({"genus": row["genus"], "prediction": prediction})

    df_results = pd.DataFrame(results)
    accuracy = metrics.accuracy_score(df_results["genus"], df_results["prediction"])
    precision = metrics.precision_score(df_results["genus"], df_results["prediction"], average="weighted")
    recall = metrics.recall_score(df_results["genus"], df_results["prediction"], average="weighted")
    logging.info(f"Accuracy: {accuracy}")
    logging.info(f"Precision: {precision}")
    logging.info(f"Recall: {recall}")


if __name__ == "__main__":
    main()