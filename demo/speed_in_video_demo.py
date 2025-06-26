#%% Importing necessary libraries
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import supervision as sv
import torch
from typing import Tuple, List
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification
from PytorchWildlife import utils as pw_utils

#%% Set the device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#%% Load models
detection_model = pw_detection.MegaDetectorV6(device=DEVICE, pretrained=True, version="MDV6-yolov9-c")
classification_model = pw_classification.AI4GAmazonRainforest(device=DEVICE, version='v2')

#%% Input and output paths
VIDEO_NAME = "03100028"
VIDEO_EXT = "MP4"
SOURCE_VIDEO_PATH = os.path.join(".", "demo_data", "videos", f"{VIDEO_NAME}.{VIDEO_EXT}")
OUTPUT_FOLDER = os.path.join(".", "speed_tracking_output")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
TARGET_VIDEO_PATH = os.path.join(OUTPUT_FOLDER, f"{VIDEO_NAME}_tracked.{VIDEO_EXT}")

#%% Verify if the source video exists
if not os.path.exists(SOURCE_VIDEO_PATH):
    raise FileNotFoundError(f"Source video not found at {SOURCE_VIDEO_PATH}. Please check the path.")

#%% Annotators
box_annotator = sv.BoxAnnotator(thickness=4)
lab_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_thickness=4, text_scale=2)

#%% Callback for tracking
def callback(frame: np.ndarray, index: int) -> Tuple[np.ndarray, sv.Detections, List[Tuple[str, float]]]:
    results_det = detection_model.single_image_detection(frame, img_path=index)

    clf_labels = []
    for xyxy in results_det["detections"].xyxy:
        cropped_image = sv.crop_image(image=frame, xyxy=xyxy)
        results_clf = classification_model.single_image_classification(cropped_image)
        clf_labels.append((results_clf["prediction"], results_clf["confidence"]))

    annotated_frame = lab_annotator.annotate(
        scene=box_annotator.annotate(scene=frame, detections=results_det["detections"]),
        detections=results_det["detections"],
        labels=results_det["labels"]
    )

    return annotated_frame, results_det["detections"], clf_labels

#%% Run tracking and compute 2D speed
image_width_px, t1, x1, y1, t2, x2, y2, speed_px_s = pw_utils.speed_in_video(
    source_path=SOURCE_VIDEO_PATH,
    target_path=TARGET_VIDEO_PATH,
    callback=callback,
    target_fps=10,
    codec="mp4v"
)

#%% Save CSV with the points x, y, speed
df = pd.DataFrame(columns=["Image Width (px)", "t1 (s)", "x1 (px)", "y1 (px)", "t2 (s)", "x2 (px)", "y2 (px)", "speed (px/s)"])
df.loc[0] = [image_width_px, t1, x1, y1,t2,x2,y2,speed_px_s]
csv_path = os.path.join(OUTPUT_FOLDER, f"{VIDEO_NAME}_speed.csv")
df.to_csv(csv_path, index=False, float_format="%.3f")
