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

#%% Set the name of the species to track
SPECIES_NAME = None
if SPECIES_NAME:
    if SPECIES_NAME in pw_utils.REAL_ANIMAL_HEIGHT_M:
        print(f"Tracking species: {SPECIES_NAME}")
        animal_height_m = pw_utils.REAL_ANIMAL_HEIGHT_M[SPECIES_NAME]
    else:
        raise ValueError(f"Species '{SPECIES_NAME}' not found in the classification model.")
else:
    animal_height_m = None
    print("No specific species selected for tracking. Speed will be calculated in pixels per second.")

#%% Input and output paths
SOURCE_FOLDER_PATH = os.path.join(".", "demo_data", "speed_tracking_videos")
if not os.path.exists(SOURCE_FOLDER_PATH):
    raise FileNotFoundError(f"Source video not found at {SOURCE_FOLDER_PATH}. Please check the path.")

OUTPUT_FOLDER = os.path.join(".", "speed_tracking_output")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

#%% Set the device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#%% Load models
detection_model = pw_detection.MegaDetectorV6(device=DEVICE, pretrained=True, version="MDV6-yolov9-c")
classification_model = pw_classification.AI4GAmazonRainforest(device=DEVICE, version='v2')

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

#%% Initialize DataFrame to store speed data
if animal_height_m:
    print(f"Using animal height for speed calculation: {animal_height_m} m")
    df = pd.DataFrame(columns=["Video", "Image Width (px)", "t1 (s)", "x1 (px)", "y1 (px)", "t2 (s)", "x2 (px)", "y2 (px)", "speed (m/s)"])
else:
    df = pd.DataFrame(columns=["Video", "Image Width (px)", "t1 (s)", "x1 (px)", "y1 (px)", "t2 (s)", "x2 (px)", "y2 (px)", "speed (px/s)"])

#%% Run tracking and compute 2D speed for each video in the source folder
tracks = 0

for video_name in os.listdir(SOURCE_FOLDER_PATH):
    if not video_name.lower().endswith((".mp4", ".avi", ".mov")):
        continue  # Skip non-video files

    SOURCE_VIDEO_PATH = os.path.join(SOURCE_FOLDER_PATH, video_name)
    TARGET_VIDEO_PATH = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(video_name)[0]}_tracked.mp4")
    print(f"Processing video: {video_name}")

    try:
        image_width_px, track_summaries = pw_utils.speed_in_video(
            source_path=SOURCE_VIDEO_PATH,
            target_path=TARGET_VIDEO_PATH,
            callback=callback,
            target_fps=10,
            codec="mp4v"
        )
        for i, key in enumerate(track_summaries):
            t1, x1, y1 = track_summaries[key]['points'][0]
            t2, x2, y2 = track_summaries[key]['points'][1]
            speed_px_s = track_summaries[key]['speed']
            if animal_height_m:
                # Convert speed from pixels per second to meters per second
                speed = (speed_px_s * animal_height_m) / image_width_px
            else:
                speed = speed_px_s
            # Append data to DataFrame
            df.loc[tracks] = [video_name, image_width_px, t1, x1, y1, t2, x2, y2, speed]
            tracks += 1
    except Exception as e:
        print(f"Error processing video {video_name}: {e}")
        continue

#%% Save CSV with the points x, y, speed
csv_path = os.path.join(OUTPUT_FOLDER,"speed.csv")
df.to_csv(csv_path, index=False, float_format="%.3f")
