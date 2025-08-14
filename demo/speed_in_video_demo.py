#%% Importing necessary libraries
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import supervision as sv
import torch
from tqdm import tqdm
from typing import Tuple, List
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification
from PytorchWildlife import utils as pw_utils

#%% Set the animal height in meters
animal_height_m = 1.5
if animal_height_m:
    cols = ["Video", "Image Width (px)", "t1 (s)", "x1 (px)", "y1 (px)", "t2 (s)", "x2 (px)", "y2 (px)", "speed (m/s)"]
    print(f"Using height ~ {animal_height_m} m for conversion.")
    df = pd.DataFrame(columns=cols)
    using_meters = True
else:
    cols = ["Video", "Image Width (px)", "t1 (s)", "x1 (px)", "y1 (px)", "t2 (s)", "x2 (px)", "y2 (px)", "speed (px/s)"]
    print("No animal height specified. Speed will be in pixels/second.")
    df = pd.DataFrame(columns=cols)
    using_meters = False

#%% Set whether to assume a single individual (True) or a group (False) in the video
assume_single_individual = False

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

#%% Run tracking and compute 2D speed for each video in the source folder
tracks = 0
video_files = [f for f in os.listdir(SOURCE_FOLDER_PATH) if f.lower().endswith((".mp4", ".avi", ".mov"))]

if not video_files:
    print("No video files found. Add videos to the source folder and re-run this cell.")
else:
    iterator = video_files
    if tqdm is not None:
        iterator = tqdm(video_files, desc="Processing videos")

    for video_name in iterator:
        SOURCE_VIDEO_PATH = os.path.join(SOURCE_FOLDER_PATH, video_name)
        TARGET_VIDEO_PATH = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(video_name)[0]}_tracked.mp4")
        print(f"\nProcessing: {video_name}")

        try:
            image_width_px, track_summaries = pw_utils.speed_in_video(
                source_path=SOURCE_VIDEO_PATH,
                target_path=TARGET_VIDEO_PATH,
                callback=callback,
                target_fps=10,
                codec="mp4v",
                longest=assume_single_individual,
                min_points=6,
                min_duration_s=0.5,
                min_displacement_px=20,
                suppress_subtracks=True,
                subtrack_radius_px=50,
            )

            # Each 'track' has two points (t1,x1,y1) and (t2,x2,y2) and a speed in px/s
            for i, key in enumerate(track_summaries):
                t1, x1, y1 = track_summaries[key]['points'][0]
                t2, x2, y2 = track_summaries[key]['points'][1]
                speed_px_s = track_summaries[key]['speed']

                if using_meters and animal_height_m:
                    # Convert px/s to m/s using width-scale (height_m / image_width_px)
                    speed_val = (speed_px_s * animal_height_m) / image_width_px
                else:
                    speed_val = speed_px_s

                df.loc[tracks] = [video_name, image_width_px, t1, x1, y1, t2, x2, y2, speed_val]
                tracks += 1

        except Exception as e:
            print(f"⚠️ Error processing {video_name}: {e}")
            continue

#%% Save CSV with the points x, y, speed
csv_path = os.path.join(OUTPUT_FOLDER, "speed.csv")
if len(df) > 0:
    df.to_csv(csv_path, index=False, float_format="%.3f")
    print(f"Saved: {csv_path}")
else:
    print("Speed table is empty—nothing to save yet.")
