#%% Importing necessary libraries
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import supervision as sv
import torch
from typing import Tuple, Optional

#%% Importing PytorchWildlife modules
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification
from PytorchWildlife import utils as pw_utils

#%% Set the device to CUDA (GPU) if available, otherwise fall back to CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#%% Load the MegaDetectorV6 model with pretrained weights
detection_model = pw_detection.MegaDetectorV6(device=DEVICE, pretrained=True, version="MDV6-yolov9-c")
classification_model = pw_classification.AI4GAmazonRainforest(device=DEVICE, version='v2')

#%% Define input path for the source video
VIDEO_NAME = "03140004"
VIDEO_EXT = "MP4"
SOURCE_VIDEO_PATH = os.path.join(".", "demo_data", "videos", f"{VIDEO_NAME}.{VIDEO_EXT}")

# Create a folder to store output video and plot
OUTPUT_FOLDER = os.path.join(".", "approach_speed_output")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Output video path
TARGET_VIDEO_PATH = os.path.join(OUTPUT_FOLDER, f"{VIDEO_NAME}_processed.{VIDEO_EXT}")
#%% Create annotators for visualizing boxes and labels on frames
box_annotator = sv.BoxAnnotator(thickness=4)
lab_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_thickness=4, text_scale=2)

#%% Callback function for processing each frame
# This is passed to the video processor to handle detection + annotation
def callback(frame: np.ndarray, index: int) -> Tuple[np.ndarray, bool, Optional[float]]:
    # Run object detection on the frame
    results_det = detection_model.single_image_detection(frame, img_path=index)

    heights = []
    clf_labels = []
    for i, xyxy in enumerate(results_det["detections"].xyxy):
        x1, y1, x2, y2 = xyxy
        height = y2 - y1  # Bounding box height (used as a proxy for distance)
        heights.append(height)
        
        cropped_image = sv.crop_image(image=frame, xyxy=xyxy)
        results_clf = classification_model.single_image_classification(cropped_image)
        clf_labels.append((results_clf["prediction"], results_clf["confidence"]))
    
    # Annotate the frame with boxes and labels
    annotated_frame = lab_annotator.annotate(
        scene=box_annotator.annotate(
            scene=frame,
            detections=results_det["detections"]
        ),
        detections=results_det["detections"],
        labels=results_det["labels"]
    )

    # Determine if any detections were made
    detected = len(heights) > 0

    # Use the largest box height (closest object) for speed estimation
    max_height = max(heights) if detected else None

    return annotated_frame, detected, max_height, clf_labels

#%% Process the video using the callback, and collect time + height data

# This function:
# - Runs the callback on each frame (with skipping based on target_fps)
# - Writes the annotated video to TARGET_VIDEO_PATH
# - Returns timestamps and bounding box heights where detections occurred
image_width_px, timestamps, bounding_heights, most_voted_class, best_detection = pw_utils.approach_speed_video(
    source_path=SOURCE_VIDEO_PATH,
    target_path=TARGET_VIDEO_PATH,
    callback=callback,
    target_fps=10,
    codec="mp4v"
)

#%% Get focal length and sensor size from image metadata
SOURCE_IMG_PATH = os.path.join(".", "demo_data", "imgs", "02030011.JPG")
exif = pw_utils.get_exif_info(SOURCE_IMG_PATH)
focal_length_mm = exif.get("FocalLength", None) 
sensor_width_mm = 6.17
focal_length_px = (focal_length_mm / sensor_width_mm) * image_width_px

#%% Estimate relative distance using focal length and bounding box height
real_object_height_m = pw_utils.REAL_OBJECT_HEIGHTS_M[most_voted_class]

distances = [
    (focal_length_px * real_object_height_m) / h 
    for h in bounding_heights
]

#%% Compute approach speed using distance differences over time (Δdistance / Δtime)
speeds = []
for i in range(1, len(distances)):
    dt = timestamps[i] - timestamps[i - 1]
    if distances[i] and distances[i-1] and dt > 0:
        delta_d = abs(distances[i] - distances[i - 1])
        speed = delta_d / dt
        speeds.append(speed)
    else:
        speeds.append(None)

# Average speed
average_speed = np.nanmean(speeds)  # Ignore NaN values

#%% Plot the computed approach speed over time
fig = plt.figure()
plt.plot(timestamps[1:], speeds)
plt.xlabel("Time (s)")
plt.ylabel("Approach Speed (m/s)")
plt.title(f"Average Approach Speed: {average_speed:.2f} m/s")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(OUTPUT_FOLDER, f"approach_speed_plot_{VIDEO_NAME}.png"))

#%% Save Speed and Timestamp Data to CSV
data = {
    "timestamp": timestamps[1:],              # Exclude first timestamp to align with speed
    "bounding_height": bounding_heights[1:],  # Exclude first to align
    "distance_estimate": distances[1:],       # Exclude first to align
    "approach_speed": speeds                  # Already aligned
}

df = pd.DataFrame(data)

# Save to CSV
csv_path = os.path.join(OUTPUT_FOLDER, f"approach_speed_data_{VIDEO_NAME}.csv")
df.to_csv(csv_path, index=False)
print(f"Saved CSV to: {csv_path}")
