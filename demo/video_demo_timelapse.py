# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Video detection demo """
#%% 
# Importing necessary basic libraries and modules
import numpy as np
import supervision as sv

#%% 
# PyTorch imports for tensor operations
import torch
import os
#%% 
# Importing the models, transformations, and utility functions from PytorchWildlife 
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification
from PytorchWildlife import utils as pw_utils

# %%
# Importing the library for video processing
from tqdm import tqdm
from typing import Callable
from supervision import VideoInfo, VideoSink, get_video_frames_generator
import json
#%% 
# Setting the device to use for computations ('cuda' indicates GPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SOURCE_VIDEO_PATH = os.path.join("/home/investigacion/Andres_2/PytorchWildlife_March_2025/CameraTraps/demo/demo_data/videos/opossum_example.MP4")
TARGET_VIDEO_PATH = os.path.join(".","demo_data","videos","opossum_example_processed.MP4")

#%% 
# Initializing the MegaDetectorV6 model for image detection
# Valid versions are MDV6-yolov9-c, MDV6-yolov9-e, MDV6-yolov10-c, MDV6-yolov10-e or MDV6-rtdetr-c
detection_model = pw_detection.MegaDetectorV6(device=DEVICE, pretrained=True, version="MDV6-yolov10-e")

# Uncomment the following line to use MegaDetectorV5 instead of MegaDetectorV6
#detection_model = pw_detection.MegaDetectorV5(device=DEVICE, pretrained=True, version="a")

#%% 
# Initializing the model for image classification
classification_model = pw_classification.AI4GOpossum(device=DEVICE, pretrained=True)

#%% 
# Initializing a box annotator for visualizing detections
box_annotator = sv.BoxAnnotator(thickness=4)
lab_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_thickness=4, text_scale=2)

def callback(frame: np.ndarray, index: int) -> np.ndarray:
    """
    Callback function to process each video frame for detection and classification.
    
    Parameters:
    - frame (np.ndarray): Video frame as a numpy array.
    - index (int): Frame index.
    
    Returns:
    annotated_frame (np.ndarray): Annotated video frame.
    """
    
    results_det = detection_model.single_image_detection(frame, img_path=index)
    labels = []
    normalized_coords = []
    classifications = []
    frame_width, frame_height = frame.shape[1], frame.shape[0]
    for xyxy in results_det["detections"].xyxy:
        x_min, y_min, x_max, y_max = xyxy
        cropped_image = sv.crop_image(image=frame, xyxy=xyxy)
        results_clf = classification_model.single_image_classification(cropped_image)
        labels.append("{} {:.2f}".format(results_clf["prediction"], results_clf["confidence"]))
        norm_bbox = [
            x_min / frame_width,  # Normalize x_min
            y_min / frame_height, # Normalize y_min
            (x_max - x_min) / frame_width,  # Normalize width
            (y_max - y_min) / frame_height  # Normalize height
        ]
        classifications.append([str(results_clf["class_id"]), float(results_clf["confidence"])])
        normalized_coords.append(norm_bbox)

    annotation = {
     "category": [str(i) for i in results_det["detections"].class_id],
     "conf": [float(j) for j in results_det["detections"].confidence],
     "bbox": normalized_coords,
     "classifications": classifications,
     "frame_number": index
    }

    return annotation

def process_video_timelapse(
    source_path: str,
    target_path: str,
    callback: Callable[[np.ndarray, int], np.ndarray],
    target_fps: int = 1,
    codec: str = "mp4v",
    detection_categories: dict = {0: 'animal', 1: 'person', 2: 'vehicle'},
    clf_categories: dict = {0: 'opossum', 1: 'other'}
) -> None:
    """
    Process a video frame-by-frame, applying a callback function to each frame and saving the results 
    to a new video. This version includes a progress bar and allows codec selection.
    
    Args:
        source_path (str): 
            Path to the source video file.
        target_path (str): 
            Path to save the processed video.
        callback (Callable[[np.ndarray, int], np.ndarray]): 
            A function that takes a video frame and its index as input and returns the processed frame.
        codec (str, optional): 
            Codec used to encode the processed video. Default is "avc1".
    """
    source_video_info = VideoInfo.from_video_path(video_path=source_path)
    
    if source_video_info.fps > target_fps:
        stride = int(source_video_info.fps / target_fps)
        source_video_info.fps = target_fps
    else:
        stride = 1

    json_results = {
        "info": {"detector": "MDV6-yolov10-e"},
        "detection_categories": detection_categories,
        "classification_categories": clf_categories,
        "images": []
    }
    detections = []
    with VideoSink(target_path=target_path, video_info=source_video_info, codec=codec) as sink:
        with tqdm(total=int(source_video_info.total_frames / stride)) as pbar: 
            for index, frame in enumerate(
                get_video_frames_generator(source_path=source_path, stride=stride)
            ):
                result_frame = callback(frame, index)
                detections.append(result_frame)
                pbar.update(1)
    image_info = {"file": os.path.basename(target_path), "frame_rate":target_fps, "detections": detections}
    json_results["images"].append(image_info)

    # Save the json
    json_path = target_path.replace(".{}".format(target_path.split(".")[-1]), "_detection.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=4)
    print(f"JSON results saved to {json_path}")
process_video_timelapse(source_path=SOURCE_VIDEO_PATH, target_path=TARGET_VIDEO_PATH, callback=callback, target_fps=10, detection_categories=detection_model.CLASS_NAMES, clf_categories=classification_model.CLASS_NAMES)
