# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Miscellaneous functions."""

import supervision as sv
import numpy as np
import cv2
from supervision import VideoInfo, VideoSink, get_video_frames_generator
from PIL import Image
from PIL.ExifTags import TAGS
from collections import Counter
from collections import defaultdict
from typing import Callable, Tuple, Optional, List
from tqdm import tqdm


__all__ = [
    "process_video",
    "approach_speed_video",
    "get_exif_info",
    "REAL_OBJECT_HEIGHTS_M",
    "speed_in_video"
]


def process_video(
    source_path: str,
    target_path: str,
    callback: Callable[[np.ndarray, int], np.ndarray],
    target_fps: int = 1,
    codec: str = "mp4v"
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
    
    with VideoSink(target_path=target_path, video_info=source_video_info, codec=codec) as sink:
        with tqdm(total=int(source_video_info.total_frames / stride)) as pbar: 
            for index, frame in enumerate(
                get_video_frames_generator(source_path=source_path, stride=stride)
            ):
                result_frame = callback(frame, index)
                sink.write_frame(frame=cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))
                pbar.update(1)

def speed_in_video(
    source_path: str,
    target_path: str,
    callback: Callable[[np.ndarray, int], Tuple[np.ndarray, sv.Detections, List[Tuple[str, float]]]],
    target_fps: int = 1,
    codec: str = "mp4v"
) -> Tuple[int, float]:
    """
    Tracks animal in video and estimates 2D movement speed (px/s) using only first and last point of the longest track.
    Saves the full video but only overlays bounding boxes on first and last detection.

    Returns:
        width (int): Image width in pixels.
        speed_px_s (float): Speed in pixels per second based on 2D tracking.
    """
    cap = cv2.VideoCapture(source_path)
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    stride = int(input_fps / target_fps) if input_fps > target_fps else 1
    output_fps = input_fps / stride

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(target_path, fourcc, output_fps, (width, height))

    tracker = sv.ByteTrack()
    positions_by_id = defaultdict(list)
    frames_by_id = defaultdict(list)
    all_frames = []
    processed_frame_idxs = []

    frame_idx = 0
    pbar = tqdm()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            timestamp = frame_idx / input_fps
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            annotated_frame, detections, labels = callback(frame_rgb.copy(), frame_idx)

            if detections:
                tracked = tracker.update_with_detections(detections)
                i = 0
                for xyxy, track_id in zip(tracked.xyxy, tracked.tracker_id):
                    x1, y1, x2, y2 = xyxy
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    positions_by_id[track_id].append((timestamp, cx, cy))
                    frames_by_id[track_id].append((frame_rgb.copy(), xyxy, labels[i][0], frame_idx))
                    i += 1
            all_frames.append(frame_rgb)
            processed_frame_idxs.append(frame_idx)
        frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    # Get longest track
    if not positions_by_id:
        raise ValueError("No tracks found.")

    longest_id = max(positions_by_id.items(), key=lambda x: len(x[1]))[0]
    track_points = positions_by_id[longest_id]

    if len(track_points) < 2:
        raise ValueError("Track too short for speed calculation.")

    # Speed calculation
    (t1, x1, y1), (t2, x2, y2) = track_points[0], track_points[-1]
    dt = t2 - t1
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    speed = distance / dt if dt > 0 else 0

    # First and last frame info
    (frame1, box1, label1, idx1), (frame2, box2, label2, idx2) = frames_by_id[longest_id][0], frames_by_id[longest_id][-1]

    # Prepare annotations
    box_annotator = sv.BoxAnnotator(thickness=3, color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_thickness=2, text_scale=1, color_lookup=sv.ColorLookup.INDEX)

    idx1_in_all = processed_frame_idxs.index(idx1)
    idx2_in_all = processed_frame_idxs.index(idx2)

    for idx, frame in enumerate(all_frames):
        annotated = frame.copy()
        if idx >= idx1_in_all:
            det = sv.Detections(xyxy=np.array([box1]))
            annotated = label_annotator.annotate(box_annotator.annotate(annotated, det), det, labels=[label1])
        if idx >= idx2_in_all:
            det = sv.Detections(xyxy=np.array([box2]))
            annotated = label_annotator.annotate(box_annotator.annotate(annotated, det), det, labels=[label2])
        
        # Dibuja la trayectoria acumulada hasta este frame
        for i in range(1, len(track_points)):
            (t_prev, x_prev, y_prev) = track_points[i - 1]
            (t_curr, x_curr, y_curr) = track_points[i]

            # Recupera frame_idx del punto actual en la trayectoria
            frame_curr = frames_by_id[longest_id][i][3]  # índice de frame del punto actual

            if frame_curr <= processed_frame_idxs[idx]:
                cv2.line(
                    annotated,
                    (int(x_prev), int(y_prev)),
                    (int(x_curr), int(y_curr)),
                    (0, 255, 0),
                    2
                )

            # write the speed in a text box at track point idx2_in_all
            cv2.putText(annotated, f"Speed: {speed:.2f} px/s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        out.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    out.release()
    return width, t1, x1, y1, t2, x2, y2, speed

def approach_speed_video(
    source_path: str,
    target_path: str,
    callback: Callable[[np.ndarray, int], Tuple[np.ndarray, bool]],
    target_fps: int = 1,
    codec: str = "mp4v"
) -> Tuple[List[float], List[int], List[float]]:
    """
    Processes a video frame-by-frame using a callback, saving the result to a video,
    and returns timestamps and frame indices where animals were detected.

    Args:
        source_path (str): Path to the source video.
        target_path (str): Path to save the processed video.
        callback (Callable): A function taking (frame, index) and returning (annotated_frame, detected_flag).
        target_fps (int): Output FPS. Default is 1.
        codec (str): FourCC codec string for output video. Default is 'mp4v'.

    Returns:
        Tuple[List[float], List[int]]: Timestamps and frame indices where animals were detected.
    """
    cap = cv2.VideoCapture(source_path)
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stride = int(input_fps / target_fps) if input_fps > target_fps else 1
    output_fps = input_fps / stride

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(target_path, fourcc, output_fps, (width, height))

    timestamps = []
    bounding_heights = []
    species = []

    frame_idx = 0
    pbar = tqdm(total=total_frames // stride)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            timestamp = frame_idx / input_fps

            annotated_frame, detected, height, specie = callback(frame, frame_idx)

            if detected:
                timestamps.append(timestamp)
                bounding_heights.append(height)
                species.append(specie)

            out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            pbar.update(1)

        frame_idx += 1
    
    cap.release()
    out.release()
    pbar.close()

    all_labels = [label for dets in species for (label, _) in dets]
    label_counts = Counter(all_labels)
    most_voted_class = label_counts.most_common(1)[0]  # (label, count)

    # Find highest confidence
    max_conf = -1
    best_detection = None
    for dets in species:
        for label, conf in dets:
            if conf > max_conf:
                max_conf = conf
                best_detection = (label, conf)

    print(f"✅ Most voted class: {most_voted_class[0]} ({most_voted_class[1]} votes)")
    print(f"⭐ Highest confidence detection: {best_detection[0]} ({best_detection[1]:.2f})")
    
    return width, timestamps, bounding_heights, most_voted_class[0], best_detection[0]

def get_exif_info(img_path):
    img = Image.open(img_path)
    exif_data = img._getexif()

    if exif_data is None:
        print("No EXIF data found.")
        return None

    exif = {}
    for tag, value in exif_data.items():
        tag_name = TAGS.get(tag, tag)
        exif[tag_name] = value

    return exif

REAL_OBJECT_HEIGHTS_M = {
    'Dasyprocta': 0.3,      # agouti
    'Bos': 1.4,             # cattle
    'Pecari': 0.5,          # peccary
    'Mazama': 0.6,          # brocket deer
    'Cuniculus': 0.3,       # paca
    'Leptotila': 0.2,       # dove
    'Human': 1.6,           # adult human (shoulder-ish)
    'Aramides': 0.3,        # wood-rail
    'Tinamus': 0.3,         # tinamou
    'Eira': 0.3,            # tayra
    'Crax': 0.5,            # curassow
    'Procyon': 0.4,         # raccoon
    'Capra': 0.8,           # goat
    'Dasypus': 0.25,        # armadillo
    'Sciurus': 0.2,         # squirrel
    'Crypturellus': 0.3,    # small tinamou
    'Tamandua': 0.4,        # lesser anteater
    'Proechimys': 0.2,      # spiny rat
    'Leopardus': 0.4,       # ocelot/margay
    'Equus': 1.5,           # horse
    'Columbina': 0.15,      # ground dove
    'Nyctidromus': 0.15,    # nightjar
    'Ortalis': 0.4,         # chachalaca
    'Emballonura': 0.1,     # bat
    'Odontophorus': 0.3,    # quail
    'Geotrygon': 0.2,       # quail-dove
    'Metachirus': 0.25,     # opossum
    'Catharus': 0.1,        # small thrush
    'Cerdocyon': 0.4,       # crab-eating fox
    'Momotus': 0.25,        # motmot
    'Tapirus': 1.0,         # tapir (shoulder)
    'Canis': 0.7,           # dog
    'Furnarius': 0.15,      # ovenbird
    'Didelphis': 0.3,       # opossum
    'Sylvilagus': 0.25,     # cottontail
    'Opossum': 0.3,         # Opossum   
    'Unknown': 0.25         # default fallback value
}
