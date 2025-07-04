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
    "speed_in_video",
    "REAL_ANIMAL_HEIGHT_M",
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
) -> Tuple[int, dict]:
    """
    Tracks animal in video and estimates 2D movement speed (px/s) using only first and last point of the longest track.
    Saves the full video but only overlays bounding boxes on first and last detection.

    Returns:
        width (int): Image width in pixels.
        track_summaries (dict): Dictionary of track_id -> dict with speed and key points.
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
                for i, (xyxy, track_id) in enumerate(zip(tracked.xyxy, tracked.tracker_id)):
                    x1, y1, x2, y2 = xyxy
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    positions_by_id[track_id].append((timestamp, cx, cy))
                    frames_by_id[track_id].append((frame_rgb.copy(), xyxy, labels[i][0], frame_idx))
            all_frames.append(frame_rgb)
            processed_frame_idxs.append(frame_idx)
        frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    # Get longest track
    if not positions_by_id:
        raise ValueError("No tracks found.")

    track_summaries = {}

    # Compute speed + summary info for each track
    for track_id, points in positions_by_id.items():
        if len(points) < 2:
            continue  # skip short tracks

        (t1, x1, y1), (t2, x2, y2) = points[0], points[-1]
        dt = t2 - t1
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        speed = distance / dt if dt > 0 else 0

        (frame1, box1, label1, idx1) = frames_by_id[track_id][0]
        (frame2, box2, label2, idx2) = frames_by_id[track_id][-1]

        track_summaries[track_id] = {
            'speed': speed,
            'points': ((t1, x1, y1), (t2, x2, y2)),
            'boxes': (box1, box2),
            'idxs': (idx1, idx2)
        }

    # Assign colors
    colors = {track_id: tuple(np.random.randint(0, 256, size=3).tolist())
              for track_id in track_summaries}

    # Define visual parameters based on video width
    scale = width / 1920
    box_thickness = max(int(3 * scale), 1)
    point_radius = int(6 * scale)
    line_thickness = int(3 * scale)
    font_scale = 1.0 * scale
    font_thickness = int(3 * scale)

    # Write frames with annotations
    for idx, frame in enumerate(all_frames):
        annotated = frame.copy()

        for i, (track_id, summary) in enumerate(track_summaries.items()):
            box1, box2 = summary['boxes']
            (_, c1_x, c1_y), (_, c2_x, c2_y) = summary['points']
            idx1, idx2 = summary['idxs']
            speed = summary['speed']
            color = colors[track_id]

            if idx >= processed_frame_idxs.index(idx1):
                x1_box1, y1_box1, x2_box1, y2_box1 = [int(v) for v in box1]
                cv2.rectangle(annotated, (x1_box1, y1_box1), (x2_box1, y2_box1), color, box_thickness)
                cv2.circle(annotated, (int(c1_x), int(c1_y)), point_radius, color, -1)

            if idx >= processed_frame_idxs.index(idx2):
                x1_box2, y1_box2, x2_box2, y2_box2 = [int(v) for v in box2]
                cv2.rectangle(annotated, (x1_box2, y1_box2), (x2_box2, y2_box2), color, box_thickness)
                cv2.circle(annotated, (int(c2_x), int(c2_y)), point_radius, color, -1)
                cv2.line(annotated, (int(c1_x), int(c1_y)), (int(c2_x), int(c2_y)), color, line_thickness)
                cv2.putText(
                    annotated, f"Speed: {speed:.2f} px/s",
                    (10, int(30 * scale) + int(40 * scale) * int(i)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv2.LINE_AA
                )

        out.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    out.release()
    return width, track_summaries

REAL_ANIMAL_HEIGHT_M = {
    'Dasyprocta': 0.3,      
    'Bos': 1.4,             
    'Pecari': 0.5,          
    'Mazama': 0.6,          
    'Cuniculus': 0.3,       
    'Leptotila': 0.2,       
    'Human': 1.6,          
    'Aramides': 0.3,        
    'Tinamus': 0.3,         
    'Eira': 0.3,           
    'Crax': 0.5,           
    'Procyon': 0.4,         
    'Capra': 0.8,           
    'Dasypus': 0.25,        
    'Sciurus': 0.2,        
    'Crypturellus': 0.3,    
    'Tamandua': 0.4,        
    'Proechimys': 0.2,     
    'Leopardus': 0.4,       
    'Equus': 1.5,           
    'Columbina': 0.15,      
    'Nyctidromus': 0.15,    
    'Ortalis': 0.4,        
    'Emballonura': 0.1,     
    'Odontophorus': 0.3,    
    'Geotrygon': 0.2,      
    'Metachirus': 0.25,     
    'Catharus': 0.1,        
    'Cerdocyon': 0.4,      
    'Momotus': 0.25,        
    'Tapirus': 1.0,        
    'Canis': 0.7,          
    'Furnarius': 0.15,      
    'Didelphis': 0.3,       
    'Sylvilagus': 0.25,     
    'Unknown': 0.25         
}
