# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Miscellaneous functions."""

import numpy as np
from tqdm import tqdm
from typing import Callable
from supervision import VideoInfo, VideoSink, get_video_frames_generator

__all__ = [
    "process_video"
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
                sink.write_frame(frame=result_frame)
                pbar.update(1)
