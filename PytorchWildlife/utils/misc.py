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
from typing import Callable, Tuple, Optional, List, Dict
from tqdm import tqdm
from bisect import bisect_left


__all__ = [
    "process_video",
    "speed_in_video",
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
    codec: str = "mp4v",
    *,
    # selection
    longest: bool = False,
    # filters
    min_points: int = 2,
    min_duration_s: float | None = None,
    min_displacement_px: float | None = None,
    # subtrack cleanup
    suppress_subtracks: bool = False,
    subtrack_radius_px: float = 30.0,
    # UX
    show_progress: bool = False,
) -> Tuple[int, Dict[int, dict]]:
    """
    Track animals and estimate speed (px/s) from first->last observation of each track,
    or only the single longest track when `longest=True`.

    Filters:
      - min_points: require at least N observations
      - min_duration_s: require duration >= this
      - min_displacement_px: require end-to-end displacement >= this
    Cleanup:
      - suppress_subtracks: remove tracks contained in time within a longer track whose
        mean temporal nearest-neighbor distance is <= subtrack_radius_px

    Returns:
      width, track_summaries: {track_id: {'speed', 'points', 'boxes', 'idxs'}}
    """
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {source_path}")

    input_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not input_fps or np.isnan(input_fps) or input_fps <= 0:
        input_fps = float(target_fps) if target_fps and target_fps > 0 else 30.0

    stride = max(1, int(round(input_fps / max(target_fps, 1))))
    output_fps = input_fps

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(target_path, fourcc, output_fps, (width, height))

    tracker = sv.ByteTrack()

    positions_by_id: Dict[int, List[Tuple[float, float, float]]] = defaultdict(list)
    frames_by_id: Dict[int, List[Tuple[np.ndarray, np.ndarray, str, int]]] = defaultdict(list)
    all_frames: List[np.ndarray] = []

    pbar = None
    if show_progress:
        try:
            from tqdm.auto import tqdm
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
            pbar = tqdm(total=total, desc="Reading frames", leave=False, dynamic_ncols=True)
        except Exception:
            pbar = None

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        all_frames.append(frame_rgb)

        if (frame_idx % stride) == 0:
            t = frame_idx / input_fps
            _, detections, labels = callback(frame_rgb.copy(), frame_idx)

            if detections is not None and len(detections) > 0:
                tracked = tracker.update_with_detections(detections)
                for i, (xyxy, track_id) in enumerate(zip(tracked.xyxy, tracked.tracker_id)):
                    x1, y1, x2, y2 = xyxy
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    tid = int(track_id)
                    positions_by_id[tid].append((t, cx, cy))
                    label_text = labels[i][0] if (labels and i < len(labels)) else ""
                    frames_by_id[tid].append((frame_rgb.copy(), xyxy, label_text, frame_idx))

        frame_idx += 1
        if pbar is not None:
            pbar.update(1)

    cap.release()
    if pbar is not None:
        pbar.close()

    if not positions_by_id:
        out.release()
        raise ValueError("No tracks found.")

    # ----- Filter by points/duration/displacement/speed -----
    def stats_for(pts):
        (t1, x1, y1), (t2, x2, y2) = pts[0], pts[-1]
        duration = t2 - t1
        disp = float(np.hypot(x2 - x1, y2 - y1))
        speed = (disp / duration) if duration > 0 else 0.0
        return t1, t2, duration, disp, speed

    kept_ids = []
    track_pts = {}
    for tid, pts in positions_by_id.items():
        if len(pts) < min_points:
            continue
        t1, t2, dur, disp, spd = stats_for(pts)
        if (min_duration_s is not None) and (dur < float(min_duration_s)):
            continue
        if (min_displacement_px is not None) and (disp < float(min_displacement_px)):
            continue
        kept_ids.append(tid)
        track_pts[tid] = pts

    if not kept_ids:
        out.release()
        raise ValueError("No tracks passed the thresholds.")

    # ----- Optional sub-track suppression (temporal containment + spatial proximity) -----
    def mean_temporal_nn_distance(A, B):
        # A, B: [(t,x,y), ...] sorted by t
        tB = [p[0] for p in B]
        if not tB:
            return float("inf")
        dists = []
        for t, x, y in A:
            if t < tB[0] or t > tB[-1]:
                continue  # only compare where time overlaps
            j = bisect_left(tB, t)
            cand = []
            for k in (j - 1, j):
                if 0 <= k < len(B):
                    tb, xb, yb = B[k]
                    cand.append(np.hypot(x - xb, y - yb))
            if cand:
                dists.append(min(cand))
        return (sum(dists) / len(dists)) if dists else float("inf")

    if suppress_subtracks and len(kept_ids) > 1:
        # Sort by (num points, duration) descending → prefer keeping longer, more stable tracks
        order = sorted(
            kept_ids,
            key=lambda tid: (len(track_pts[tid]), track_pts[tid][-1][0] - track_pts[tid][0][0]),
            reverse=True,
        )
        to_drop = set()
        for i, tid_main in enumerate(order):
            if tid_main in to_drop:
                continue
            t1m, t2m, _, _, _ = stats_for(track_pts[tid_main])
            for tid_other in order[i + 1:]:
                if tid_other in to_drop:
                    continue
                t1o, t2o, _, _, _ = stats_for(track_pts[tid_other])
                # temporal containment: other is fully inside main
                if t1m <= t1o and t2o <= t2m:
                    md = mean_temporal_nn_distance(track_pts[tid_other], track_pts[tid_main])
                    if md <= subtrack_radius_px:
                        to_drop.add(tid_other)
        kept_ids = [tid for tid in kept_ids if tid not in to_drop]

    # ----- Optionally keep only the single longest among remaining -----
    if longest and len(kept_ids) > 1:
        kept_ids = sorted(
            kept_ids,
            key=lambda tid: (len(track_pts[tid]), track_pts[tid][-1][0] - track_pts[tid][0][0]),
            reverse=True,
        )[:1]

    # ----- Build summaries -----
    track_summaries: Dict[int, dict] = {}
    for tid in kept_ids:
        pts = track_pts[tid]
        (t1, x1, y1), (t2, x2, y2) = pts[0], pts[-1]
        dt = max(t2 - t1, 0.0)
        disp = float(np.hypot(x2 - x1, y2 - y1))
        spd = (disp / dt) if dt > 0 else 0.0

        (frame1, box1, label1, idx1) = frames_by_id[tid][0]
        (frame2, box2, label2, idx2) = frames_by_id[tid][-1]

        track_summaries[int(tid)] = {
            "speed": spd,
            "points": ((t1, x1, y1), (t2, x2, y2)),
            "boxes": (box1, box2),
            "idxs": (idx1, idx2),
            "labels": (label1, label2),
        }

    if not track_summaries:
        out.release()
        raise ValueError("No tracks left after filtering.")

    # ----- Render -----
    rng = np.random.default_rng(0)
    colors = {tid: tuple(int(c) for c in rng.integers(0, 256, size=3)) for tid in track_summaries}

    scale = max(width, 1) / 1920.0
    box_thickness  = max(int(3 * scale), 1)
    point_radius   = max(int(6 * scale), 2)
    line_thickness = max(int(3 * scale), 1)
    font_scale     = 1.0 * scale
    font_thickness = max(int(3 * scale), 1)

    for idx, frame in enumerate(all_frames):
        annotated = frame.copy()
        for i, (tid, summary) in enumerate(track_summaries.items()):
            box1, box2 = summary["boxes"]
            (_, c1x, c1y), (_, c2x, c2y) = summary["points"]
            idx1, idx2 = summary["idxs"]
            spd = summary["speed"]
            color = colors[tid]

            if idx >= idx1:
                x1b, y1b, x2b, y2b = [int(v) for v in box1]
                cv2.rectangle(annotated, (x1b, y1b), (x2b, y2b), color, box_thickness)
                cv2.circle(annotated, (int(c1x), int(c1y)), point_radius, color, -1)

            if idx >= idx2:
                x1b, y1b, x2b, y2b = [int(v) for v in box2]
                cv2.rectangle(annotated, (x1b, y1b), (x2b, y2b), color, box_thickness)
                cv2.circle(annotated, (int(c2x), int(c2y)), point_radius, color, -1)
                cv2.line(annotated, (int(c1x), int(c1y)), (int(c2x), int(c2y)), color, line_thickness)
                cv2.putText(
                    annotated, f"Speed: {spd:.2f} px/s",
                    (10, int(90 * scale) + int(40 * scale) * int(i)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv2.LINE_AA
                )
        out.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    out.release()
    return width, height, track_summaries