# frame_processor.py
"""
Frame processing functions for the motion detection project.
"""

import cv2
import numpy as np


def process_video(video_path, target_fps=5, resize_dim=(1280, 720)):
    """
    Extract frames from a video at a specified frame rate.

    Args:
        video_path: Path to the video file
        target_fps: Target frames per second to extract
        resize_dim: Dimensions to resize frames to (width, height)

    Returns:
        List of extracted frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    # Get original video FPS and frame count
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame interval for target FPS sampling
    if orig_fps <= 0:
        orig_fps = 30  # default fallback

    frame_interval = max(int(orig_fps // target_fps), 1)

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Sample frames at interval
        if frame_idx % frame_interval == 0:
            # Resize frame to target dimensions (width, height)
            resized = cv2.resize(frame, resize_dim)
            frames.append(resized)

        frame_idx += 1

    cap.release()
    
    return frames

