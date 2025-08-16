# viewport_tracker.py
"""
Viewport tracking functions for creating a smooth "virtual camera".
"""

import cv2
import numpy as np


def calculate_region_of_interest(motion_boxes, frame_shape):
    """
    Calculate the primary region of interest based on motion boxes.

    Args:
        motion_boxes: List of motion detection bounding boxes
        frame_shape: Shape of the video frame (height, width)

    Returns:
        Tuple (x, y, w, h) representing the region of interest center point and dimensions
    """
    if not motion_boxes:
        # If no motion is detected, use the center of the frame
        height, width = frame_shape[:2]
        return (width // 2, height // 2, 0, 0)

    # Choose largest motion box by area
    largest_box = max(motion_boxes, key=lambda b: b[2] * b[3])
    x, y, w, h = largest_box
    x_center = x + w // 2
    y_center = y + h // 2
    
    return (x_center, y_center, w, h)


def track_viewport(frames, motion_results, viewport_size, smoothing_factor=0.3):
    """
    Track viewport position across frames with smoothing.

    Args:
        frames: List of video frames
        motion_results: List of motion detection results for each frame
        viewport_size: Tuple (width, height) of the viewport
        smoothing_factor: Factor for smoothing viewport movement (0-1)
                          Lower values create smoother movement

    Returns:
        List of viewport positions for each frame as (x, y) center coordinates
    """
    viewport_positions = []

    if not frames:
        return viewport_positions

    frame_h, frame_w = frames[0].shape[:2]
    vp_w, vp_h = viewport_size

    # Initialize viewport center to frame center
    prev_x, prev_y = frame_w // 2, frame_h // 2

    for i, motion_boxes in enumerate(motion_results):
        # Calculate region of interest center
        roi_x, roi_y, roi_w, roi_h = calculate_region_of_interest(motion_boxes, frames[i].shape)

        # If no motion detected (roi_w, roi_h == 0), keep previous center
        if roi_w == 0 and roi_h == 0:
            target_x, target_y = prev_x, prev_y
        else:
            target_x, target_y = roi_x, roi_y

        # Smooth viewport center using exponential moving average
        new_x = int(prev_x * (1 - smoothing_factor) + target_x * smoothing_factor)
        new_y = int(prev_y * (1 - smoothing_factor) + target_y * smoothing_factor)

        # Clamp viewport so it stays fully inside frame
        half_vp_w, half_vp_h = vp_w // 2, vp_h // 2
        new_x = max(half_vp_w, min(new_x, frame_w - half_vp_w))
        new_y = max(half_vp_h, min(new_y, frame_h - half_vp_h))

        viewport_positions.append((new_x, new_y))
        prev_x, prev_y = new_x, new_y

    return viewport_positions
