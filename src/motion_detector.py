
# motion_detector.py
"""
Motion detection functions for the sports video analysis project.
"""

import cv2
import numpy as np

def detect_motion(frames, frame_idx, threshold=25, min_area=100):
    """
    Detect motion in the current frame by comparing with previous frame.

    Args:
        frames: List of video frames
        frame_idx: Index of the current frame
        threshold: Threshold for frame difference detection
        min_area: Minimum contour area to consider

    Returns:
        List of bounding boxes for detected motion regions
    """
    # We need at least 2 frames to detect motion
    if frame_idx < 1 or frame_idx >= len(frames):
        return []

    # Get current and previous frame
    current_frame = frames[frame_idx]
    prev_frame = frames[frame_idx - 1]


    # 1. Convert frames to grayscale
    gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Apply Gaussian blur to reduce noise (hint: cv2.GaussianBlur)
    blurred_current = cv2.GaussianBlur(gray_current, (5, 5), 0)
    blurred_prev = cv2.GaussianBlur(gray_prev, (5, 5), 0)
    
    # 3. Calculate absolute difference between frames (hint: cv2.absdiff)
    diff = cv2.absdiff(blurred_current, blurred_prev)
    
    # 4. Apply threshold to highlight differences (hint: cv2.threshold)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # 5. Dilate the thresholded image to fill in holes (hint: cv2.dilate)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    # 6. Find contours in the thresholded image (hint: cv2.findContours)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 7. Filter contours by area and extract bounding boxes
    motion_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            motion_boxes.append((x, y, w, h))

    # Your implementation here

    return motion_boxes