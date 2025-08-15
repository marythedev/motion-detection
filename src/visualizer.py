# visualizer.py
"""
Visualization functions for displaying motion detection and viewport tracking results.
"""

import os
import cv2
import numpy as np


def visualize_results(frames, motion_results, viewport_positions, viewport_size, output_dir):
    """
    Create visualization of motion detection and viewport tracking results.

    Args:
        frames: List of video frames
        motion_results: List of motion detection results for each frame
        viewport_positions: List of viewport center positions for each frame
        viewport_size: Tuple (width, height) of the viewport
        output_dir: Directory to save visualization results
    """
    # Create output directory for frames
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    viewport_dir = os.path.join(output_dir, "viewport")
    os.makedirs(viewport_dir, exist_ok=True)

    # Get dimensions for the output video
    height, width = frames[0].shape[:2]

    # Create video writers
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(output_dir, "motion_detection.mp4")
    video_writer = cv2.VideoWriter(video_path, fourcc, 5, (width, height))

    viewport_video_path = os.path.join(output_dir, "viewport_tracking.mp4")
    vp_width, vp_height = viewport_size
    viewport_writer = cv2.VideoWriter(
        viewport_video_path, fourcc, 5, (vp_width, vp_height)
    )

    # TODO: Implement visualization
    # 1. Process each frame
    #    a. Create a copy of the frame for visualization
    #    b. Draw bounding boxes around motion regions
    #       (hint: cv2.rectangle with green color (0, 255, 0))
    #    c. Draw the viewport rectangle
    #       (hint: cv2.rectangle with blue color (255, 0, 0))
    #    d. Extract the viewport content (the area inside the viewport)
    #    e. Add frame number to the visualization (hint: cv2.putText)
    #    f. Save visualization frames and viewport frames as images
    #    g. Write frames to both video writers
    # 2. Release the video writers when done

    # Example starter code:
    for i, frame in enumerate(frames):
        annotated = frame.copy()

        # Draw motion bounding boxes in green
        for (x, y, w, h) in motion_results[i]:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw viewport rectangle in blue
        vp_cx, vp_cy = viewport_positions[i]
        vp_x1 = max(0, vp_cx - vp_width // 2)
        vp_y1 = max(0, vp_cy - vp_height // 2)
        vp_x2 = min(width, vp_x1 + vp_width)
        vp_y2 = min(height, vp_y1 + vp_height)
        cv2.rectangle(annotated, (vp_x1, vp_y1), (vp_x2, vp_y2), (255, 0, 0), 2)

        # Add frame number text (top-left corner)
        cv2.putText(
            annotated,
            f"Frame {i + 1}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        
        # Extract viewport content, pad if near edges
        viewport_crop = frame[vp_y1:vp_y2, vp_x1:vp_x2]
        # Pad if crop smaller than viewport size (e.g., edges)
        crop_h, crop_w = viewport_crop.shape[:2]
        if crop_w < vp_width or crop_h < vp_height:
            pad_right = vp_width - crop_w
            pad_bottom = vp_height - crop_h
            viewport_crop = cv2.copyMakeBorder(
                viewport_crop,
                0,
                pad_bottom,
                0,
                pad_right,
                cv2.BORDER_REPLICATE,
        )
            
        # Save frames as PNG images
        cv2.imwrite(os.path.join(frames_dir, f"frame_{i:05d}.png"), annotated)
        cv2.imwrite(os.path.join(viewport_dir, f"viewport_{i:05d}.png"), viewport_crop)

        # Write frames to video writers
        video_writer.write(annotated)
        viewport_writer.write(viewport_crop)
        
    video_writer.release()
    viewport_writer.release()

    print(f"Visualization saved to {video_path}")
    print(f"Viewport video saved to {viewport_video_path}")
    print(f"Individual frames saved to {frames_dir} and {viewport_dir}")