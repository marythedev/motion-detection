# visualizer.py
"""
Visualization functions for displaying motion detection and viewport tracking results.
"""

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# loading model that detects players and balls
model_path = os.path.join(os.path.dirname(__file__), 'human_vs_ball_model.h5')
model = load_model(model_path)

CLASS_NAMES = 'human', 'ball'

def resize_with_padding(image, target_size=128):
    """
    resising the roi to send to the model while keeping the proportions
    
    """
    h, w = image.shape[:2]
    aspect = w / h

    if w > h:
        new_w = target_size
        new_h = int(target_size / aspect)
    else:
        new_h = target_size
        new_w = int(target_size * aspect)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x = (target_size - new_w) // 2
    y = (target_size - new_h) // 2
    padded[y:y+new_h, x:x+new_w] = resized

    return padded


def predict_object(roi):
    """
    classifies roi as human or ball (returns the label and confidence)
    
    takes: motion detected ROI
    
    """
    
    if roi.size == 0:
        return 'unknown', 0.0

    # preprocessing
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)      # turning roi into rgb
    img = resize_with_padding(img, 128)             # resizing
    img = img / 255.0                               # normalizing
    img = np.expand_dims(img, axis=0)               # preparing a batch

    # getting prediction
    pred = model.predict(img, verbose=0)
    class_idx = np.argmax(pred)
    label = CLASS_NAMES[class_idx]
    confidence = float(np.max(pred))

    return label, confidence



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

    ball_trajectory = []

    for i, frame in enumerate(frames):
        annotated = frame.copy()
        
        best_ball = None
        best_conf = 0.0
        best_center = None

        for (x, y, w, h) in motion_results[i]:
            
            roi = frame[y:y+h, x:x+w]
            if w < 15 or h < 15 or roi.size == 0:
                continue
            
            # drawing the box and labeling roi
            label, confidence = predict_object(roi)
            if (label == 'human'):
                label = 'player'
            
            if label == 'ball' and confidence > best_conf:
                best_conf = confidence
                best_ball = (x, y, w, h)
                best_center = (x + w // 2, y + h // 2)
            
            # ball labeling
            if best_ball and best_conf > 0.7:
                x, y, w, h = best_ball
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(annotated, f"ball", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if best_center:
                    ball_trajectory.append(best_center)
            else:
                # player labeling
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(annotated, f"{label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ball trajectory visualization based on current and last appearance
            for j in range(1, len(ball_trajectory)):
                pt1 = ball_trajectory[j - 1]
                pt2 = ball_trajectory[j]
                if pt1 and pt2:
                    cv2.line(annotated, pt1, pt2, (0, 0, 255), 2)

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
            (0, 0, 0),
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