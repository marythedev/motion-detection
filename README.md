# Computer Vision Final Project


## Video Motion Detection and Viewport Tracking
This project implements a motion detection system with a moving viewport that follows detected motion in a video.  

How to Run

### 1. Install Dependencies
pip install -r requirements.txt

### 2. Run the Script
python src/main.py --video path/to/video.mp4 [--output path/to/output.mp4] [--fps 5] [--viewport_size WIDTHxHEIGHT]

| Argument          | Required | Description                                         |
| ----------------- | -------- | --------------------------------------------------- |
| `--video`         | Yes        | Path to the input video file                        |
| `--output`        | No        | Path to save processed output video                 |
| `--fps`           | No        | Frames per second to sample from video (default: 5) |
| `--viewport_size` | No        | Dimensions of viewport in pixels (e.g., `640x480`)  |

Example:   
python src/main.py --video data/sample_video_clip.mp4 --output results/output.mp4 --fps 5 --viewport_size 640x480

## Approach
The system consists of several modular components:

1. Frame Extraction (frame_processor.py)
- Reads video, downsampled to target FPS.
- Resizes frames for consistent processing.

2. Motion Detection (motion_detector.py)
- Uses cv2.createBackgroundSubtractorMOG2 for detecting moving objects.
- Thresholding + contour detection to locate motion.

3. Viewport Tracking (viewport_tracker.py)
- Maintains a moving window that follows motion.
- Ensures the viewport stays within bounds.

4. Visualization (visualizer.py)
- Draws bounding boxes around motion.
- Overlays the viewport rectangle.

5. Main Controller (main.py)
- Manages the process: read args → process frames → detect motion → track viewport → visualize results.