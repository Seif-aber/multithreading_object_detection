# Multithreading Object Detection
## Description
The app is a Python-based tool designed to leverage multithreading capabilities for running object detection models on video files.
## Installation
1. Clone the repository :
```bash
git clone https://github.com/Seif-aber/multithreading_object_detection
```
2. Install dependencies :
```bash
pip install -r requirements.txt
```
## Usage
To run the object detection model on a video, use the following command-line parameters:
```bash
python object_detection.py --v /path/to/video/file --m /path/to/Yolov8/model --f --s --c
```
### Script Parameters:
- `--v`: Path of the video file (required).
- `--m`: Path of the Yolov8 model (default: `Models/yolov8n.pt`).
- `--f`: Save frames and predicted images (optional, default: `False`).
- `--s`: Save the result video (optional, default: `False`).
- `--c`: Save the database in a csv file (optional, default: `False`).

## Examples
- Detect objects in a video using the default Yolov8n model :
```bash
python object_detection.py --v Videos/sample_video.mp4
```
- Specify a custom Yolov8 model and save frames/predicted images along with the resulting video :
```bash
python object_detection.py --v videos/sample_video.mp4 --m Models/custom_yolov8.pt --f --s
```
