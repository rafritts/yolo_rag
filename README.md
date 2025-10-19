# Webcam Object Detection

Real-time object detection using YOLOv8 and your webcam, optimized for NVIDIA GPUs like the RTX 4090.

## Features

- Real-time object detection using your webcam
- Utilizes GPU acceleration (CUDA) for fast inference
- Displays frames per second (FPS) for performance monitoring
- Simple interface with easy exit (press 'q')

## Requirements

- Python 3.8-3.11
- Poetry (for dependency management)
- NVIDIA GPU with CUDA support (optimized for RTX series)
- Webcam

## Installation

1. Clone this repository
2. Install dependencies using Poetry:

```bash
# Install Poetry if you don't have it
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
cd webcam-object-detection
poetry install
```

## Usage

Run the application using Poetry:

```bash
poetry run python -m webcam_object_detection.main
```

The application will:
1. Load the YOLOv8 model (downloading it if needed)
2. Open your webcam
3. Display a window with real-time object detection
4. Show FPS in the top-left corner
5. Exit when you press 'q'

## Customization

You can modify the following parameters in `main.py`:

- Model size: Change `yolov8n.pt` to use larger models:
  - `yolov8n.pt`: Nano (fastest)
  - `yolov8s.pt`: Small
  - `yolov8m.pt`: Medium
  - `yolov8l.pt`: Large
  - `yolov8x.pt`: Extra Large (most accurate)

- Detection confidence: Adjust `conf=0.25` to show more or fewer detections
- Camera source: Change `cv2.VideoCapture(0)` to use a different camera

## License

MIT