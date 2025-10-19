# Refactoring Plan for yolo_rag.py

## Overview
The `yolo_rag.py` script currently has a monolithic `main()` function with ~120 lines. This document outlines a refactoring plan to improve code organization, readability, and maintainability.

## Current Issues
- Main loop contains ~90 lines of nested logic
- Box saving logic is ~40 lines within the main loop
- Multiple responsibilities mixed together (setup, processing, saving, display)
- Hardcoded configuration values scattered throughout
- Difficult to test individual components

## Proposed Refactoring

### High Priority (Most Impact)

#### 1. Box & Image Saving Logic (lines 58-97)
**Current State:** ~40 lines of nested logic inside the main loop

**Proposed Function:**
```python
def save_detection_results(results, frame, frame_number, boxes_dir, model):
    """
    Save detection results including cropped images and JSON metadata.

    Args:
        results: YOLO detection results
        frame: Original frame (numpy array)
        frame_number: Current frame number
        boxes_dir: Base directory for saving
        model: YOLO model (for class names)

    Returns:
        int: Number of boxes saved
    """
```

**Benefits:**
- Main loop becomes much cleaner
- Saving logic is isolated and testable
- Easy to modify save format or add new export types

#### 2. Initialization/Setup (lines 9-20)
**Current State:** Camera and model setup mixed with main logic

**Proposed Function:**
```python
def setup_camera_and_model(model_path="yolo11x.pt", device="cuda", camera_index=0):
    """
    Initialize webcam and load YOLO model.

    Args:
        model_path: Path to YOLO model weights
        device: Device to run model on ("cuda" or "cpu")
        camera_index: Camera device index

    Returns:
        tuple: (VideoCapture object, YOLO model)

    Raises:
        RuntimeError: If camera cannot be opened
    """
```

**Benefits:**
- Clear separation of setup vs. processing
- Easy to switch models or camera sources
- Better error handling

#### 3. Display Setup (lines 22-27)
**Current State:** Window configuration scattered in main

**Proposed Function:**
```python
def setup_display(boxes_dir, window_name="YOLO Object Detection", window_size=(1280, 720)):
    """
    Configure OpenCV display window and print startup messages.

    Args:
        boxes_dir: Directory where boxes will be saved (for user message)
        window_name: Name of the display window
        window_size: Tuple of (width, height)

    Returns:
        str: Window name for use in main loop
    """
```

**Benefits:**
- All display configuration in one place
- Easy to customize window settings

#### 4. Frame Annotation (lines 99-114)
**Current State:** Results plotting and FPS overlay mixed together

**Proposed Function:**
```python
def annotate_frame(results, fps, frame_count):
    """
    Create annotated frame with detections and overlay information.

    Args:
        results: YOLO detection results
        fps: Current FPS value
        frame_count: Total frame count

    Returns:
        numpy.ndarray: Annotated frame ready for display
    """
```

**Benefits:**
- Annotation logic separated from main loop
- Easy to add/modify overlay elements
- Can be tested independently

### Medium Priority

#### 5. FPS Calculation (lines 41-47)
**Current State:** Manual FPS tracking with state variables

**Proposed Approach:**
Create a simple FPS tracker class or function:

```python
def update_fps(frame_count, start_time, elapsed_threshold=1.0):
    """
    Calculate FPS and determine if counter should reset.

    Args:
        frame_count: Number of frames since last reset
        start_time: Time of last reset
        elapsed_threshold: Seconds before recalculating FPS

    Returns:
        tuple: (fps, new_frame_count, new_start_time)
    """
```

**Alternative:** Create an `FPSCounter` class for cleaner state management

### Low Priority (Nice to Have)

#### 6. Configuration Constants
Extract hardcoded values to module-level constants or config:

```python
# Configuration
SAVE_INTERVAL = 300  # Save boxes every N frames
DETECTION_CONF = 0.50
DETECTION_IOU = 0.5
MAX_DETECTIONS = 50
BOXES_DIR = "tmp_boxes"
MODEL_PATH = "yolo11x.pt"
DEVICE = "cuda"
WINDOW_SIZE = (1280, 720)
```

#### 7. Box Data Extraction Helper
**Proposed Function:**
```python
def extract_box_info(box, model, box_idx):
    """
    Extract structured information from a detection box.

    Args:
        box: YOLO box object
        model: YOLO model (for class names)
        box_idx: Index of this box

    Returns:
        dict: Box information including coords, confidence, class, image_file
    """
```

## Expected Main Loop After Refactoring

```python
def main():
    # Setup (3-5 lines)
    cap, model = setup_camera_and_model()
    os.makedirs(BOXES_DIR, exist_ok=True)
    window_name = setup_display(BOXES_DIR)

    # State variables (3-4 lines)
    frame_count = 0
    total_frame_count = 0
    start_time = time.time()
    fps = 0

    # Main loop (~15-20 lines)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        total_frame_count += 1
        fps, frame_count, start_time = update_fps(frame_count + 1, start_time)

        results = model(frame, conf=DETECTION_CONF, iou=DETECTION_IOU,
                       max_det=MAX_DETECTIONS, verbose=False)

        if total_frame_count % SAVE_INTERVAL == 0:
            save_detection_results(results, frame, total_frame_count, BOXES_DIR, model)

        annotated_frame = annotate_frame(results, fps, total_frame_count)
        cv2.imshow(window_name, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup (3 lines)
    cap.release()
    cv2.destroyAllWindows()
    print("Program terminated.")
```

## Benefits Summary

1. **Readability:** Main loop reduces from ~90 to ~20 lines
2. **Maintainability:** Each function has a single, clear responsibility
3. **Testability:** Individual components can be unit tested
4. **Reusability:** Functions can be imported and used elsewhere
5. **Configurability:** Constants make it easy to adjust behavior
6. **Extensibility:** Easy to add new features (e.g., different save formats, multiple cameras)

## Implementation Priority

1. Start with High Priority refactorings (biggest impact)
2. Ensure all tests pass after each refactoring
3. Add Medium Priority if time permits
4. Low Priority can be done incrementally

## Next Steps

- [ ] Review and approve this plan
- [ ] Implement High Priority refactorings
- [ ] Test with actual webcam to ensure functionality unchanged
- [ ] Consider adding unit tests for new functions
- [ ] Document any configuration options in README
