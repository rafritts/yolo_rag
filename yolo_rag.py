import cv2
from ultralytics import YOLO
import time
import os
import json
import shutil


# Configuration
SAVE_INTERVAL = 60  # Save boxes every N frames
DETECTION_CONF = 0.50
DETECTION_IOU = 0.5
MAX_DETECTIONS = 50
BOXES_DIR = "tmp_boxes"
MODEL_PATH = "yolo11x.pt"
WINDOW_SIZE = (1280, 720)


def setup_camera_and_model(camera_index=0):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError("Error: Could not open webcam.")

    model = YOLO(MODEL_PATH)
    model.to("cuda")

    return cap, model


def setup_display(boxes_dir, window_name="YOLO Object Detection", window_size=(1280, 720)):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_size[0], window_size[1])

    print("Press 'ctrl+c' to quit.")
    print(f"Saving bounding boxes to '{boxes_dir}' every {SAVE_INTERVAL} frames.")

    return window_name


def update_fps(frame_count, start_time, elapsed_threshold=1.0):
    elapsed_time = time.time() - start_time
    if elapsed_time > elapsed_threshold:
        fps = frame_count / elapsed_time
        return fps, 0, time.time()
    return None, frame_count, start_time


def annotate_frame(results, fps, frame_count):
    annotated_frame = results[0].plot(
        line_width=2,
        font_size=1.0,
        labels=True,
        conf=True
    )

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f} | Frame: {frame_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    return annotated_frame


def extract_box_info(box, model, box_idx):
    coords = box.xyxy.cpu().tolist()[0]  # [x1, y1, x2, y2]
    box_info = {
        "xyxy": coords,
        "confidence": float(box.conf.cpu()),
        "class_id": int(box.cls.cpu()),
        "class_name": model.names[int(box.cls.cpu())],
        "image_file": f"box_{box_idx:03d}.jpg"
    }
    return box_info

def delete_previous_boxes(boxes_dir):
    if os.path.exists(boxes_dir):
        try:
            shutil.rmtree(boxes_dir)
            print(f"Deleted previous boxes from '{boxes_dir}'")
            return True
        except Exception as e:
            print(f"Error deleting boxes directory: {e}")
            return False
    return True


def save_detection_results(results, frame, frame_number, boxes_dir, model):
    boxes_data = []
    frame_img_dir = os.path.join(boxes_dir, f"frame_{frame_number:06d}")
    os.makedirs(frame_img_dir, exist_ok=True)

    box_idx = 0
    for result in results:
        boxes = result.boxes
        for box in boxes:
            box_info = extract_box_info(box, model, box_idx)
            boxes_data.append(box_info)

            coords = box_info["xyxy"]
            x1, y1, x2, y2 = map(int, coords)
            cropped_img = frame[y1:y2, x1:x2]

            if cropped_img.size > 0:
                img_path = os.path.join(frame_img_dir, f"box_{box_idx:03d}.jpg")
                cv2.imwrite(img_path, cropped_img)

            box_idx += 1

    filename = os.path.join(boxes_dir, f"boxes_frame_{frame_number:06d}.json")
    with open(filename, 'w') as f:
        json.dump({
            "frame": frame_number,
            "timestamp": time.time(),
            "boxes": boxes_data
        }, f, indent=2)
    print(f"Saved {len(boxes_data)} boxes and images to {frame_img_dir}")

    return len(boxes_data)


def main():
    cap, model = setup_camera_and_model()
    os.makedirs(BOXES_DIR, exist_ok=True)
    delete_previous_boxes(BOXES_DIR)
    window_name = setup_display(BOXES_DIR, window_size=WINDOW_SIZE)

    frame_count = 0
    total_frame_count = 0
    start_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to grab frame.")
            break

        total_frame_count += 1
        frame_count += 1

        new_fps, frame_count, start_time = update_fps(frame_count, start_time)
        if new_fps is not None:
            fps = new_fps

        results = model(
            frame,
            conf=DETECTION_CONF,
            iou=DETECTION_IOU,
            max_det=MAX_DETECTIONS,
            verbose=False
        )

        if total_frame_count % SAVE_INTERVAL == 0:
            save_detection_results(results, frame, total_frame_count, BOXES_DIR, model)

        annotated_frame = annotate_frame(results, fps, total_frame_count)

        cv2.imshow(window_name, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Program terminated.")


if __name__ == "__main__":
    main()
