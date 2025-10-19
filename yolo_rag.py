import cv2
from ultralytics import YOLO
import time
import os
import json


def main():
    cap = cv2.VideoCapture(0)

    model = YOLO("yolo11x.pt")
    model.to("cuda")

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Create tmp_boxes directory if it doesn't exist
    boxes_dir = "tmp_boxes"
    os.makedirs(boxes_dir, exist_ok=True)

    window_name = "YOLO Object Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("Press 'ctrl+c' to quit.")
    print(f"Saving bounding boxes to '{boxes_dir}' every 300 frames.")

    frame_count = 0
    total_frame_count = 0
    start_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to grab frame.")
            break

        frame_count += 1
        total_frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        results = model(
            frame,
            conf=0.50,      # Confidence threshold
            iou=0.5,        # Intersection over Union threshold
            max_det=50,     # Maximum detections per frame
            verbose=False
        )

        # Save boxes every 300 frames
        if total_frame_count % 300 == 0:
            boxes_data = []
            # Create subdirectory for this frame's images
            frame_img_dir = os.path.join(boxes_dir, f"frame_{total_frame_count:06d}")
            os.makedirs(frame_img_dir, exist_ok=True)

            box_idx = 0
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Extract box information
                    coords = box.xyxy.cpu().tolist()[0]  # [x1, y1, x2, y2]
                    box_info = {
                        "xyxy": coords,
                        "confidence": float(box.conf.cpu()),
                        "class_id": int(box.cls.cpu()),
                        "class_name": model.names[int(box.cls.cpu())],
                        "image_file": f"box_{box_idx:03d}.jpg"
                    }
                    boxes_data.append(box_info)

                    # Crop and save the box image
                    x1, y1, x2, y2 = map(int, coords)
                    cropped_img = frame[y1:y2, x1:x2]

                    if cropped_img.size > 0:  # Check if crop is valid
                        img_path = os.path.join(frame_img_dir, f"box_{box_idx:03d}.jpg")
                        cv2.imwrite(img_path, cropped_img)

                    box_idx += 1

            # Save to JSON file
            filename = os.path.join(boxes_dir, f"boxes_frame_{total_frame_count:06d}.json")
            with open(filename, 'w') as f:
                json.dump({
                    "frame": total_frame_count,
                    "timestamp": time.time(),
                    "boxes": boxes_data
                }, f, indent=2)
            print(f"Saved {len(boxes_data)} boxes and images to {frame_img_dir}")

        annotated_frame = results[0].plot(
            line_width=2,
            font_size=1.0,
            labels=True,
            conf=True
        )

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f} | Frame: {total_frame_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow(window_name, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Program terminated.")


if __name__ == "__main__":
    main()
