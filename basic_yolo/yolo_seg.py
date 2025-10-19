import cv2
from ultralytics import YOLO
import time


def main():
    cap = cv2.VideoCapture(0)

    model = YOLO("yolo11x-seg.pt")
    model.to("cuda")

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    window_name = "YOLO Segmentation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("Press 'ctrl+c' to quit.")

    frame_count = 0
    start_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to grab frame.")
            break

        frame_count += 1
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

        annotated_frame = results[0].plot(
            line_width=2,
            font_size=1.0,
            labels=True,
            conf=True,
            boxes=True,
            masks=True
        )

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
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
