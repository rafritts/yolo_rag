import cv2
from ultralytics import YOLO
import time
import signal

running = True

def signal_handler(signum, frame):
    global running
    print("\nShutting down...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

def main():
    global running

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Loading models...")
    model_detect = YOLO("yolo11x.pt")
    model_segment = YOLO("yolo11x-seg.pt")
    model_pose = YOLO("yolo11x-pose.pt")

    model_detect.to("cuda")
    model_segment.to("cuda")
    model_pose.to("cuda")

    print("All models loaded on CUDA")

    window_detect = "YOLO Detection"
    window_segment = "YOLO Segmentation"
    window_pose = "YOLO Pose"

    cv2.namedWindow(window_detect, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_segment, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_pose, cv2.WINDOW_NORMAL)

    cv2.resizeWindow(window_detect, 1280, 720)
    cv2.resizeWindow(window_segment, 1280, 720)
    cv2.resizeWindow(window_pose, 1280, 720)

    print("Press 'Ctrl+C' to quit.")

    frame_count = 0
    start_time = time.time()
    fps = 0

    while running:
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

        results_detect = model_detect(
            frame,
            conf=0.50,      # Confidence threshold
            iou=0.5,        # Intersection over Union threshold
            max_det=50,     # Maximum detections per frame
            verbose=False
        )
        results_segment = model_segment(
            frame,
            conf=0.50,      # Confidence threshold
            iou=0.5,        # Intersection over Union threshold
            max_det=50,     # Maximum detections per frame
            verbose=False
        )
        results_pose = model_pose(
            frame,
            conf=0.50,      # Confidence threshold
            iou=0.5,        # Intersection over Union threshold
            max_det=50,     # Maximum detections per frame
            verbose=False
        )

        annotated_detect = results_detect[0].plot(
            line_width=2,
            font_size=1.0,
            labels=True,
            conf=True
        )

        annotated_segment = results_segment[0].plot(
            line_width=2,
            font_size=1.0,
            labels=True,
            conf=True,
            boxes=True,
            masks=True
        )

        annotated_pose = results_pose[0].plot(
            line_width=2,
            font_size=1.0,
            labels=True,
            conf=True,
            boxes=True,
            kpt_radius=5
        )

        fps_text = f"FPS: {fps:.1f}"
        for frame in [annotated_detect, annotated_segment, annotated_pose]:
            cv2.putText(
                frame,
                fps_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        cv2.imshow(window_detect, annotated_detect)
        cv2.imshow(window_segment, annotated_segment)
        cv2.imshow(window_pose, annotated_pose)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Program terminated.")

if __name__ == "__main__":
    main()
