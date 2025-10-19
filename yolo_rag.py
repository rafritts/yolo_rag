import cv2
from ultralytics import YOLO
import time
import os
import json
import shutil
import numpy as np
import redis
from PIL import Image
from sentence_transformers import SentenceTransformer


# Configuration
SAVE_INTERVAL = 60  # Save segmentation masks every N frames
DETECTION_CONF = 0.50
DETECTION_IOU = 0.5
MAX_DETECTIONS = 50
SEGS_DIR = "tmp_segs"
MODEL_PATH = "yolo11x-seg.pt"  # Segmentation model
WINDOW_SIZE = (1280, 720)

# Redis Configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = "clip-ViT-B-32"  # CLIP model for image embeddings

# Global variables for Redis and embedding model
redis_client = None
embedding_model = None


def setup_camera_and_model(camera_index=0):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError("Error: Could not open webcam.")

    model = YOLO(MODEL_PATH)
    model.to("cuda")

    return cap, model


def setup_redis_and_embeddings():
    """Initialize Redis connection and embedding model."""
    global redis_client, embedding_model

    # Initialize Redis client
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=False  # We'll store binary data
        )
        redis_client.ping()
        print(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    except redis.ConnectionError as e:
        print(f"Warning: Could not connect to Redis: {e}")
        print("Person embeddings will not be stored.")
        redis_client = None

    # Initialize embedding model
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model loaded successfully.")

    return redis_client, embedding_model


def setup_display(segs_dir, window_name="YOLO Segmentation", window_size=(1280, 720)):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_size[0], window_size[1])

    print("Press 'ctrl+c' to quit.")
    print(f"Saving segmentation masks to '{segs_dir}' every {SAVE_INTERVAL} frames.")

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


def extract_seg_info(box, mask, model, seg_idx):
    coords = box.xyxy.cpu().tolist()[0]  # [x1, y1, x2, y2]

    # Extract polygon coordinates from mask
    polygon = mask.xy[0].tolist() if mask is not None else []

    seg_info = {
        "xyxy": coords,
        "confidence": float(box.conf.cpu()),
        "class_id": int(box.cls.cpu()),
        "class_name": model.names[int(box.cls.cpu())],
        "polygon": polygon,  # Segmentation polygon coordinates
        "image_file": f"seg_{seg_idx:03d}.jpg"
    }
    return seg_info

def delete_previous_segs(segs_dir):
    if os.path.exists(segs_dir):
        try:
            shutil.rmtree(segs_dir)
            print(f"Deleted previous segmentation masks from '{segs_dir}'")
            return True
        except Exception as e:
            print(f"Error deleting segmentation directory: {e}")
            return False
    return True


def save_detection_results(results, frame, frame_number, segs_dir, model):
    segs_data = []
    frame_img_dir = os.path.join(segs_dir, f"frame_{frame_number:06d}")
    os.makedirs(frame_img_dir, exist_ok=True)

    seg_idx = 0
    for result in results:
        # Check if masks exist (segmentation model)
        if result.masks is None:
            continue

        boxes = result.boxes
        masks = result.masks

        for box, mask in zip(boxes, masks):
            seg_info = extract_seg_info(box, mask, model, seg_idx)
            segs_data.append(seg_info)

            # Get bounding box coordinates
            coords = seg_info["xyxy"]
            x1, y1, x2, y2 = map(int, coords)

            # Get binary mask and resize to original frame dimensions
            mask_data = mask.data.cpu().numpy()[0]  # Binary mask
            mask_resized = cv2.resize(mask_data, (frame.shape[1], frame.shape[0]))
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

            # Create masked image (object with transparent/black background)
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask_binary.astype(np.uint8))
            cropped_masked = masked_frame[y1:y2, x1:x2]

            if cropped_masked.size > 0:
                img_path = os.path.join(frame_img_dir, f"seg_{seg_idx:03d}.jpg")
                cv2.imwrite(img_path, cropped_masked)

            seg_idx += 1

    filename = os.path.join(segs_dir, f"segs_frame_{frame_number:06d}.json")
    with open(filename, 'w') as f:
        json.dump({
            "frame": frame_number,
            "timestamp": time.time(),
            "segmentations": segs_data
        }, f, indent=2)
    print(f"Saved {len(segs_data)} segmentation masks and images to {frame_img_dir}")
    process_segs(segs_dir, frame_number)

    return len(segs_data)


def generate_embedding(image_path):
    """Generate an embedding vector from an image file."""
    global embedding_model

    if embedding_model is None:
        raise RuntimeError("Embedding model not initialized")

    # Load image using PIL
    img = Image.open(image_path)

    # Generate embedding
    embedding = embedding_model.encode(img)

    return embedding


def store_person_in_redis(person_name, embedding, metadata=None):
    """Store person embedding in Redis with metadata."""
    global redis_client

    if redis_client is None:
        print("Redis client not available. Cannot store person.")
        return False

    try:
        # Create a unique key for this person
        person_key = f"person:{person_name}:{int(time.time())}"

        # Store embedding as bytes
        embedding_bytes = embedding.tobytes()

        # Create metadata dictionary
        full_metadata = {
            "name": person_name,
            "timestamp": time.time(),
            "embedding_shape": str(embedding.shape),
            "embedding_dtype": str(embedding.dtype)
        }

        if metadata:
            full_metadata.update(metadata)

        # Store in Redis using a hash
        redis_client.hset(person_key, "embedding", embedding_bytes)
        redis_client.hset(person_key, "metadata", json.dumps(full_metadata))

        print(f"Stored person '{person_name}' in Redis with key '{person_key}'")
        return True

    except Exception as e:
        print(f"Error storing person in Redis: {e}")
        return False


def search_person_in_redis(query_embedding, similarity_threshold=0.90):
    global redis_client

    if redis_client is None:
        return None, 0

    try:
        # Get all person keys from Redis
        person_keys = redis_client.keys("person:*")

        if not person_keys:
            return None, 0

        best_match_name = None
        best_similarity = 0

        for key in person_keys:
            # Get the stored embedding and metadata
            embedding_bytes = redis_client.hget(key, "embedding")
            metadata_json = redis_client.hget(key, "metadata")

            if not embedding_bytes or not metadata_json:
                continue

            metadata = json.loads(metadata_json)

            # Reconstruct the numpy array from bytes
            stored_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

            # Calculate cosine similarity
            # cosine_similarity = dot(A, B) / (norm(A) * norm(B))
            dot_product = np.dot(query_embedding, stored_embedding)
            norm_query = np.linalg.norm(query_embedding)
            norm_stored = np.linalg.norm(stored_embedding)

            if norm_query == 0 or norm_stored == 0:
                continue

            similarity = dot_product / (norm_query * norm_stored)

            # Track the best match
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_name = metadata.get('name')

        # Return the best match if it's above threshold
        if best_similarity >= similarity_threshold:
            return best_match_name, best_similarity
        else:
            return None, best_similarity

    except Exception as e:
        print(f"Error searching Redis: {e}")
        return None, 0


def identify_and_store_persons(persons_to_process):
    """Display person images, collect names, generate embeddings, and store in Redis."""
    if not persons_to_process:
        return

    print(f"\nFound {len(persons_to_process)} person(s) to process.")

    for idx, person_data in enumerate(persons_to_process, 1):
        image_path = person_data['image_path']
        seg_info = person_data['seg_info']

        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Could not find image: {image_path}")
            continue

        print(f"\n[Person {idx}/{len(persons_to_process)}]")
        print(f"Frame: {person_data['frame_number']}")

        try:
            # Generate embedding first for vector search
            print("Generating embedding...")
            embedding = generate_embedding(image_path)

            # Search Redis for a match
            print("Searching for matches in database...")
            matched_name, similarity = search_person_in_redis(embedding)

            if matched_name:
                # Found a match!
                print(f"✓ Recognized as '{matched_name}' (similarity: {similarity:.2%})")
                print(f"Skipping identification prompt and storage.")

            else:
                # No match found - ask user to identify
                if similarity > 0:
                    print(f"No confident match found (best similarity: {similarity:.2%})")
                else:
                    print("No previous persons in database.")

                # Open and display the person image using PIL
                person_img = Image.open(image_path)
                print(f"Opening image in default viewer: {image_path}")
                person_img.show(title=f"Who is this? ({idx}/{len(persons_to_process)})")

                # Get person name from user input
                person_name = input("Enter person's name (or press Enter to skip): ").strip()

                if person_name:
                    # Store in Redis with the user-provided name
                    metadata = {
                        'confidence': seg_info.get('confidence'),
                        'frame_number': person_data['frame_number'],
                        'image_path': image_path
                    }
                    store_person_in_redis(person_name, embedding, metadata)
                else:
                    print("Skipped.")

        except Exception as e:
            print(f"Error processing person: {e}")

    print(f"\nFinished processing {len(persons_to_process)} person(s).\n")


def process_segs(segs_dir, specific_frame_number=None):
    """Process saved segmentation JSON files and identify persons.

    Args:
        segs_dir: Directory containing segmentation data
        specific_frame_number: If provided, only process this specific frame
    """
    if not os.path.exists(segs_dir):
        print(f"Segmentation directory '{segs_dir}' does not exist.")
        return

    # Track if we found any persons to process
    persons_to_process = []

    # If specific frame number provided, only process that one
    if specific_frame_number is not None:
        json_file = f"segs_frame_{specific_frame_number:06d}.json"
        json_path = os.path.join(segs_dir, json_file)

        if not os.path.exists(json_path):
            print(f"Frame {specific_frame_number} not found")
            return

        with open(json_path, 'r') as f:
            data = json.load(f)

        frame_number = data.get('frame', 0)
        frame_dir = os.path.join(segs_dir, f"frame_{frame_number:06d}")

        # Check each segmentation in the frame
        for seg in data.get('segmentations', []):
            if seg.get('class_name') == 'person':
                # Get the image path for this person
                image_file = seg.get('image_file')
                image_path = os.path.join(frame_dir, image_file)

                if os.path.exists(image_path):
                    persons_to_process.append({
                        'image_path': image_path,
                        'seg_info': seg,
                        'frame_number': frame_number
                    })
    else:
        # Process all JSON files (original behavior)
        json_files = sorted([f for f in os.listdir(segs_dir) if f.endswith('.json')])

        if not json_files:
            print(f"No JSON files found in '{segs_dir}'")
            return

        for json_file in json_files:
            json_path = os.path.join(segs_dir, json_file)

            with open(json_path, 'r') as f:
                data = json.load(f)

            frame_number = data.get('frame', 0)
            frame_dir = os.path.join(segs_dir, f"frame_{frame_number:06d}")

            # Check each segmentation in the frame
            for seg in data.get('segmentations', []):
                if seg.get('class_name') == 'person':
                    # Get the image path for this person
                    image_file = seg.get('image_file')
                    image_path = os.path.join(frame_dir, image_file)

                    if os.path.exists(image_path):
                        persons_to_process.append({
                            'image_path': image_path,
                            'seg_info': seg,
                            'frame_number': frame_number
                        })

    # Process all persons found
    identify_and_store_persons(persons_to_process)


def main():
    cap, model = setup_camera_and_model()
    setup_redis_and_embeddings()
    os.makedirs(SEGS_DIR, exist_ok=True)
    #delete_previous_segs(SEGS_DIR)
    window_name = setup_display(SEGS_DIR, window_size=WINDOW_SIZE)

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
            save_detection_results(results, frame, total_frame_count, SEGS_DIR, model)

        annotated_frame = annotate_frame(results, fps, total_frame_count)

        cv2.imshow(window_name, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Program terminated.")


if __name__ == "__main__":
    main()
