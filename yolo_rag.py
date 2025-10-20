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
import logging

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


# Configuration
PROCESS_INTERVAL = 60  # Process segmentation masks every N frames
DETECTION_CONF = 0.80
DETECTION_IOU = 0.5
MAX_DETECTIONS = 50
SEGS_DIR = "tmp_segs"
MODEL_PATH = "yolo11x-seg.pt"  # Segmentation model
WINDOW_SIZE = (1280, 720)
SAVE_SEGS_TO_DISK = False  # Toggle to save segmentation images/JSON to disk
SHOW_PERSON_NAMES = True  # Toggle to show recognized person names on video feed

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
        log.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    except redis.ConnectionError as e:
        log.warning(f"Could not connect to Redis: {e}")
        log.warning("Person embeddings will not be stored.")
        redis_client = None

    # Initialize embedding model
    log.info(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    log.info("Embedding model loaded successfully.")

    return redis_client, embedding_model


def setup_display(segs_dir, window_name="YOLO Segmentation", window_size=(1280, 720)):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_size[0], window_size[1])

    log.info("Press 'ctrl+c' to quit.")
    log.info(f"Saving segmentation masks to '{segs_dir}' every {PROCESS_INTERVAL} frames.")

    return window_name


def update_fps(frame_count, start_time, elapsed_threshold=1.0):
    elapsed_time = time.time() - start_time
    if elapsed_time > elapsed_threshold:
        fps = frame_count / elapsed_time
        return fps, 0, time.time()
    return None, frame_count, start_time


def recognize_persons_in_frame(results, frame, model):
    """Recognize persons in current frame and return their names/boxes."""
    person_annotations = []

    for result in results:
        if result.boxes is None or result.masks is None:
            continue

        boxes = result.boxes
        masks = result.masks

        for box, mask in zip(boxes, masks):
            # Only process persons
            class_id = int(box.cls.cpu())
            class_name = model.names[class_id]

            if class_name != 'person':
                continue

            # Get bounding box coordinates
            coords = box.xyxy.cpu().tolist()[0]
            x1, y1, x2, y2 = map(int, coords)

            # Get binary mask and extract person image
            mask_data = mask.data.cpu().numpy()[0]
            mask_resized = cv2.resize(mask_data, (frame.shape[1], frame.shape[0]))
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

            # Create masked image
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask_binary.astype(np.uint8))
            cropped_masked = masked_frame[y1:y2, x1:x2]

            if cropped_masked.size > 0:
                try:
                    # Convert to PIL Image for embedding
                    cropped_rgb = cv2.cvtColor(cropped_masked, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(cropped_rgb)

                    # Generate embedding and search
                    embedding = generate_embedding(pil_image)
                    matched_name, similarity = search_person_in_redis(embedding)

                    person_annotations.append({
                        'box': (x1, y1, x2, y2),
                        'name': matched_name if matched_name else 'Unknown',
                        'similarity': similarity,
                        'confidence': float(box.conf.cpu())
                    })
                except Exception as e:
                    # If recognition fails, mark as unknown
                    person_annotations.append({
                        'box': (x1, y1, x2, y2),
                        'name': 'Unknown',
                        'similarity': 0,
                        'confidence': float(box.conf.cpu())
                    })

    return person_annotations


def annotate_frame(results, fps, frame_count, person_annotations=None):
    annotated_frame = results[0].plot(
        line_width=2,
        font_size=1.0,
        labels=True,
        conf=True
    )

    # Add FPS counter
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f} | Frame: {frame_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Overlay person names if provided
    if person_annotations:
        for person in person_annotations:
            x1, y1, x2, y2 = person['box']
            name = person['name']
            similarity = person['similarity']

            # Create label with name and similarity
            if name != 'Unknown':
                label = f"{name} ({similarity:.0%})"
                color = (0, 255, 0)  # Green for recognized
            else:
                label = "Unknown"
                color = (0, 165, 255)  # Orange for unknown

            # Draw filled rectangle for text background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0] + 10, y1),
                color,
                -1
            )

            # Draw name text
            cv2.putText(
                annotated_frame,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 0),  # Black text
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
            log.debug(f"Deleted previous segmentation masks from '{segs_dir}'")
            return True
        except Exception as e:
            log.error(f"Error deleting segmentation directory: {e}")
            return False
    return True


def process_persons_from_results(results, frame, frame_number, model):
    """Extract persons from results and process them for Redis."""
    persons_to_process = []

    # Only create temporary directory if saving to disk
    if SAVE_SEGS_TO_DISK:
        temp_dir = os.path.join(SEGS_DIR, "temp_persons")
        os.makedirs(temp_dir, exist_ok=True)

    person_idx = 0
    for result in results:
        # Check if masks exist (segmentation model)
        if result.masks is None:
            continue

        boxes = result.boxes
        masks = result.masks

        for box, mask in zip(boxes, masks):
            # Only process persons
            class_id = int(box.cls.cpu())
            class_name = model.names[class_id]

            if class_name != 'person':
                continue

            # Get bounding box coordinates
            coords = box.xyxy.cpu().tolist()[0]
            x1, y1, x2, y2 = map(int, coords)

            # Get binary mask and resize to original frame dimensions
            mask_data = mask.data.cpu().numpy()[0]
            mask_resized = cv2.resize(mask_data, (frame.shape[1], frame.shape[0]))
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

            # Create masked image (object with transparent/black background)
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask_binary.astype(np.uint8))
            cropped_masked = masked_frame[y1:y2, x1:x2]

            if cropped_masked.size > 0:
                person_data = {
                    'seg_info': {
                        'confidence': float(box.conf.cpu()),
                        'class_name': class_name
                    },
                    'frame_number': frame_number
                }

                if SAVE_SEGS_TO_DISK:
                    # Save person image to disk
                    temp_img_path = os.path.join(temp_dir, f"person_{frame_number:06d}_{person_idx:03d}.jpg")
                    cv2.imwrite(temp_img_path, cropped_masked)
                    person_data['image_path'] = temp_img_path
                else:
                    # Convert to PIL Image and keep in memory
                    # OpenCV uses BGR, PIL uses RGB
                    cropped_rgb = cv2.cvtColor(cropped_masked, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(cropped_rgb)
                    person_data['image_data'] = pil_image

                persons_to_process.append(person_data)
                person_idx += 1

    # Process all persons found
    if persons_to_process:
        identify_and_store_persons(persons_to_process)


def save_detection_results(results, frame, frame_number, segs_dir, model):
    """Save all detection results to disk (optional based on SAVE_SEGS_TO_DISK)."""
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
    log.debug(f"Saved {len(segs_data)} segmentation masks and images to {frame_img_dir}")

    return len(segs_data)


def generate_embedding(image_input):
    """Generate an embedding vector from an image file path or PIL Image object.

    Args:
        image_input: Either a file path (str) or PIL Image object
    """
    global embedding_model

    if embedding_model is None:
        raise RuntimeError("Embedding model not initialized")

    # Handle both file paths and PIL Image objects
    if isinstance(image_input, str):
        # It's a file path
        img = Image.open(image_input)
    else:
        # It's already a PIL Image
        img = image_input

    # Generate embedding
    embedding = embedding_model.encode(img)

    return embedding


def store_person_in_redis(person_name, embedding, metadata=None):
    """Store person embedding in Redis with metadata."""
    global redis_client

    if redis_client is None:
        log.warning("Redis client not available. Cannot store person.")
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

        log.info(f"Stored person '{person_name}' in Redis with key '{person_key}'")
        return True

    except Exception as e:
        log.error(f"Error storing person in Redis: {e}")
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
        log.error(f"Error searching Redis: {e}")
        return None, 0


def identify_and_store_persons(persons_to_process):
    """Display person images, collect names, generate embeddings, and store in Redis."""
    if not persons_to_process:
        return

    log.debug(f"\nFound {len(persons_to_process)} person(s) to process.")

    for idx, person_data in enumerate(persons_to_process, 1):
        seg_info = person_data['seg_info']

        # Get image either from path or data
        if 'image_path' in person_data:
            image_path = person_data['image_path']
            # Check if image exists
            if not os.path.exists(image_path):
                log.debug(f"Could not find image: {image_path}")
                continue
            image_input = image_path
        elif 'image_data' in person_data:
            image_input = person_data['image_data']
        else:
            log.debug(f"No image data available for person {idx}")
            continue

        log.debug(f"\n[Person {idx}/{len(persons_to_process)}]")
        log.debug(f"Frame: {person_data['frame_number']}")

        try:
            # Generate embedding first for vector search
            log.debug("Generating embedding...")
            embedding = generate_embedding(image_input)

            # Search Redis for a match
            log.debug("Searching for matches in database...")
            matched_name, similarity = search_person_in_redis(embedding)

            if matched_name:
                # Found a match!
                log.debug(f"✓ Recognized as '{matched_name}' (similarity: {similarity:.2%})")
                log.debug(f"Skipping identification prompt and storage.")

            else:
                # No match found - ask user to identify
                if similarity > 0:
                    log.debug(f"No confident match found (best similarity: {similarity:.2%})")
                else:
                    log.debug("No previous persons in database.")

                # Convert image to OpenCV format for display
                if isinstance(image_input, str):
                    person_img_pil = Image.open(image_input)
                else:
                    person_img_pil = image_input

                # Convert PIL to OpenCV (RGB to BGR)
                person_img_cv = cv2.cvtColor(np.array(person_img_pil), cv2.COLOR_RGB2BGR)

                # Display image in OpenCV window
                window_id = f"Who is this? ({idx}/{len(persons_to_process)})"
                cv2.namedWindow(window_id, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_id, 400, 600)
                cv2.imshow(window_id, person_img_cv)

                # Give the window time to render properly (multiple refresh cycles)
                for _ in range(10):
                    cv2.waitKey(50)  # Wait 50ms per iteration

                log.debug(f"Image displayed in window...")

                # Get person name from user input
                person_name = input("Enter person's name (or press Enter to skip): ").strip()

                # Close the window immediately after input
                cv2.destroyWindow(window_id)
                cv2.waitKey(1)  # Ensure window is destroyed

                if person_name:
                    # Store in Redis with the user-provided name
                    metadata = {
                        'confidence': seg_info.get('confidence'),
                        'frame_number': person_data['frame_number']
                    }
                    # Only include image_path in metadata if it exists
                    if 'image_path' in person_data:
                        metadata['image_path'] = person_data['image_path']

                    store_person_in_redis(person_name, embedding, metadata)
                else:
                    log.debug("Skipped.")

        except Exception as e:
            log.error(f"Error processing person: {e}")

    log.debug(f"\nFinished processing {len(persons_to_process)} person(s).\n")


def process_segs(segs_dir, specific_frame_number=None):
    """Process saved segmentation JSON files and identify persons.

    Args:
        segs_dir: Directory containing segmentation data
        specific_frame_number: If provided, only process this specific frame
    """
    if not os.path.exists(segs_dir):
        log.debug(f"Segmentation directory '{segs_dir}' does not exist.")
        return

    # Track if we found any persons to process
    persons_to_process = []

    # If specific frame number provided, only process that one
    if specific_frame_number is not None:
        json_file = f"segs_frame_{specific_frame_number:06d}.json"
        json_path = os.path.join(segs_dir, json_file)

        if not os.path.exists(json_path):
            log.debug(f"Frame {specific_frame_number} not found")
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
            log.debug(f"No JSON files found in '{segs_dir}'")
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
            log.error("Failed to grab frame.")
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

        if total_frame_count % PROCESS_INTERVAL == 0:
            # Always process persons for Redis
            process_persons_from_results(results, frame, total_frame_count, model)

            # Optionally save all segmentation data to disk
            if SAVE_SEGS_TO_DISK:
                save_detection_results(results, frame, total_frame_count, SEGS_DIR, model)

        # Recognize persons in current frame for display (if enabled)
        person_annotations = None
        if SHOW_PERSON_NAMES:
            person_annotations = recognize_persons_in_frame(results, frame, model)

        annotated_frame = annotate_frame(results, fps, total_frame_count, person_annotations)

        cv2.imshow(window_name, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    log.info("Program terminated.")


if __name__ == "__main__":
    main()
