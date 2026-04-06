import io
import os
import sys
import numpy as np
import cv2

# ---------------------------------------------------------
# PATH ROUTING: Allow importing from the 'src' folder
# ---------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Now we can cleanly import your custom logic
from src.tracking import ObjectTracker
from src.analytics import AnalyticsManager

# ---------------------------------------------------------
# GLOBAL STATE INITIALIZATION
# ---------------------------------------------------------
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best.pt")

# We keep these global so they remember history across API frames
tracker = None
analytics = None

def initialize_models():
    """Initializes or resets the tracker and analytics state."""
    global tracker, analytics
    
    # 1. Initialize Tracker
    tracker = ObjectTracker(
        model_path=MODEL_PATH,
        classes_to_track=[0, 1, 2, 3, 4, 5],
        conf=0.25,
        tracker_config="bytetrack.yaml"
    )
    
    # 2. Get sorted class names
    class_mapping = tracker.get_class_names()
    class_names = [class_mapping[i] for i in sorted(class_mapping.keys())]
    
    # 3. Calculate meters_per_pixel exactly like your src/main.py
    # reference_point_1 = (300, 500), reference_point_2 = (700, 500)
    # pixel distance = 400. Real distance = 12m.
    meters_per_pixel = 12.0 / 400.0 
    
    # 4. Initialize Analytics
    analytics = AnalyticsManager(
        class_names=class_names,
        max_history=10,
        meters_per_pixel=meters_per_pixel
    )

# Run initialization on startup
initialize_models()

def process_image_frame(image_bytes: bytes, video_fps: float = 30.0):
    """
    Converts bytes to OpenCV format, runs your custom tracking, 
    and returns a clean JSON with speeds and IDs.
    """
    # 1. Convert bytes to OpenCV image (BGR format)
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        raise ValueError("Could not decode image bytes into OpenCV frame.")

    # 2. Run tracking using your custom class
    tracked_objects = tracker.track_frame(frame)

    # 3. Update analytics (speed, trails, counts)
    analytics.update(tracked_objects, video_fps)
    
    # 4. Format the output for the Streamlit frontend
    detections = []
    for obj in tracked_objects:
        track_id = obj["track_id"]
        
        # Safely fetch speed from your analytics manager
        speed = 0.0
        if track_id is not None:
            speed = analytics.get_speed_kmh(track_id)
            
        detections.append({
            "class_name": obj["class_name"],
            "confidence": round(obj["confidence"], 2),
            "bbox": obj["bbox"],     # [x1, y1, x2, y2]
            "center": obj["center"], # [cx, cy]
            "tracking_id": track_id,
            "speed_kmh": round(speed, 2)
        })
        
    return {
        "detections": detections,
        "summary": analytics.get_summary()
    }