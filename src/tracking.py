# tracking.py

# We import YOLO from ultralytics because YOLO is the model class
# that will load our trained weights and perform tracking.
from ultralytics import YOLO


class ObjectTracker:
    # This class is responsible for:
    # 1. loading the trained YOLO model
    # 2. running tracking on a single frame
    # 3. returning tracking results in a clean Python format

    def __init__(self, model_path, classes_to_track=None, conf=0.25, tracker_config="bytetrack.yaml"):
        # model_path:
        # path to the YOLO model weights
        # example: "runs/detect/train/weights/best.pt"

        # classes_to_track:
        # optional list of class IDs to track
        # example: [0, 1, 2, 3, 4, 5]
        # if None, then all classes in the model are used

        # conf:
        # confidence threshold
        # detections below this score will be ignored

        # tracker_config:
        # tracker algorithm config file
        # "bytetrack.yaml" is a very good starting choice for MOT projects

        # Load the YOLO model from the given path.
        # This creates the model object in memory.
        self.model = YOLO(model_path)

        # Save the model path as an attribute.
        # This is useful for debugging or printing later if needed.
        self.model_path = model_path

        # Save the selected classes.
        # We keep this so the tracker knows which object classes to track.
        self.classes_to_track = classes_to_track

        # Save the confidence threshold.
        # This lets us control how strict detection/tracking should be.
        self.conf = conf

        # Save the tracker config name.
        # We pass this later into model.track().
        self.tracker_config = tracker_config

        # model.names is a dictionary created by Ultralytics.
        # It maps class IDs to class names.
        # Example:
        # {0: 'person', 1: 'bicycle', 2: 'car', ...}
        self.class_names = self.model.names

    def track_frame(self, frame):
        # This function accepts ONE video frame as input.
        # Then it runs YOLO tracking on that frame.
        # Finally, it returns a list of tracked objects.

        # Run tracking on the current frame.
        # persist=True means the tracker remembers objects from previous frames
        # and tries to keep the same track ID over time.
        #
        # source/frame:
        # we pass the current image frame directly
        #
        # conf=self.conf:
        # use the confidence threshold we stored earlier
        #
        # classes=self.classes_to_track:
        # only track selected classes if provided
        #
        # tracker=self.tracker_config:
        # tells YOLO which tracker to use, such as ByteTrack
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            classes=self.classes_to_track,
            tracker=self.tracker_config,
            verbose=False
        )

        # Create an empty list where we will store all tracked objects
        # from the current frame.
        tracked_objects = []

        # Sometimes results may be empty for a frame.
        # If no results are returned, we safely return an empty list.
        if not results:
            return tracked_objects

        # results[0] is the first result object because we passed one frame.
        result = results[0]

        # If there are no detected boxes in this frame,
        # then there is nothing to parse.
        if result.boxes is None or len(result.boxes) == 0:
            return tracked_objects

        # Extract bounding boxes in xyxy format:
        # x1, y1 = top-left corner
        # x2, y2 = bottom-right corner
        #
        # .cpu() moves tensor data from GPU to CPU so Python can use it easily.
        boxes = result.boxes.xyxy.cpu().tolist()

        # Extract confidence scores for each detection.
        # Example: 0.91, 0.84, etc.
        confidences = result.boxes.conf.cpu().tolist()

        # Extract predicted class IDs for each detection.
        # Example: 0 for person, 2 for car, etc.
        class_ids = result.boxes.cls.int().cpu().tolist()

        # Extract tracking IDs.
        # Important:
        # Sometimes track IDs may be missing, especially at startup.
        # So we must handle that case safely.
        if result.boxes.id is not None:
            # Convert track IDs to normal Python integers.
            track_ids = result.boxes.id.int().cpu().tolist()
        else:
            # If track IDs are missing, create a list of None values
            # with the same length as the number of boxes.
            track_ids = [None] * len(boxes)

        # Loop through each detected/tracked object in the current frame.
        # zip(...) combines the matching bbox, confidence, class_id, and track_id.
        for box, confidence, class_id, track_id in zip(boxes, confidences, class_ids, track_ids):
            # Convert box coordinates to integers because pixel positions
            # should be whole numbers for drawing and center calculation.
            x1, y1, x2, y2 = map(int, box)

            # Compute object center point.
            # This is useful later for:
            # - drawing center dots
            # - line crossing
            # - speed estimation
            # - trajectory tracking
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Convert class ID into readable class name.
            # Example: 2 -> "car"
            class_name = self.class_names[class_id]

            # Create one clean dictionary for this tracked object.
            # A dictionary makes downstream code easier to read.
            obj = {
                "bbox": [x1, y1, x2, y2],      # bounding box coordinates
                "center": [cx, cy],            # center point of the object
                "confidence": float(confidence),  # detection confidence score
                "class_id": int(class_id),     # numeric class ID
                "class_name": class_name,      # readable class name
                "track_id": track_id           # unique ID assigned by tracker
            }

            # Add this object's dictionary to our output list.
            tracked_objects.append(obj)

        # Return the final list of all tracked objects for this frame.
        return tracked_objects

    def get_class_names(self):
        # This small helper function returns the class-name mapping.
        # It can be useful in other files if you want to inspect the labels.
        return self.class_names