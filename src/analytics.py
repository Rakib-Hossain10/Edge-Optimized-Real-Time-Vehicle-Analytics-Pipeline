

# analytics.py

from collections import defaultdict
import math


class AnalyticsManager:
    # This class handles:
    # 1. unique counting
    # 2. class-wise counting
    # 3. trajectory history
    # 4. approximate speed estimation in km/h using scene calibration

    def __init__(self, class_names, max_history=30, meters_per_pixel=0.05):
        # class_names:
        # list of class names from the model
        self.class_names = class_names

        # max_history:
        # number of recent center points to keep for trajectory trails
        self.max_history = max_history

        # meters_per_pixel:
        # conversion factor from pixel distance to real-world meters
        # this is the key calibration value
        #
        # IMPORTANT:
        # This should ideally be computed from a known distance in the scene,
        # not guessed randomly.
        self.meters_per_pixel = meters_per_pixel

        # Store already-counted unique track IDs
        self.counted_ids = set()

        # Store class-wise unique counts
        self.total_counts = {class_name: 0 for class_name in self.class_names}

        # Store total unique count
        self.total_unique_objects = 0

        # Store center-point history for each track ID
        self.track_history = defaultdict(list)

        # Store latest speed for each track ID in km/h
        self.speed_kmh = {}

    def calculate_distance(self, point1, point2):
        # Compute Euclidean distance between two points
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def update(self, tracked_objects, video_fps):
        # Update counts, trails, and speed estimates for current frame

        for obj in tracked_objects:
            # Read track ID
            track_id = obj["track_id"]

            # Read class name
            class_name = obj["class_name"]

            # Read center point
            center = obj["center"]

            # Skip objects without stable ID
            if track_id is None:
                continue

            # -----------------------------
            # 1. UNIQUE COUNTING
            # -----------------------------
            if track_id not in self.counted_ids:
                self.counted_ids.add(track_id)

                if class_name in self.total_counts:
                    self.total_counts[class_name] += 1

                self.total_unique_objects += 1

            # -----------------------------
            # 2. STORE CENTER HISTORY
            # -----------------------------
            center_point = (center[0], center[1])
            self.track_history[track_id].append(center_point)

            # Keep only recent history points
            if len(self.track_history[track_id]) > self.max_history:
                self.track_history[track_id].pop(0)

            # -----------------------------
            # 3. SPEED ESTIMATION IN KM/H
            # -----------------------------
            # We need at least 2 points to estimate motion
            if len(self.track_history[track_id]) >= 2:
                prev_point = self.track_history[track_id][-2]
                curr_point = self.track_history[track_id][-1]

                # Pixel movement between last two frames
                pixel_distance = self.calculate_distance(prev_point, curr_point)

                # Convert per-frame movement to pixels per second
                pixel_speed_per_second = pixel_distance * video_fps

                # Convert pixels/second to meters/second
                meters_per_second = pixel_speed_per_second * self.meters_per_pixel

                # Convert m/s to km/h
                kmh = meters_per_second * 3.6

                # Optional small smoothing:
                # if a previous speed exists, blend old and new speed
                if track_id in self.speed_kmh:
                    previous_kmh = self.speed_kmh[track_id]
                    kmh = 0.7 * previous_kmh + 0.3 * kmh

                self.speed_kmh[track_id] = kmh

    def get_class_counts(self):
        return self.total_counts

    def get_total_unique_count(self):
        return self.total_unique_objects

    def get_track_history(self, track_id):
        return self.track_history.get(track_id, [])

    def get_all_track_histories(self):
        return self.track_history

    def get_speed_kmh(self, track_id):
        # Return latest approximate km/h for this object
        return self.speed_kmh.get(track_id, 0.0)

    def get_all_speed_kmh(self):
        return self.speed_kmh

    def reset(self):
        self.counted_ids.clear()
        self.total_counts = {class_name: 0 for class_name in self.class_names}
        self.total_unique_objects = 0
        self.track_history.clear()
        self.speed_kmh.clear()

    def get_summary(self):
        return {
            "total_unique_objects": self.total_unique_objects,
            "class_counts": self.total_counts,
            "tracks_with_speed": len(self.speed_kmh)
        }