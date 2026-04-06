

# main.py

# OpenCV is used for reading video, showing frames, and saving output video.
import cv2

# time is used to calculate live processing FPS.
import time

# math is used to calculate pixel distance for scene calibration.
import math

# Path helps us build safe file paths relative to the project folder.
from pathlib import Path

# Import tracker class from tracking.py
from tracking import ObjectTracker

# Import analytics manager from analytics.py
from analytics import AnalyticsManager

# Import drawing/helper functions from utils.py
from utils import (
    draw_all_tracked_objects,
    draw_count_panel,
    draw_label,
    draw_all_trajectory_trails,
    draw_horizontal_line
)


def calculate_meters_per_pixel(point1, point2, real_distance_meters):
    # This function computes how many real-world meters correspond to one pixel.
    # We use two points in the image whose real-world distance is known.
    #
    # Example:
    # point1 = (300, 500)
    # point2 = (700, 500)
    # real_distance_meters = 12

    # Unpack first point coordinates.
    x1, y1 = point1

    # Unpack second point coordinates.
    x2, y2 = point2

    # Compute pixel distance between the two points.
    pixel_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Prevent division by zero.
    if pixel_distance == 0:
        raise ValueError("Calibration points must not be identical.")

    # Return meters represented by one pixel.
    return real_distance_meters / pixel_distance


def main():
    # --------------------------------------------------
    # 1. BUILD PROJECT ROOT PATH
    # --------------------------------------------------

    # __file__ = path of this file (src/main.py)
    # .resolve() -> absolute path
    # .parent -> src/
    # .parent.parent -> project root folder
    BASE_DIR = Path(__file__).resolve().parent.parent

    # --------------------------------------------------
    # 2. USER CONFIGURATION
    # --------------------------------------------------

    # Path to trained YOLO model.
    model_path = BASE_DIR / "models" / "best.pt"

    # Path to test video.
    video_path = BASE_DIR / "test_videos" / "0.mp4"

    # Output folder path.
    output_dir = BASE_DIR / "outputs"

    # Create outputs folder if it does not exist.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Final output video path.
    output_path = output_dir / "tracked_output_final_kmh.mp4"

    # Track these class IDs from your custom dataset.
    # 0 = person
    # 1 = bicycle
    # 2 = car
    # 3 = motorcycle
    # 4 = bus
    # 5 = truck
    classes_to_track = [0, 1, 2, 3, 4, 5]

    # Detection confidence threshold.
    conf_threshold = 0.25

    # Maximum number of stored center points for trajectory trails.
    max_history = 10

    # --------------------------------------------------
    # 3. SPEED CALIBRATION SETTINGS
    # --------------------------------------------------
    #
    # IMPORTANT:
    # Replace these with points from YOUR own video.
    # These two points should lie on a road segment (or any scene segment)
    # where you know the real-world distance.
    #
    # Example:
    # If the distance between these two image points corresponds to 12 meters
    # in the real world, we use that to estimate meters_per_pixel.
    #
    # You must adjust these values for your scene.

    reference_point_1 = (300, 500)
    reference_point_2 = (700, 500)
    real_distance_meters = 12.0

    # Compute scene scale for speed estimation.
    meters_per_pixel = calculate_meters_per_pixel(
        reference_point_1,
        reference_point_2,
        real_distance_meters
    )

    # --------------------------------------------------
    # 4. OPTIONAL PATH CHECKS
    # --------------------------------------------------

    # Check that model file exists.
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return

    # Check that video file exists.
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return

    # --------------------------------------------------
    # 5. INITIALIZE TRACKER
    # --------------------------------------------------

    # Create YOLO tracker object.
    tracker = ObjectTracker(
        model_path=str(model_path),
        classes_to_track=classes_to_track,
        conf=conf_threshold,
        tracker_config="bytetrack.yaml"
    )

    # --------------------------------------------------
    # 6. GET CLASS NAMES
    # --------------------------------------------------

    # Get class ID -> class name mapping from the model.
    class_name_mapping = tracker.get_class_names()

    # Convert mapping into a list ordered by class ID.
    class_names = [class_name_mapping[i] for i in sorted(class_name_mapping.keys())]

    # --------------------------------------------------
    # 7. INITIALIZE ANALYTICS
    # --------------------------------------------------

    # Create analytics manager with scene calibration.
    analytics = AnalyticsManager(
        class_names=class_names,
        max_history=max_history,
        meters_per_pixel=meters_per_pixel
    )

    # --------------------------------------------------
    # 8. OPEN VIDEO
    # --------------------------------------------------

    # Open input video.
    cap = cv2.VideoCapture(str(video_path))

    # Check if video opened successfully.
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    # --------------------------------------------------
    # 9. READ VIDEO PROPERTIES
    # --------------------------------------------------

    # Get frame width.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Get frame height.
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get original video FPS.
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    # Fallback if FPS is invalid.
    if video_fps == 0:
        video_fps = 30

    # --------------------------------------------------
    # 10. CREATE VIDEO WRITER
    # --------------------------------------------------

    # Codec for mp4 output.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Create writer object for saving processed video.
    out = cv2.VideoWriter(str(output_path), fourcc, video_fps, (frame_width, frame_height))

    # --------------------------------------------------
    # 11. FPS VARIABLES
    # --------------------------------------------------

    # Stores previous timestamp for FPS calculation.
    previous_time = 0.0

    # Stores current timestamp.
    current_time = 0.0

    # Live processing FPS value.
    live_fps = 0.0

    # --------------------------------------------------
    # 12. PROCESS VIDEO FRAME BY FRAME
    # --------------------------------------------------

    while cap.isOpened():
        # Read one frame from the input video.
        ret, frame = cap.read()

        # Stop if no more frames are available.
        if not ret:
            break

        # Record current time for live FPS measurement.
        current_time = time.time()

        # --------------------------------------
        # 12A. TRACK OBJECTS
        # --------------------------------------

        # Run tracking on current frame.
        tracked_objects = tracker.track_frame(frame)

        # --------------------------------------
        # 12B. UPDATE ANALYTICS
        # --------------------------------------

        # Update counts, trails, and speed estimation using video FPS.
        analytics.update(tracked_objects, video_fps)

        # Get class-wise counts.
        total_counts = analytics.get_class_counts()

        # Get total unique object count.
        total_unique_count = analytics.get_total_unique_count()

        # Get all trajectory histories.
        all_track_histories = analytics.get_all_track_histories()

        # Get all speed estimates in km/h.
        all_speeds_kmh = analytics.get_all_speed_kmh()

        # --------------------------------------
        # 12C. DRAW OPTIONAL CALIBRATION LINE
        # --------------------------------------

        # Draw calibration/reference line on the frame so you can see
        # which scene segment is being used for speed scaling.
        #cv2.line(frame, reference_point_1, reference_point_2, (255, 255, 0), 2)

        # Draw text describing the reference distance.
        #draw_label(
           # frame=frame,
            #text=f"Reference: {real_distance_meters:.1f} m",
            #position=(reference_point_1[0], max(reference_point_1[1] - 10, 20)),
            #color=(255, 255, 0),
            #font_scale=0.6,
            #thickness=2
       # )

        # --------------------------------------
        # 12D. DRAW TRAJECTORY TRAILS
        # --------------------------------------

        # Draw trails first so boxes and labels stay visible on top.
        draw_all_trajectory_trails(frame, tracked_objects, all_track_histories)

        # --------------------------------------
        # 12E. DRAW TRACKED OBJECTS + KM/H
        # --------------------------------------

        # Draw all tracked objects and show speed labels like:
        # ID:7 car | 24.3 km/h
        draw_all_tracked_objects(frame, tracked_objects, speed_dict=all_speeds_kmh)

        # --------------------------------------
        # 12F. DRAW COUNT PANEL
        # --------------------------------------

        # Draw class-wise counts in top-left.
        draw_count_panel(
            frame=frame,
            total_counts=total_counts,
            start_x=10,
            start_y=30,
            line_gap=30,
            color=(255, 255, 255)
        )

        # Draw total unique object count near bottom-left.
        draw_label(
            frame=frame,
            text=f"Total Unique Objects: {total_unique_count}",
            position=(10, frame_height - 20),
            color=(255, 255, 255),
            font_scale=0.8,
            thickness=2
        )

        # --------------------------------------
        # 12G. CALCULATE LIVE FPS
        # --------------------------------------

        # Time taken for one loop iteration.
        time_diff = current_time - previous_time

        # Avoid division by zero.
        if time_diff > 0:
            current_fps = 1 / time_diff

            # Smooth the FPS so the number does not jump too much.
            if live_fps == 0.0:
                live_fps = current_fps
            else:
                live_fps = 0.9 * live_fps + 0.1 * current_fps

        # Update previous time for next iteration.
        previous_time = current_time

        # --------------------------------------
        # 12H. DRAW LIVE FPS
        # --------------------------------------

        # Draw live processing FPS at top-right.
        draw_label(
            frame=frame,
            text=f"FPS: {live_fps:.2f}",
            position=(frame_width - 150, 30),
            color=(0, 255, 255),
            font_scale=0.7,
            thickness=2
        )

        # --------------------------------------
        # 12I. SHOW AND SAVE FRAME
        # --------------------------------------

        # Show processed frame.
        cv2.imshow("Tracking + Trajectory Trails + KM/H Speed", frame)

        # Save processed frame to output video.
        out.write(frame)

        # Stop early if user presses q.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --------------------------------------------------
    # 13. RELEASE RESOURCES
    # --------------------------------------------------

    # Release input video.
    cap.release()

    # Release output video writer.
    out.release()

    # Close all OpenCV windows.
    cv2.destroyAllWindows()

    # Print completion message.
    print("Video processing complete.")
    print(f"Output saved to: {output_path}")


# Run main() only when this file is executed directly.
if __name__ == "__main__":
    main()