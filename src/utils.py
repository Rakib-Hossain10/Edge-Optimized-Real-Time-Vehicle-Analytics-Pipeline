

# utils.py

import cv2


# This dictionary stores one display color for each class name.
# OpenCV uses BGR color format, not RGB.
CLASS_COLORS = {
    "person": (255, 0, 0),        # Blue
    "bicycle": (0, 255, 255),     # Yellow
    "car": (0, 255, 0),           # Green
    "motorcycle": (255, 0, 255),  # Purple
    "bus": (0, 165, 255),         # Orange
    "truck": (0, 0, 255)          # Red
}


def get_class_color(class_name):
    # Convert class name to lowercase so matching remains safe
    # even if model output uses different capitalization.
    class_name = class_name.lower()

    # Return the class-specific color if found.
    # Otherwise return white as a fallback.
    return CLASS_COLORS.get(class_name, (255, 255, 255))


def draw_bbox(frame, bbox, color=(0, 255, 0), thickness=2):
    # Draw one bounding box on the frame.
    # bbox format: [x1, y1, x2, y2]
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_center_point(frame, center, color=(0, 0, 255), radius=4):
    # Draw a filled circle at the object center.
    # center format: [cx, cy] or (cx, cy)
    cx, cy = center
    cv2.circle(frame, (cx, cy), radius, color, -1)


def draw_label(frame, text, position, color=(0, 255, 255), font_scale=0.6, thickness=2):
    # Draw text on the frame at the given position.
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness
    )


def draw_count_panel(frame, total_counts, start_x=10, start_y=30, line_gap=30, color=(0, 255, 0)):
    # Draw class-wise count text on the frame.
    y = start_y

    for class_name, count in total_counts.items():
        text = f"{class_name}: {count}"
        draw_label(
            frame=frame,
            text=text,
            position=(start_x, y),
            color=color,
            font_scale=0.7,
            thickness=2
        )
        y += line_gap


def draw_tracked_object(frame, obj, speed_kmh=None):
    # Draw one tracked object on the frame.
    #
    # obj is expected to be a dictionary like:
    # {
    #     "bbox": [x1, y1, x2, y2],
    #     "center": [cx, cy],
    #     "confidence": 0.91,
    #     "class_id": 2,
    #     "class_name": "car",
    #     "track_id": 7
    # }
    #
    # speed_kmh:
    # optional numeric speed value for this object
    # if provided, label will show:
    # ID:7 car | 24.3 km/h

    bbox = obj["bbox"]
    center = obj["center"]
    class_name = obj["class_name"]
    track_id = obj["track_id"]

    # Choose display color based on class.
    color = get_class_color(class_name)

    # Draw object box and center point.
    draw_bbox(frame, bbox, color=color, thickness=2)
    draw_center_point(frame, center, color=(0, 0, 255), radius=4)

    # Use top-left corner of bounding box to place label text.
    x1, y1, x2, y2 = bbox

    # Build the label text.
    #
    # Preferred output format:
    # ID:7 car | 24.3 km/h
    #
    # If speed is not available yet:
    # ID:7 car
    #
    # If track ID is missing:
    # car
    if track_id is not None:
        if speed_kmh is not None:
            label_text = f"ID:{track_id} {class_name} | {speed_kmh:.1f} km/h"
        else:
            label_text = f"ID:{track_id} {class_name}"
    else:
        if speed_kmh is not None:
            label_text = f"{class_name} | {speed_kmh:.1f} km/h"
        else:
            label_text = f"{class_name}"

    # Put label slightly above the box.
    label_position = (x1, max(y1 - 10, 20))

    draw_label(
        frame=frame,
        text=label_text,
        position=label_position,
        color=color,
        font_scale=0.6,
        thickness=2
    )


def draw_all_tracked_objects(frame, tracked_objects, speed_dict=None):
    # Draw all tracked objects in the current frame.
    #
    # speed_dict:
    # optional dictionary mapping track_id -> speed in km/h
    # example:
    # {7: 24.3, 9: 17.8}

    for obj in tracked_objects:
        track_id = obj["track_id"]

        # Default: no speed available
        speed_kmh = None

        # If speed_dict is provided and this object has a valid track ID,
        # try to get the speed for that specific object.
        if speed_dict is not None and track_id is not None:
            speed_kmh = speed_dict.get(track_id, None)

        draw_tracked_object(frame, obj, speed_kmh=speed_kmh)


def draw_trajectory_trail(frame, history_points, color=(255, 255, 255), thickness=2):
    # Draw one object's trajectory trail using its stored center points.
    #
    # history_points example:
    # [(100, 200), (104, 202), (109, 205), (115, 210)]

    if len(history_points) < 2:
        return

    for i in range(1, len(history_points)):
        pt1 = history_points[i - 1]
        pt2 = history_points[i]
        cv2.line(frame, pt1, pt2, color, thickness)


def draw_all_trajectory_trails(frame, tracked_objects, all_track_histories):
    # Draw trajectory trails for all currently visible tracked objects.
    #
    # all_track_histories example:
    # {
    #   7: [(100, 200), (105, 203), (111, 207)],
    #   9: [(300, 120), (301, 126), (305, 131)]
    # }

    for obj in tracked_objects:
        track_id = obj["track_id"]
        class_name = obj["class_name"]

        if track_id is None:
            continue

        history_points = all_track_histories.get(track_id, [])
        color = get_class_color(class_name)

        draw_trajectory_trail(frame, history_points, color=color, thickness=2)


def draw_horizontal_line(frame, y, color=(255, 255, 0), thickness=2):
    # Draw a horizontal line across the frame.
    frame_height, frame_width = frame.shape[:2]
    cv2.line(frame, (0, y), (frame_width, y), color, thickness)


def draw_vertical_line(frame, x, color=(255, 255, 0), thickness=2):
    # Draw a vertical line across the frame.
    frame_height, frame_width = frame.shape[:2]
    cv2.line(frame, (x, 0), (x, frame_height), color, thickness)