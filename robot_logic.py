from typing import Dict, List, Tuple
from .config import PRIORITY_GROUPS


def object_position(xyxy: Tuple[float, float, float, float], frame_width: int) -> str:
    x1, _, x2, _ = xyxy
    cx = (x1 + x2) / 2

    left_border = frame_width / 3
    right_border = 2 * frame_width / 3

    if cx < left_border:
        return "left"
    if cx > right_border:
        return "right"
    return "center"


def object_size_ratio(xyxy: Tuple[float, float, float, float], frame_width: int, frame_height: int) -> float:
    x1, y1, x2, y2 = xyxy
    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    frame_area = max(1, frame_width * frame_height)
    return float(box_area / frame_area)


def decide_robot_action(detections: List[Dict], frame_width: int, frame_height: int) -> str:
    if not detections:
        return "CONTINUE_SCANNING"

    detections = sorted(
        detections,
        key=lambda d: (PRIORITY_GROUPS.get(d["group"], 99), -d["confidence"])
    )

    main = detections[0]
    pos = object_position(main["xyxy"], frame_width)
    size = object_size_ratio(main["xyxy"], frame_width, frame_height)

    if main["group"] == "safety_object":
        if pos == "center" or size > 0.12:
            return "STOP_PERSON_DETECTED"
        return f"SLOW_DOWN_PERSON_{pos.upper()}"

    if main["group"] in {"vehicle", "moving_object"}:
        if pos == "center" or size > 0.18:
            return "SLOW_DOWN_MOVING_OBJECT"
        return f"MONITOR_OBJECT_{pos.upper()}"

    if main["group"] in {"static_obstacle", "cargo_like_object"}:
        if pos == "left":
            return "TURN_RIGHT_OBSTACLE_LEFT"
        if pos == "right":
            return "TURN_LEFT_OBSTACLE_RIGHT"
        return "STOP_OR_CHANGE_PATH_OBSTACLE_CENTER"

    if main["group"] == "infrastructure":
        return f"MARKER_DETECTED_{main['class_name'].upper().replace(' ', '_')}"

    return "OBJECT_DETECTED"
