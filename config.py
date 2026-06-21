from pathlib import Path

MODEL_NAME = "yolo11n.pt"

CONF_THRESHOLD = 0.35
IMG_SIZE = 640

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_IMAGES_DIR = PROJECT_ROOT / "outputs" / "images"
OUTPUT_VIDEOS_DIR = PROJECT_ROOT / "outputs" / "videos"
BENCHMARK_DIR = PROJECT_ROOT / "outputs" / "benchmark"

ROBOT_CLASS_GROUPS = {
    "person": "safety_object",
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "motorcycle": "moving_object",
    "bicycle": "moving_object",
    "traffic light": "infrastructure",
    "stop sign": "infrastructure",
    "suitcase": "cargo_like_object",
    "backpack": "cargo_like_object",
}

PRIORITY_GROUPS = {
    "safety_object": 1,
    "vehicle": 2,
    "moving_object": 3,
    "static_obstacle": 4,
    "cargo_like_object": 5,
    "infrastructure": 6,
}