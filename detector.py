from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from .config import MODEL_NAME, CONF_THRESHOLD, IMG_SIZE, ROBOT_CLASS_GROUPS
from .preprocessing import preprocess_frame
from .robot_logic import decide_robot_action, object_position


class RobotObjectDetector:
    def __init__(self, model_name: str = MODEL_NAME, conf: float = CONF_THRESHOLD, imgsz: int = IMG_SIZE):
        self.model_name = model_name
        self.conf = conf
        self.imgsz = imgsz
        self.model = YOLO(model_name)

    def detect(self, frame: np.ndarray, preprocess: str = "none") -> Dict:
        processed = preprocess_frame(frame, preprocess)
        h, w = processed.shape[:2]

        result = self.model.predict(
            processed,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False
        )[0]

        detections: List[Dict] = []

        if result.boxes is not None:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = names.get(cls_id, str(cls_id))
                confidence = float(box.conf[0])

                if class_name not in ROBOT_CLASS_GROUPS:
                    continue

                xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
                group = ROBOT_CLASS_GROUPS[class_name]

                detections.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "group": group,
                    "confidence": confidence,
                    "xyxy": xyxy,
                    "position": object_position(xyxy, w),
                })

        action = decide_robot_action(detections, w, h)

        return {
            "frame": processed,
            "detections": detections,
            "robot_action": action,
            "width": w,
            "height": h,
        }

    def draw(self, result: Dict) -> np.ndarray:
        frame = result["frame"].copy()

        for det in result["detections"]:
            x1, y1, x2, y2 = map(int, det["xyxy"])
            label = f'{det["class_name"]} {det["confidence"]:.2f} | {det["group"]} | {det["position"]}'

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 180, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            f'Robot action: {result["robot_action"]}',
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        return frame
