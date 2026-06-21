import cv2
import numpy as np


def preprocess_frame(frame: np.ndarray, mode: str = "none") -> np.ndarray:
    if frame is None:
        raise ValueError("Empty frame")

    mode = (mode or "none").lower()

    if mode == "none":
        return frame

    if mode == "low_light":
        return cv2.convertScaleAbs(frame, alpha=0.55, beta=-15)

    if mode == "overexposure":
        return cv2.convertScaleAbs(frame, alpha=1.35, beta=55)

    if mode == "contrast":
        return cv2.convertScaleAbs(frame, alpha=1.35, beta=5)

    if mode == "clahe":
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        merged = cv2.merge((l2, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    if mode == "blur":
        return cv2.GaussianBlur(frame, (9, 9), 0)

    if mode == "noise":
        noise = np.random.normal(0, 18, frame.shape).astype(np.int16)
        noisy = frame.astype(np.int16) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    if mode == "occlusion":
        out = frame.copy()
        h, w = out.shape[:2]
        x1, y1 = int(w * 0.35), int(h * 0.35)
        x2, y2 = int(w * 0.65), int(h * 0.65)
        cv2.rectangle(out, (x1, y1), (x2, y2), (30, 30, 30), -1)
        return out

    raise ValueError(
        f"Unknown preprocess mode: {mode}. "
        "Use: none, low_light, overexposure, contrast, clahe, blur, noise, occlusion."
    )
