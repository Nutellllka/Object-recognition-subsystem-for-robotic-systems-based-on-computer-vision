#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO


APP_TITLE = "ROBOT VISION HUD"
SETTINGS_FILE = Path("hud_settings.json")

DEFAULT_SETTINGS = {
  "custom_model": ".\\models\\logistics3_clean_to_hybrid_ft15\\weights\\best.pt",
  "general_model": ".\\yolo11n.pt",
  "mode": "dual",
  "custom_conf": 0.25,
  "general_conf": 0.40,
  "suppress_iou": 0.45,
  "custom_iou": 0.70,
  "general_iou": 0.70,
  "max_det": 80,
  "imgsz": 640,
  "device": "0",
  "camera_id": 0
}


def load_settings() -> Dict:
    if SETTINGS_FILE.exists():
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            merged = dict(DEFAULT_SETTINGS)
            merged.update(data)
            return merged
        except Exception:
            return dict(DEFAULT_SETTINGS)
    return dict(DEFAULT_SETTINGS)


def save_settings(settings: Dict) -> None:
    SETTINGS_FILE.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")


CUSTOM_CLASS_INFO: Dict[str, Dict[str, str]] = {
    "barcode": {
        "category": "VISUAL MARKER",
        "action": "IDENTIFY / READ MARKER",
        "priority": "MEDIUM",
    },
    "cargo_object": {
        "category": "TARGET CARGO",
        "action": "FIX TARGET COORDINATES",
        "priority": "HIGH",
    },
    "loading_equipment": {
        "category": "TECHNICAL OBJECT",
        "action": "WARNING / KEEP DISTANCE",
        "priority": "HIGH",
    },
}

GENERAL_CLASS_INFO: Dict[str, Dict[str, str]] = {
    "person": {
        "category": "SAFETY OBJECT",
        "action": "STOP / AVOID COLLISION",
        "priority": "CRITICAL",
    },
    "car": {
        "category": "VEHICLE",
        "action": "TRACK / KEEP DISTANCE",
        "priority": "HIGH",
    },
    "truck": {
        "category": "VEHICLE",
        "action": "TRACK / KEEP DISTANCE",
        "priority": "HIGH",
    },
    "bus": {
        "category": "VEHICLE",
        "action": "TRACK / KEEP DISTANCE",
        "priority": "HIGH",
    },
    "motorcycle": {
        "category": "VEHICLE",
        "action": "TRACK / KEEP DISTANCE",
        "priority": "HIGH",
    },
    "bicycle": {
        "category": "VEHICLE",
        "action": "TRACK / KEEP DISTANCE",
        "priority": "MEDIUM",
    },
    "traffic light": {
        "category": "INFRASTRUCTURE",
        "action": "REGISTER ROAD ELEMENT",
        "priority": "MEDIUM",
    },
    "stop sign": {
        "category": "INFRASTRUCTURE",
        "action": "REGISTER ROAD ELEMENT",
        "priority": "MEDIUM",
    },
}

DEFAULT_GENERAL_CLASSES = set(GENERAL_CLASS_INFO.keys())


def get_zone(x1: int, x2: int, frame_w: int) -> str:
    cx = (x1 + x2) / 2
    if cx < frame_w / 3:
        return "LEFT"
    if cx > frame_w * 2 / 3:
        return "RIGHT"
    return "CENTER"


def iou(box_a: List[int], box_b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def interpret_custom(class_name: str, x1: int, x2: int, frame_w: int) -> Dict[str, str]:
    info = CUSTOM_CLASS_INFO.get(class_name, {
        "category": "SPECIAL OBJECT",
        "action": "REGISTER DETECTION",
        "priority": "LOW",
    })
    result = dict(info)
    result["zone"] = get_zone(x1, x2, frame_w)
    result["source"] = "CUSTOM"
    return result


def interpret_general(class_name: str, x1: int, x2: int, frame_w: int) -> Dict[str, str]:
    info = GENERAL_CLASS_INFO.get(class_name, {
        "category": "GENERAL OBJECT",
        "action": "REGISTER DETECTION",
        "priority": "LOW",
    })
    result = dict(info)
    result["zone"] = get_zone(x1, x2, frame_w)
    result["source"] = "GENERAL"
    return result


RED_DARK = (0, 0, 95)
RED_LIGHT = (80, 80, 255)
WHITE_RED = (190, 205, 255)
ORANGE = (0, 190, 255)
GREEN = (80, 255, 80)
CYAN = (255, 220, 90)


def draw_text(img, text, org, scale=0.55, color=RED_LIGHT, thickness=1):
    cv2.putText(img, str(text), org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def priority_color(priority: str, source: str):
    if priority == "CRITICAL":
        return GREEN
    if source == "GENERAL":
        return CYAN
    if priority == "HIGH":
        return RED_LIGHT
    if priority == "MEDIUM":
        return ORANGE
    return WHITE_RED


def draw_corner_box(img, x1, y1, x2, y2, color=RED_LIGHT, thickness=2):
    w = max(12, x2 - x1)
    h = max(12, y2 - y1)
    corner = max(18, min(w, h) // 5)

    cv2.line(img, (x1, y1), (x1 + corner, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + corner), color, thickness)

    cv2.line(img, (x2, y1), (x2 - corner, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + corner), color, thickness)

    cv2.line(img, (x1, y2), (x1 + corner, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - corner), color, thickness)

    cv2.line(img, (x2, y2), (x2 - corner, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner), color, thickness)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)


def draw_scanlines(img, step=8, alpha=0.13):
    overlay = img.copy()
    h, w = img.shape[:2]
    for y in range(0, h, step):
        cv2.line(overlay, (0, y), (w, y), RED_DARK, 1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def draw_vignette(img, alpha=0.35):
    h, w = img.shape[:2]
    x_kernel = cv2.getGaussianKernel(w, w / 2.4)
    y_kernel = cv2.getGaussianKernel(h, h / 2.4)
    kernel = y_kernel @ x_kernel.T
    mask = kernel / kernel.max()
    mask = np.dstack([mask] * 3)
    dark = img.astype(np.float32) * mask + img.astype(np.float32) * (1 - alpha) * (1 - mask)
    return np.clip(dark, 0, 255).astype(np.uint8)


def draw_crosshair(img):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(img, (cx - 28, cy), (cx - 8, cy), RED_DARK, 1)
    cv2.line(img, (cx + 8, cy), (cx + 28, cy), RED_DARK, 1)
    cv2.line(img, (cx, cy - 28), (cx, cy - 8), RED_DARK, 1)
    cv2.line(img, (cx, cy + 8), (cx, cy + 28), RED_DARK, 1)
    cv2.circle(img, (cx, cy), 36, RED_DARK, 1)
    cv2.circle(img, (cx, cy), 3, RED_LIGHT, -1)


def draw_header(img, fps, source_label, custom_label, general_label, mode):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 46), (0, 0, 0), -1)
    cv2.line(img, (0, 46), (w, 46), RED_DARK, 1)

    draw_text(img, "ROBOT VISION HUD // OBJECT RECOGNITION SUBSYSTEM", (16, 19), 0.55, RED_LIGHT, 1)
    draw_text(img, f"SRC: {source_label}", (16, 39), 0.42, WHITE_RED, 1)
    draw_text(img, f"MODE: {mode.upper()}", (w // 2 - 140, 39), 0.42, WHITE_RED, 1)
    draw_text(img, f"CUSTOM: {custom_label}", (w // 2 - 40, 39), 0.42, WHITE_RED, 1)
    if general_label:
        draw_text(img, f"GENERAL: {general_label}", (w // 2 + 125, 39), 0.42, WHITE_RED, 1)
    draw_text(img, f"FPS: {fps:05.1f}", (w - 120, 24), 0.55, RED_LIGHT, 1)


def draw_detection(img, det):
    x1, y1, x2, y2 = det["bbox"]
    color = priority_color(det["priority"], det["source"])

    draw_corner_box(img, x1, y1, x2, y2, color, 2)

    label1 = f"{det['class'].upper()}  {det['conf']:.2f}"
    label2 = f"{det['category']} | {det['zone']}"

    tx = x1
    ty = max(62, y1 - 30)

    (tw1, _), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    (tw2, _), _ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)
    bg_w = min(img.shape[1] - tx - 4, max(tw1, tw2) + 14)

    cv2.rectangle(img, (tx, ty - 18), (tx + bg_w, ty + 24), (0, 0, 0), -1)
    cv2.rectangle(img, (tx, ty - 18), (tx + bg_w, ty + 24), color, 1)

    draw_text(img, label1, (tx + 6, ty), 0.50, color, 1)
    draw_text(img, label2, (tx + 6, ty + 18), 0.43, WHITE_RED, 1)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    cv2.line(img, (tx + bg_w, ty + 2), (cx, cy), color, 1)


def draw_side_panel(img, detections, custom_conf, general_conf, mode):
    h, w = img.shape[:2]
    panel_w = 330 if w >= 1050 else 285
    x0 = max(0, w - panel_w)

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, 46), (w, h), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(overlay, 0.56, img, 0.44, 0)

    cv2.line(img, (x0, 46), (x0, h), RED_DARK, 1)
    draw_text(img, "DETECTION STATUS", (x0 + 16, 74), 0.57, RED_LIGHT, 1)
    draw_text(img, f"MODE: {mode.upper()}", (x0 + 16, 98), 0.42, WHITE_RED, 1)
    draw_text(img, f"CUSTOM THR: {custom_conf:.2f}", (x0 + 16, 118), 0.42, WHITE_RED, 1)
    if mode in ("dual", "general"):
        draw_text(img, f"GENERAL THR: {general_conf:.2f}", (x0 + 16, 138), 0.42, WHITE_RED, 1)

    y = 172 if mode in ("dual", "general") else 148
    draw_text(img, f"OBJECTS: {len(detections)}", (x0 + 16, y), 0.44, WHITE_RED, 1)
    y += 32

    if not detections:
        draw_text(img, "NO TARGETS DETECTED", (x0 + 16, y), 0.52, RED_LIGHT, 1)
        draw_text(img, "SCANNING...", (x0 + 16, y + 24), 0.48, WHITE_RED, 1)
        return

    for idx, det in enumerate(detections[:6], start=1):
        if y + 92 > h:
            break

        color = priority_color(det["priority"], det["source"])
        cv2.rectangle(img, (x0 + 12, y - 18), (w - 12, y + 72), color, 1)

        draw_text(img, f"[{idx:02d}] {det['class'].upper()}  {det['conf']:.2f}", (x0 + 22, y), 0.48, color, 1)
        draw_text(img, f"SRC: {det['source']}  CAT: {det['category']}", (x0 + 22, y + 20), 0.38, WHITE_RED, 1)
        draw_text(img, f"ZONE: {det['zone']}  PRIORITY: {det['priority']}", (x0 + 22, y + 40), 0.38, WHITE_RED, 1)
        draw_text(img, f"ACT: {det['action']}", (x0 + 22, y + 60), 0.36, WHITE_RED, 1)

        y += 100


def extract_detections(model, results, frame_w, allowed_names=None, source="CUSTOM"):
    detections = []
    if not results or results[0].boxes is None:
        return detections

    names = model.names

    for box in results[0].boxes:
        conf = float(box.conf[0].item())
        cls_id = int(box.cls[0].item())
        cls_name = str(names.get(cls_id, f"class_{cls_id}"))

        if allowed_names is not None and cls_name not in allowed_names:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()

        if source == "CUSTOM":
            info = interpret_custom(cls_name, x1, x2, frame_w)
        else:
            info = interpret_general(cls_name, x1, x2, frame_w)

        detections.append({
            "class": cls_name,
            "conf": conf,
            "bbox": [x1, y1, x2, y2],
            **info,
        })

    return detections


def suppress_custom_over_general(custom_dets, general_dets, iou_thr=0.35):

    kept = []
    for c in custom_dets:
        suppress = False
        for g in general_dets:
            if g.get("class") == "person" and c.get("class") in {"cargo_object", "loading_equipment"}:
                if iou(c["bbox"], g["bbox"]) >= iou_thr:
                    suppress = True
                    break
        if not suppress:
            kept.append(c)
    return kept


def suppress_general_vehicle_over_custom(general_dets, custom_dets, iou_thr=0.35):

    vehicle_like = {"truck", "car", "bus", "motorcycle", "bicycle"}
    kept = []
    for g in general_dets:
        suppress = False
        if g.get("class") in vehicle_like:
            for c in custom_dets:
                if c.get("class") == "loading_equipment" and iou(g["bbox"], c["bbox"]) >= iou_thr:
                    suppress = True
                    break
        if not suppress:
            kept.append(g)
    return kept


def predict_frame(custom_model, general_model, frame, args):
    custom_results = None
    general_results = None

    if args.mode in ("custom", "dual"):
        custom_results = custom_model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.custom_conf,
            iou=args.custom_iou,
            max_det=args.max_det,
            agnostic_nms=False,
            device=args.device,
            verbose=False,
        )

    if args.mode in ("general", "dual") and general_model is not None:
        general_results = general_model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.general_conf,
            iou=args.general_iou,
            max_det=args.max_det,
            agnostic_nms=False,
            device=args.device,
            verbose=False,
        )

    return custom_results, general_results


def scale_detections(detections, scale, dx, dy):
    scaled = []
    for d in detections:
        nd = dict(d)
        x1, y1, x2, y2 = d["bbox"]
        nd["bbox"] = [
            int(x1 * scale + dx),
            int(y1 * scale + dy),
            int(x2 * scale + dx),
            int(y2 * scale + dy),
        ]
        scaled.append(nd)
    return scaled


def fit_frame_to_area(frame, area_w, area_h):
    h, w = frame.shape[:2]
    scale = min(area_w / w, area_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((area_h, area_w, 3), dtype=np.uint8)
    dx = (area_w - new_w) // 2
    dy = (area_h - new_h) // 2
    canvas[dy:dy + new_h, dx:dx + new_w] = resized
    return canvas, scale, dx, dy


def draw_header_wide(canvas, fps, source_label, custom_label, general_label, mode, video_w, panel_x):
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (w, 54), (0, 0, 0), -1)
    cv2.line(canvas, (0, 54), (w, 54), RED_DARK, 1)
    cv2.line(canvas, (video_w, 54), (video_w, h), RED_DARK, 1)

    draw_text(canvas, "ROBOT VISION HUD // OBJECT RECOGNITION SUBSYSTEM", (18, 22), 0.58, RED_LIGHT, 1)
    src_short = str(source_label)
    if len(src_short) > 45:
        src_short = src_short[:42] + "..."
    draw_text(canvas, f"SRC: {src_short}", (18, 43), 0.42, WHITE_RED, 1)
    draw_text(canvas, f"MODE: {mode.upper()}", (panel_x + 18, 24), 0.48, WHITE_RED, 1)
    draw_text(canvas, f"CUSTOM: {custom_label}", (panel_x + 18, 43), 0.40, WHITE_RED, 1)
    if general_label:
        draw_text(canvas, f"GENERAL: {general_label}", (panel_x + 165, 43), 0.40, WHITE_RED, 1)
    draw_text(canvas, f"FPS: {fps:05.1f}", (w - 124, 24), 0.55, RED_LIGHT, 1)


def draw_side_panel_clean(canvas, detections, custom_conf, general_conf, mode, panel_x, panel_w):
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (panel_x, 54), (w, h), (0, 0, 0), -1)
    for y in range(54, h, 8):
        cv2.line(canvas, (panel_x, y), (w, y), RED_DARK, 1)

    x0 = panel_x + 18
    draw_text(canvas, "DETECTION STATUS", (x0, 86), 0.62, RED_LIGHT, 1)
    draw_text(canvas, f"MODE: {mode.upper()}", (x0, 114), 0.45, WHITE_RED, 1)
    draw_text(canvas, f"CUSTOM THR: {custom_conf:.2f}", (x0, 136), 0.42, WHITE_RED, 1)
    if mode in ("dual", "general"):
        draw_text(canvas, f"GENERAL THR: {general_conf:.2f}", (x0, 158), 0.42, WHITE_RED, 1)

    y = 198
    draw_text(canvas, f"OBJECTS: {len(detections)}", (x0, y), 0.46, WHITE_RED, 1)
    y += 36

    if not detections:
        cv2.rectangle(canvas, (x0, y - 20), (w - 18, y + 54), RED_LIGHT, 1)
        draw_text(canvas, "NO TARGETS DETECTED", (x0 + 12, y + 4), 0.50, RED_LIGHT, 1)
        draw_text(canvas, "SCANNING...", (x0 + 12, y + 30), 0.45, WHITE_RED, 1)
        return

    for idx, det in enumerate(detections[:8], start=1):
        if y + 94 > h:
            break

        color = priority_color(det["priority"], det["source"])
        cv2.rectangle(canvas, (x0, y - 20), (w - 18, y + 70), color, 1)

        draw_text(canvas, f"[{idx:02d}] {det['class'].upper()}  {det['conf']:.2f}", (x0 + 12, y), 0.48, color, 1)
        draw_text(canvas, f"SRC: {det['source']}  CAT: {det['category']}", (x0 + 12, y + 22), 0.36, WHITE_RED, 1)
        draw_text(canvas, f"ZONE: {det['zone']}  PRIORITY: {det['priority']}", (x0 + 12, y + 43), 0.36, WHITE_RED, 1)
        draw_text(canvas, f"ACT: {det['action']}", (x0 + 12, y + 63), 0.34, WHITE_RED, 1)

        y += 102


def make_hud(frame, custom_results, general_results, custom_model, general_model, fps, source_label, args):
    out_w, out_h = 1280, 720
    panel_w = 340
    video_w = out_w - panel_w
    panel_x = video_w

    frame_area, scale, dx, dy = fit_frame_to_area(frame, video_w, out_h - 54)
    frame_area = draw_vignette(frame_area, alpha=0.34)
    frame_area = draw_scanlines(frame_area, step=8, alpha=0.12)

    h0, w0 = frame.shape[:2]
    custom_dets = []
    general_dets = []

    if args.mode in ("custom", "dual"):
        custom_dets = extract_detections(custom_model, custom_results, w0, allowed_names=None, source="CUSTOM")

    if args.mode in ("general", "dual") and general_model is not None:
        general_dets = extract_detections(
            general_model,
            general_results,
            w0,
            allowed_names=DEFAULT_GENERAL_CLASSES,
            source="GENERAL",
        )

    if args.mode == "dual":
        custom_dets = suppress_custom_over_general(custom_dets, general_dets, iou_thr=args.suppress_iou)
        general_dets = suppress_general_vehicle_over_custom(general_dets, custom_dets, iou_thr=0.30)

    detections = general_dets + custom_dets

    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    detections.sort(key=lambda d: (priority_order.get(d["priority"], 9), -d["conf"]))

    scaled_dets = scale_detections(detections, scale, dx, dy)

    for det in scaled_dets:
        draw_detection(frame_area, det)

    draw_crosshair(frame_area)

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[54:out_h, 0:video_w] = frame_area[:out_h - 54, :video_w]

    general_label = Path(args.general_model).name if args.mode in ("general", "dual") else ""
    draw_header_wide(
        canvas,
        fps=fps,
        source_label=source_label,
        custom_label=Path(args.custom_model).name,
        general_label=general_label,
        mode=args.mode,
        video_w=video_w,
        panel_x=panel_x,
    )
    draw_side_panel_clean(canvas, detections, args.custom_conf, args.general_conf, args.mode, panel_x, panel_w)

    return canvas


def is_image(path: Path):
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_video(path: Path):
    return path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def load_models(args):
    custom_model = None
    general_model = None

    if args.mode in ("custom", "dual"):
        custom_path = Path(args.custom_model)
        if not custom_path.exists():
            raise FileNotFoundError(f"Custom model not found: {custom_path}")
        custom_model = YOLO(str(custom_path))
        print(f"Custom model: {custom_path}")
        print(f"Custom classes: {custom_model.names}")

    if args.mode in ("general", "dual"):
        general_path = Path(args.general_model)
        if not general_path.exists():
            raise FileNotFoundError(f"General model not found: {general_path}")
        general_model = YOLO(str(general_path))
        print(f"General model: {general_path}")
        print(f"General classes used: {sorted(DEFAULT_GENERAL_CLASSES)}")

    return custom_model, general_model


def process_image(args):
    custom_model, general_model = load_models(args)

    path = Path(args.source)
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    t0 = time.time()
    custom_results, general_results = predict_frame(custom_model, general_model, frame, args)
    fps = 1.0 / max(time.time() - t0, 1e-6)

    hud = make_hud(frame, custom_results, general_results, custom_model, general_model, fps, path.name, args)

    output = Path(args.output) if args.output else Path("outputs") / f"{path.stem}_hud{path.suffix}"
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), hud)
    print(f"Saved: {output}")

    cv2.imshow("Robot Vision HUD", hud)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_stream(args, camera=False):
    custom_model, general_model = load_models(args)

    if camera:
        cap = cv2.VideoCapture(args.camera_id)
        source_label = f"CAMERA {args.camera_id}"
    else:
        cap = cv2.VideoCapture(args.source)
        source_label = Path(args.source).name

    if not cap.isOpened():
        raise RuntimeError("Could not open source")

    writer: Optional[cv2.VideoWriter] = None
    out_path = Path(args.output) if args.output else None

    fps_smooth = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.time()
        custom_results, general_results = predict_frame(custom_model, general_model, frame, args)
        fps = 1.0 / max(time.time() - t0, 1e-6)
        fps_smooth = fps if fps_smooth == 0 else fps_smooth * 0.85 + fps * 0.15

        hud = make_hud(frame, custom_results, general_results, custom_model, general_model, fps_smooth, source_label, args)

        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if writer is None:
                h, w = hud.shape[:2]
                src_fps = cap.get(cv2.CAP_PROP_FPS)
                if src_fps <= 1 or np.isnan(src_fps):
                    src_fps = 25
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, src_fps, (w, h))
            writer.write(hud)

        cv2.imshow("Robot Vision HUD", hud)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved: {out_path}")
    cv2.destroyAllWindows()


def start_launcher():
    import os
    import tkinter as tk
    from tkinter import filedialog, messagebox

    settings = load_settings()

    root = tk.Tk()
    root.title(APP_TITLE + " // LAUNCHER")
    root.geometry("1040x650")
    root.configure(bg="#000603")
    root.resizable(False, False)

    # Green HUD palette
    bg = "#000603"
    panel_bg = "#00120a"
    panel_bg_2 = "#001b0f"
    green = "#00ff66"
    green_soft = "#66ff99"
    green_dark = "#006b35"
    green_dim = "#00361f"
    white = "#d8ffe8"
    cyan = "#52ffd2"
    warning = "#eaff8a"

    canvas = tk.Canvas(root, width=1040, height=650, bg=bg, highlightthickness=0)
    canvas.place(x=0, y=0)

    for x in range(0, 1040, 26):
        canvas.create_line(x, 0, x, 650, fill="#00160c")
    for y in range(0, 650, 26):
        canvas.create_line(0, y, 1040, y, fill="#00160c")
    for y in range(0, 650, 8):
        canvas.create_line(0, y, 1040, y, fill="#002414")

    # Main frame
    canvas.create_rectangle(18, 18, 1022, 632, outline=green_dark, width=2)
    canvas.create_rectangle(34, 78, 1006, 610, outline=green_dim, width=1)
    canvas.create_line(18, 68, 1022, 68, fill=green_dark, width=1)

    bracket = 40
    for x0, y0, sx, sy in [(18, 18, 1, 1), (1022, 18, -1, 1), (18, 632, 1, -1), (1022, 632, -1, -1)]:
        canvas.create_line(x0, y0, x0 + sx * bracket, y0, fill=green, width=2)
        canvas.create_line(x0, y0, x0, y0 + sy * bracket, fill=green, width=2)

    radar_cx, radar_cy = 874, 166
    canvas.create_oval(radar_cx - 56, radar_cy - 56, radar_cx + 56, radar_cy + 56, outline=green_dim, width=1)
    canvas.create_oval(radar_cx - 36, radar_cy - 36, radar_cx + 36, radar_cy + 36, outline=green_dim, width=1)
    canvas.create_oval(radar_cx - 16, radar_cy - 16, radar_cx + 16, radar_cy + 16, outline=green_dim, width=1)
    canvas.create_line(radar_cx - 62, radar_cy, radar_cx + 62, radar_cy, fill=green_dim)
    canvas.create_line(radar_cx, radar_cy - 62, radar_cx, radar_cy + 62, fill=green_dim)
    sweep = canvas.create_line(radar_cx, radar_cy, radar_cx + 56, radar_cy, fill=green, width=2)

    matrix_items = []
    matrix_text = ["0101", "YOLO", "BBOX", "SCAN", "OBJ", "CONF", "HUD", "CAM", "MAP50", "NMS"]
    for i in range(23):
        x = 36 + i * 43
        y = 88 + (i % 7) * 31
        item = canvas.create_text(x, y, text=matrix_text[i % len(matrix_text)], fill="#003d21",
                                  font=("Consolas", 8), anchor="nw")
        matrix_items.append(item)

    title = tk.Label(
        root,
        text="ROBOT VISION HUD // LAUNCH CONTROL",
        bg=bg,
        fg=green,
        font=("Consolas", 24, "bold"),
    )
    title.place(x=46, y=28)

    subtitle = tk.Label(
        root,
        text="OBJECT RECOGNITION SUBSYSTEM  //  SELECT INPUT SOURCE",
        bg=bg,
        fg=white,
        font=("Consolas", 11),
    )
    subtitle.place(x=50, y=61)

    status_var = tk.StringVar(value="SYSTEM READY // DUAL DETECTION ONLINE // AWAITING INPUT")
    status = tk.Label(
        root,
        textvariable=status_var,
        bg=bg,
        fg=green,
        font=("Consolas", 11, "bold"),
    )
    status.place(x=50, y=600)

    scan_y = {"value": 88}
    scan_line = canvas.create_line(36, scan_y["value"], 1004, scan_y["value"], fill="#00ff66", width=2)

    def animate():
        scan_y["value"] += 3
        if scan_y["value"] > 604:
            scan_y["value"] = 88
        canvas.coords(scan_line, 36, scan_y["value"], 1004, scan_y["value"])

        import math
        t = time.time() * 1.8
        x2 = radar_cx + 56 * math.cos(t)
        y2 = radar_cy + 56 * math.sin(t)
        canvas.coords(sweep, radar_cx, radar_cy, x2, y2)

        for idx, item in enumerate(matrix_items):
            x, y = canvas.coords(item)
            y += 0.22 + (idx % 3) * 0.08
            if y > 610:
                y = 88
            canvas.coords(item, x, y)

        root.after(35, animate)

    def hud_panel(x, y, w, h, title_text=""):
        canvas.create_rectangle(x, y, x + w, y + h, fill=panel_bg, outline=green_dim, width=1)
        canvas.create_line(x, y, x + 28, y, fill=green, width=2)
        canvas.create_line(x, y, x, y + 28, fill=green, width=2)
        canvas.create_line(x + w, y + h, x + w - 28, y + h, fill=green, width=2)
        canvas.create_line(x + w, y + h, x + w, y + h - 28, fill=green, width=2)
        if title_text:
            tk.Label(root, text=title_text, bg=panel_bg, fg=green_soft, font=("Consolas", 10, "bold")).place(x=x+16, y=y+8)

    hud_panel(56, 116, 454, 92)
    hud_panel(56, 234, 454, 92)
    hud_panel(56, 352, 454, 92)
    hud_panel(574, 116, 392, 150, "MODEL STATUS")
    hud_panel(574, 292, 392, 78)
    hud_panel(574, 394, 392, 78)
    hud_panel(574, 496, 392, 90, "SYSTEM CONTROLS")

    model_text = (
        f"CUSTOM MODEL : {Path(settings['custom_model']).name}\\n"
        f"GENERAL MODEL: {Path(settings['general_model']).name}\\n"
        f"CUSTOM THR   : {settings['custom_conf']}\\n"
        f"GENERAL THR  : {settings['general_conf']}\\n"
        f"DEVICE       : {settings['device']}     IMAGE SIZE: {settings['imgsz']}\\n"
        f"CUSTOM IOU   : {settings.get('custom_iou', 0.70)}     MAX DET: {settings.get('max_det', 80)}"
    )

    model_panel = tk.Label(
        root,
        text=model_text,
        justify="left",
        bg=panel_bg,
        fg=white,
        font=("Consolas", 10, "bold"),
        padx=14,
        pady=12,
    )
    model_panel.place(x=598, y=150, width=328, height=102)

    def build_cmd(source: str, output: str = ""):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run",
            "--source", source,
            "--mode", settings["mode"],
            "--custom-model", settings["custom_model"],
            "--general-model", settings["general_model"],
            "--custom-conf", str(settings["custom_conf"]),
            "--general-conf", str(settings["general_conf"]),
            "--suppress-iou", str(settings["suppress_iou"]),
            "--imgsz", str(settings["imgsz"]),
            "--device", str(settings["device"]),
            "--camera-id", str(settings["camera_id"]),
        ]
        if output:
            cmd.extend(["--output", output])
        return cmd

    def launch(cmd):
        try:
            status_var.set("LAUNCHING MODULE // OPEN CV HUD WINDOW STARTING...")
            subprocess.Popen(cmd, cwd=str(Path.cwd()))
        except Exception as exc:
            messagebox.showerror("Launch error", str(exc))
            status_var.set("ERROR // MODULE LAUNCH FAILED")

    def choose_image():
        file_path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            status_var.set("IMAGE MODULE // SELECTION CANCELED")
            return
        out = str(Path("outputs") / (Path(file_path).stem + "_hud" + Path(file_path).suffix))
        status_var.set(f"IMAGE MODULE // TARGET: {Path(file_path).name}")
        launch(build_cmd(file_path, out))

    def choose_video():
        file_path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[
                ("Video", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            status_var.set("VIDEO MODULE // SELECTION CANCELED")
            return
        out = str(Path("outputs") / (Path(file_path).stem + "_hud.mp4"))
        status_var.set(f"VIDEO MODULE // TARGET: {Path(file_path).name}")
        launch(build_cmd(file_path, out))

    def start_camera():
        status_var.set(f"LIVE CAMERA MODULE // CAMERA ID: {settings['camera_id']}")
        launch(build_cmd("camera"))

    def open_settings_file():
        save_settings(settings)
        try:
            os.startfile(str(SETTINGS_FILE.resolve()))
            status_var.set("SETTINGS OPENED // hud_settings_v6.json")
        except Exception:
            messagebox.showinfo(
                "Settings",
                "Settings saved to hud_settings_v6.json.\\n\\n"
                "You can edit model paths, thresholds, device and camera_id there."
            )
            status_var.set("SETTINGS FILE CREATED // hud_settings_v6.json")

    def make_button(text, command, x, y, w=454, h=92, accent=green):
        btn = tk.Button(
            root,
            text=text,
            command=command,
            bg=panel_bg,
            fg=accent,
            activebackground=panel_bg_2,
            activeforeground=white,
            font=("Consolas", 19, "bold"),
            relief="flat",
            bd=0,
            anchor="w",
            padx=28,
        )
        btn.place(x=x+8, y=y+8, width=w-16, height=h-16)

        def on_enter(_):
            btn.configure(bg=panel_bg_2, fg=white)
            status_var.set(f"MODULE SELECTED // {text.strip()}")

        def on_leave(_):
            btn.configure(bg=panel_bg, fg=accent)
            status_var.set("SYSTEM READY // DUAL DETECTION ONLINE // AWAITING INPUT")

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        return btn

    make_button("▣  IMAGE SCAN", choose_image, 56, 116)
    make_button("▶  VIDEO STREAM", choose_video, 56, 234)
    make_button("◉  LIVE CAMERA", start_camera, 56, 352)

    make_button("⚙  SETTINGS", open_settings_file, 574, 292, 392, 78, accent=cyan)
    make_button("✕  EXIT", root.destroy, 574, 394, 392, 78, accent=warning)

    controls_text = (
        "Q / ESC  - close OpenCV HUD window\\n"
        "outputs/ - saved images and videos\\n"
        "dual     - general + custom detection"
    )
    controls = tk.Label(
        root,
        text=controls_text,
        justify="left",
        bg=panel_bg,
        fg=white,
        font=("Consolas", 10, "bold"),
        padx=14,
        pady=9,
    )
    controls.place(x=598, y=526, width=328, height=52)

    canvas.create_text(50, 572, text="GENERAL: PERSON / CAR / TRUCK / BUS", fill=green_dim, font=("Consolas", 9), anchor="w")
    canvas.create_text(50, 586, text="CUSTOM : BARCODE / CARGO_OBJECT / LOADING_EQUIPMENT", fill=green_dim, font=("Consolas", 9), anchor="w")

    animate()
    root.mainloop()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run", action="store_true", help="Run processing mode. Without this flag launcher opens.")
    parser.add_argument("--custom-model", default=DEFAULT_SETTINGS["custom_model"])
    parser.add_argument("--general-model", default=DEFAULT_SETTINGS["general_model"])
    parser.add_argument("--mode", choices=["custom", "general", "dual"], default=DEFAULT_SETTINGS["mode"])

    parser.add_argument("--source", default="camera", help="camera, image path or video path")
    parser.add_argument("--camera-id", type=int, default=DEFAULT_SETTINGS["camera_id"])

    parser.add_argument("--custom-conf", type=float, default=DEFAULT_SETTINGS["custom_conf"])
    parser.add_argument("--general-conf", type=float, default=DEFAULT_SETTINGS["general_conf"])
    parser.add_argument("--suppress-iou", type=float, default=DEFAULT_SETTINGS["suppress_iou"])
    parser.add_argument("--custom-iou", type=float, default=DEFAULT_SETTINGS["custom_iou"])
    parser.add_argument("--general-iou", type=float, default=DEFAULT_SETTINGS["general_iou"])
    parser.add_argument("--max-det", type=int, default=DEFAULT_SETTINGS["max_det"])

    parser.add_argument("--imgsz", type=int, default=DEFAULT_SETTINGS["imgsz"])
    parser.add_argument("--device", default=DEFAULT_SETTINGS["device"])
    parser.add_argument("--output", default="")

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.run:
        start_launcher()
        return

    if args.source.lower() == "camera":
        process_stream(args, camera=True)
        return

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    if is_image(source):
        process_image(args)
    elif is_video(source):
        process_stream(args, camera=False)
    else:
        raise ValueError(f"Unsupported source type: {source.suffix}")


if __name__ == "__main__":
    main()
