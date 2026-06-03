from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .config import HAAR_CASCADE_DIR


@dataclass
class FacePrivacyResult:
    detected_faces: list[list[int]]
    face_overlay: np.ndarray
    privacy_result: np.ndarray
    face_mask: np.ndarray


def clamp_box(box: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box
    return [
        max(0, min(width - 1, int(x1))),
        max(0, min(height - 1, int(y1))),
        max(0, min(width - 1, int(x2))),
        max(0, min(height - 1, int(y2))),
    ]


def expand_box(box: list[int], width: int, height: int, scale: float = 1.20) -> list[int]:
    """Expande a caixa do rosto para cobrir a região facial com margem de privacidade."""
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    new_width = box_width * scale
    new_height = box_height * scale
    return clamp_box(
        [
            int(center_x - new_width / 2),
            int(center_y - new_height / 2),
            int(center_x + new_width / 2),
            int(center_y + new_height / 2),
        ],
        width,
        height,
    )


def box_area(box: list[int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_is_valid(box: list[int], min_area: int = 80) -> bool:
    x1, y1, x2, y2 = box
    return x2 > x1 and y2 > y1 and box_area(box) >= min_area


def iou(box_a: list[int], box_b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)
    intersection = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
    union = box_area(box_a) + box_area(box_b) - intersection
    return intersection / union if union else 0.0


def overlap_over_smaller_box(box_a: list[int], box_b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)
    intersection = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
    smaller_area = min(box_area(box_a), box_area(box_b))
    return intersection / smaller_area if smaller_area else 0.0


def merge_face_boxes(face_boxes: list[list[int]], iou_threshold: float = 0.15) -> list[list[int]]:
    ordered = sorted([box for box in face_boxes if box_is_valid(box)], key=box_area, reverse=True)
    merged: list[list[int]] = []
    for box in ordered:
        if all(
            iou(box, existing) < iou_threshold and overlap_over_smaller_box(box, existing) < 0.35
            for existing in merged
        ):
            merged.append(box)
    return sorted(merged, key=lambda item: (item[1], item[0]))


def merge_detections(yolo_boxes: list[list[int]], haar_boxes: list[list[int]]) -> list[list[int]]:
    """Fusão: YOLO-Face tem prioridade. Haar só adiciona faces não cobertas."""
    if not yolo_boxes:
        return haar_boxes
    if not haar_boxes:
        return yolo_boxes
    
    merged = list(yolo_boxes)
    for haar_box in haar_boxes:
        covered = any(iou(haar_box, yolo_box) > 0.3 for yolo_box in yolo_boxes)
        if not covered:
            merged.append(haar_box)
    
    return merge_face_boxes(merged)


@lru_cache(maxsize=8)
def load_cascade(cascade_path: str) -> cv2.CascadeClassifier:
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise FileNotFoundError(f"Classificador Haar não encontrado ou inválido: {cascade_path}")
    return cascade


def resolve_cascade_path(cascade_name: str) -> Path | None:
    local_path = HAAR_CASCADE_DIR / cascade_name
    if local_path.exists():
        return local_path
    return None


def detect_faces_haar(
    image_bgr: np.ndarray,
    max_detection_width: int = 640,
    min_face_size: tuple[int, int] = (24, 24),
) -> list[list[int]]:
    """Haar Cascade no frame inteiro — fallback de último recurso."""
    height, width = image_bgr.shape[:2]
    detection_scale = 1.0
    detection_image = image_bgr

    if width > max_detection_width:
        detection_scale = max_detection_width / width
        detection_height = int(height * detection_scale)
        detection_image = cv2.resize(
            image_bgr,
            (max_detection_width, detection_height),
            interpolation=cv2.INTER_AREA,
        )

    detection_gray = cv2.cvtColor(detection_image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(detection_gray)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(detection_gray)
    
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(detection_gray, table)

    cascade_names = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_profileface.xml",
        "haarcascade_frontalface_alt.xml",
        "haarcascade_frontalface_alt_tree.xml",
    ]

    face_boxes: list[list[int]] = []
    
    for cascade_name in cascade_names:
        cascade_path = resolve_cascade_path(cascade_name)
        if cascade_path is None:
            continue
        cascade = load_cascade(str(cascade_path))
            
        for image_gray in (detection_gray, equalized, clahe, gamma_corrected):
            faces = cascade.detectMultiScale(
                image_gray,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=min_face_size,
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            for face_x, face_y, face_width, face_height in faces:
                box = [
                    int(face_x / detection_scale),
                    int(face_y / detection_scale),
                    int((face_x + face_width) / detection_scale),
                    int((face_y + face_height) / detection_scale),
                ]
                expanded_box = expand_box(box, width, height, scale=1.20)
                face_boxes.append(expanded_box)

    return merge_face_boxes(face_boxes)


def detect_faces_haar_in_roi(
    image_bgr: np.ndarray,
    max_detection_width: int = 640,
    min_face_size: tuple[int, int] = (24, 24),
) -> list[list[int]]:
    """Haar Cascade para imagem já cortada (ROI)."""
    height, width = image_bgr.shape[:2]
    detection_scale = 1.0
    detection_image = image_bgr

    if width > max_detection_width:
        detection_scale = max_detection_width / width
        detection_height = int(height * detection_scale)
        detection_image = cv2.resize(
            image_bgr,
            (max_detection_width, detection_height),
            interpolation=cv2.INTER_AREA,
        )

    detection_gray = cv2.cvtColor(detection_image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(detection_gray)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(detection_gray)
    
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(detection_gray, table)

    cascade_names = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_profileface.xml",
        "haarcascade_frontalface_alt.xml",
        "haarcascade_frontalface_alt_tree.xml",
    ]

    face_boxes: list[list[int]] = []
    
    for cascade_name in cascade_names:
        cascade_path = resolve_cascade_path(cascade_name)
        if cascade_path is None:
            continue
        cascade = load_cascade(str(cascade_path))
            
        for image_gray in (detection_gray, equalized, clahe, gamma_corrected):
            faces = cascade.detectMultiScale(
                image_gray,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=min_face_size,
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            for face_x, face_y, face_width, face_height in faces:
                box = [
                    int(face_x / detection_scale),
                    int(face_y / detection_scale),
                    int((face_x + face_width) / detection_scale),
                    int((face_y + face_height) / detection_scale),
                ]
                expanded_box = expand_box(box, width, height, scale=1.20)
                face_boxes.append(expanded_box)

    return merge_face_boxes(face_boxes)


def detect_faces(
    image_bgr: np.ndarray,
    yolo_detector=None,
    roi_tracker=None,
) -> list[list[int]]:
    """
    Detecção híbrida com ROI:
    1. YOLO-Face (melhor precisão)
    2. Haar Cascade em ROIs (sem falsos positivos)
    3. Haar Cascade no frame inteiro (último recurso)
    """
    height, width = image_bgr.shape[:2]
    
    # 1. YOLO-Face primário
    if yolo_detector is not None:
        try:
            detections = yolo_detector.detect(image_bgr)
            yolo_boxes = [det["box"] for det in detections]
            
            if yolo_boxes:
                yolo_boxes = [expand_box(box, width, height, scale=1.20) for box in yolo_boxes]
                yolo_boxes = merge_face_boxes(yolo_boxes)
                
                if roi_tracker is not None:
                    roi_tracker.update_from_detections(yolo_boxes)
                
                return yolo_boxes
            
        except Exception:
            pass

    # 2. Haar Cascade com ROI
    if roi_tracker is not None and roi_tracker.rois:
        roi_boxes = roi_tracker.detect_faces_in_rois(
            image_bgr,
            detector_function=detect_faces_haar_in_roi,
        )
        if roi_boxes:
            global_boxes = [expand_box(box, width, height, scale=1.20) for box in roi_boxes]
            global_boxes = merge_face_boxes(global_boxes)
            return global_boxes

    # 3. Fallback: Haar no frame inteiro
    return detect_faces_haar(image_bgr)


def build_face_mask(image_shape: tuple[int, int, int], face_boxes: list[list[int]], feather: int = 51) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for x1, y1, x2, y2 in face_boxes:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    feather = max(3, feather if feather % 2 == 1 else feather + 1)
    return cv2.GaussianBlur(mask, (feather, feather), 0)


def draw_face_overlay(image_bgr: np.ndarray, face_boxes: list[list[int]]) -> np.ndarray:
    output = image_bgr.copy()
    for index, (x1, y1, x2, y2) in enumerate(face_boxes, start=1):
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 80, 80), 3)
        cv2.putText(
            output,
            f"face {index}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 80, 80),
            2,
            cv2.LINE_AA,
        )
    return output


def blur_faces_with_mask(
    image_bgr: np.ndarray,
    face_boxes: list[list[int]],
    blur_kernel: int = 399,
    pixelation_factor: int = 14,
) -> tuple[np.ndarray, np.ndarray]:
    """Blur extremo + pixelação agressiva para máxima privacidade."""
    mask = build_face_mask(image_bgr.shape, face_boxes)
    blur_kernel = max(3, blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1)
    
    strong_blur = cv2.GaussianBlur(image_bgr, (blur_kernel, blur_kernel), 0)
    pixelated = image_bgr.copy()

    for x1, y1, x2, y2 in face_boxes:
        roi = pixelated[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        roi_height, roi_width = roi.shape[:2]
        small_width = max(1, roi_width // pixelation_factor)
        small_height = max(1, roi_height // pixelation_factor)
        small = cv2.resize(roi, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
        pixelated_roi = cv2.resize(small, (roi_width, roi_height), interpolation=cv2.INTER_NEAREST)
        pixelated[y1:y2, x1:x2] = pixelated_roi

    anonymized = cv2.addWeighted(strong_blur, 0.95, pixelated, 0.05, 0)
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    output = (alpha * anonymized + (1.0 - alpha) * image_bgr).astype(np.uint8)

    return output, mask


def build_privacy_composition(image_bgr: np.ndarray, face_boxes: list[list[int]]) -> FacePrivacyResult:
    face_overlay = draw_face_overlay(image_bgr, face_boxes)
    privacy_result, face_mask = blur_faces_with_mask(image_bgr, face_boxes)
    return FacePrivacyResult(
        detected_faces=face_boxes,
        face_overlay=face_overlay,
        privacy_result=privacy_result,
        face_mask=face_mask,
    )
