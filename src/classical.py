from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessBundle:
    gray: np.ndarray
    gaussian_blur: np.ndarray
    median_blur: np.ndarray
    threshold_binary: np.ndarray
    threshold_adaptive: np.ndarray
    threshold_otsu: np.ndarray


@dataclass
class EdgeCornerBundle:
    sobel_magnitude: np.ndarray
    canny: np.ndarray
    harris_overlay: np.ndarray


def preprocess_image(image_bgr: np.ndarray) -> PreprocessBundle:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    median_blur = cv2.medianBlur(gray, 5)
    _, threshold_binary = cv2.threshold(gaussian_blur, 127, 255, cv2.THRESH_BINARY)
    threshold_adaptive = cv2.adaptiveThreshold(
        median_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    _, threshold_otsu = cv2.threshold(
        gaussian_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return PreprocessBundle(
        gray=gray,
        gaussian_blur=gaussian_blur,
        median_blur=median_blur,
        threshold_binary=threshold_binary,
        threshold_adaptive=threshold_adaptive,
        threshold_otsu=threshold_otsu,
    )


def detect_edges_and_corners(image_bgr: np.ndarray, gray: np.ndarray) -> EdgeCornerBundle:
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
    sobel_magnitude = np.uint8(np.clip(sobel_magnitude, 0, 255))
    canny = cv2.Canny(gray, 80, 160)

    harris_response = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
    harris_response = cv2.dilate(harris_response, None)
    harris_overlay = image_bgr.copy()
    harris_overlay[harris_response > 0.01 * harris_response.max()] = [0, 0, 255]

    return EdgeCornerBundle(
        sobel_magnitude=sobel_magnitude,
        canny=canny,
        harris_overlay=harris_overlay,
    )


def detect_people_hog(image_bgr: np.ndarray) -> list[dict]:
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, weights = hog.detectMultiScale(
        image_bgr,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )

    detections: list[dict] = []
    for (x, y, w, h), weight in zip(rects, weights):
        detections.append(
            {
                "box": [int(x), int(y), int(x + w), int(y + h)],
                "score": float(weight),
                "label": "person",
                "method": "hog",
            }
        )
    return non_max_suppression(detections, iou_threshold=0.35)


def non_max_suppression(detections: list[dict], iou_threshold: float) -> list[dict]:
    if not detections:
        return []

    boxes = np.array([det["box"] for det in detections], dtype=np.float32)
    scores = np.array([det["score"] for det in detections], dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        index = int(order[0])
        keep.append(index)
        xx1 = np.maximum(x1[index], x1[order[1:]])
        yy1 = np.maximum(y1[index], y1[order[1:]])
        xx2 = np.minimum(x2[index], x2[order[1:]])
        yy2 = np.minimum(y2[index], y2[order[1:]])
        width = np.maximum(0.0, xx2 - xx1 + 1)
        height = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width * height
        overlap = intersection / (areas[index] + areas[order[1:]] - intersection + 1e-6)
        remaining = np.where(overlap <= iou_threshold)[0]
        order = order[remaining + 1]

    return [detections[index] for index in keep]


def apply_privacy_blur(image_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    output = image_bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        roi = output[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        blurred_roi = cv2.GaussianBlur(roi, (31, 31), 0)
        output[y1:y2, x1:x2] = blurred_roi
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
    return output


def apply_spotlight_overlay(image_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    darkened = (image_bgr * 0.35).astype(np.uint8)
    output = darkened.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        roi = image_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        output[y1:y2, x1:x2] = roi
        cv2.rectangle(output, (x1, y1), (x2, y2), (80, 255, 80), 2)
    return output
