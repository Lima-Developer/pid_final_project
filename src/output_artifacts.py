from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .classical import EdgeBundle, PreprocessBundle
from .face_privacy import detect_faces_haar, draw_face_overlay
from .utils import ensure_dir
from .visualization import build_panel
from .yolo_face_onnx import YOLOFaceOnnxDetector, draw_yolo_face_detections


def save_preprocessing_outputs(
    output_dir: Path,
    frame_bgr: np.ndarray,
    preprocess: PreprocessBundle,
    edges: EdgeBundle,
) -> None:
    """Salva imagens individuais e painel do pré-processamento clássico."""

    preprocessing_dir = ensure_dir(output_dir / "preprocessing")
    outputs = [
        ("01_original_frame.png", frame_bgr),
        ("02_grayscale.png", preprocess.gray),
        ("03_gaussian_blur.png", preprocess.gaussian_blur),
        ("04_threshold_otsu.png", preprocess.threshold_otsu),
        ("05_threshold_adaptive_comparison.png", preprocess.threshold_adaptive),
        ("06_sobel_magnitude.png", edges.sobel_magnitude),
        ("07_canny_comparison.png", edges.canny),
    ]
    for filename, image in outputs:
        cv2.imwrite(str(preprocessing_dir / filename), image)

    build_panel(
        [
            ("1. Original", frame_bgr),
            ("2. Escala de cinza", preprocess.gray),
            ("3. Gaussian Blur", preprocess.gaussian_blur),
            ("4. Otsu", preprocess.threshold_otsu),
            ("5. Sobel", edges.sobel_magnitude),
            ("6. Canny comparativo", edges.canny),
        ],
        preprocessing_dir / "preprocessing_panel.png",
        cell_width=480,
        cell_height=270,
        columns=3,
    )


def save_roi_mask_blend_panel(
    output_dir: Path,
    frame_bgr: np.ndarray,
    face_boxes: list[list[int]],
    face_mask: np.ndarray,
    final_result: np.ndarray,
) -> None:
    """Salva painel 2x2 com ROI, máscara e blend final."""

    roi_image = frame_bgr.copy()
    for index, (left, top, right, bottom) in enumerate(face_boxes, start=1):
        cv2.rectangle(roi_image, (left, top), (right, bottom), (0, 255, 255), 4)
        cv2.putText(
            roi_image,
            f"ROI {index}",
            (left, max(35, top - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )

    colored_mask = cv2.applyColorMap(face_mask, cv2.COLORMAP_JET)
    mask_overlay = cv2.addWeighted(frame_bgr, 0.65, colored_mask, 0.35, 0)
    build_panel(
        [
            ("1. ROI detectada", roi_image),
            ("2. Máscara binária", face_mask),
            ("3. Máscara + frame", mask_overlay),
            ("4. Blend final", final_result),
        ],
        output_dir / "roi_mask_blend_2x2.png",
        cell_width=640,
        cell_height=360,
        columns=2,
    )


def save_yolo_face_panel(
    output_dir: Path,
    frame_bgr: np.ndarray,
    yolo_detector: YOLOFaceOnnxDetector | None,
) -> int | None:
    """Salva painel comparativo entre frame original e detecção YOLO-Face."""

    if yolo_detector is None:
        return None

    detections = yolo_detector.detect(frame_bgr)
    yolo_frame = draw_yolo_face_detections(frame_bgr, detections)
    build_panel(
        [
            ("1. Frame original", frame_bgr),
            ("2. YOLO-Face", yolo_frame),
        ],
        output_dir / "yolo_face_panel.png",
        cell_width=640,
        cell_height=360,
        columns=2,
    )
    return len(detections)


def save_haar_vs_yolo_comparison(
    output_dir: Path,
    frame_bgr: np.ndarray,
    yolo_detector: YOLOFaceOnnxDetector | None,
) -> None:
    """Salva comparação visual direta entre Haar Cascade e YOLO-Face."""

    haar_boxes = detect_faces_haar(frame_bgr)
    haar_frame = draw_face_overlay(frame_bgr, haar_boxes)

    if yolo_detector is None:
        return

    yolo_detections = yolo_detector.detect(frame_bgr)
    yolo_frame = draw_yolo_face_detections(frame_bgr, yolo_detections)
    build_panel(
        [
            (f"Haar Cascade ({len(haar_boxes)} caixa(s))", haar_frame),
            (f"YOLO-Face ({len(yolo_detections)} caixa(s))", yolo_frame),
        ],
        output_dir / "haar_vs_yolo_face_comparison.png",
        cell_width=640,
        cell_height=360,
        columns=2,
    )
