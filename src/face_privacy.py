from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .config import HAAR_CASCADE_DIR


@dataclass
class FacePrivacyResult:
    """Resultado visual da composição de privacidade facial."""

    detected_faces: list[list[int]]
    face_overlay: np.ndarray
    privacy_result: np.ndarray
    face_mask: np.ndarray


def clamp_box(box: list[int], image_width: int, image_height: int) -> list[int]:
    """Limita uma caixa às dimensões válidas da imagem."""

    left, top, right, bottom = box
    return [
        max(0, min(image_width - 1, int(left))),
        max(0, min(image_height - 1, int(top))),
        max(0, min(image_width - 1, int(right))),
        max(0, min(image_height - 1, int(bottom))),
    ]


def expand_box(box: list[int], image_width: int, image_height: int, scale: float = 1.20) -> list[int]:
    """Expande a caixa facial para cobrir a região sensível com margem de segurança."""

    left, top, right, bottom = box
    box_width = right - left
    box_height = bottom - top
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    expanded_width = box_width * scale
    expanded_height = box_height * scale
    return clamp_box(
        [
            int(center_x - expanded_width / 2),
            int(center_y - expanded_height / 2),
            int(center_x + expanded_width / 2),
            int(center_y + expanded_height / 2),
        ],
        image_width,
        image_height,
    )


def box_area(box: list[int]) -> int:
    """Calcula a área de uma caixa no formato [x1, y1, x2, y2]."""

    left, top, right, bottom = box
    return max(0, right - left) * max(0, bottom - top)


def iou(first_box: list[int], second_box: list[int]) -> float:
    """Calcula Intersection over Union entre duas caixas."""

    first_left, first_top, first_right, first_bottom = first_box
    second_left, second_top, second_right, second_bottom = second_box
    intersection_left = max(first_left, second_left)
    intersection_top = max(first_top, second_top)
    intersection_right = min(first_right, second_right)
    intersection_bottom = min(first_bottom, second_bottom)
    intersection_area = max(0, intersection_right - intersection_left) * max(0, intersection_bottom - intersection_top)
    union_area = box_area(first_box) + box_area(second_box) - intersection_area
    return intersection_area / union_area if union_area else 0.0


def overlap_over_smaller_box(first_box: list[int], second_box: list[int]) -> float:
    """Mede quanto uma caixa cobre a menor das duas caixas comparadas."""

    first_left, first_top, first_right, first_bottom = first_box
    second_left, second_top, second_right, second_bottom = second_box
    intersection_left = max(first_left, second_left)
    intersection_top = max(first_top, second_top)
    intersection_right = min(first_right, second_right)
    intersection_bottom = min(first_bottom, second_bottom)
    intersection_area = max(0, intersection_right - intersection_left) * max(0, intersection_bottom - intersection_top)
    smaller_area = min(box_area(first_box), box_area(second_box))
    return intersection_area / smaller_area if smaller_area else 0.0


def merge_face_boxes(face_boxes: list[list[int]], iou_threshold: float = 0.15) -> list[list[int]]:
    """Remove caixas duplicadas ou fortemente sobrepostas."""

    valid_boxes = [box for box in face_boxes if box_area(box) >= 80 and box[2] > box[0] and box[3] > box[1]]
    ordered_boxes = sorted(valid_boxes, key=box_area, reverse=True)
    merged_boxes: list[list[int]] = []
    for box in ordered_boxes:
        should_keep = all(
            iou(box, existing_box) < iou_threshold and overlap_over_smaller_box(box, existing_box) < 0.35
            for existing_box in merged_boxes
        )
        if should_keep:
            merged_boxes.append(box)
    return sorted(merged_boxes, key=lambda item: (item[1], item[0]))


def merge_detections(yolo_boxes: list[list[int]], haar_boxes: list[list[int]]) -> list[list[int]]:
    """Combina caixas do YOLO-Face com caixas Haar ainda não cobertas."""

    if not yolo_boxes:
        return haar_boxes
    if not haar_boxes:
        return yolo_boxes

    merged_boxes = list(yolo_boxes)
    for haar_box in haar_boxes:
        is_covered = any(iou(haar_box, yolo_box) > 0.3 for yolo_box in yolo_boxes)
        if not is_covered:
            merged_boxes.append(haar_box)
    return merge_face_boxes(merged_boxes)


@lru_cache(maxsize=8)
def load_cascade(cascade_path: str) -> cv2.CascadeClassifier:
    """Carrega um classificador Haar Cascade local."""

    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise FileNotFoundError(f"Classificador Haar inválido: {cascade_path}")
    return cascade


def resolve_cascade_path(cascade_name: str) -> Path | None:
    """Retorna o caminho local de um XML Haar Cascade."""

    cascade_path = HAAR_CASCADE_DIR / cascade_name
    return cascade_path if cascade_path.exists() else None


def preprocess_for_haar(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gera variações em cinza para aumentar a chance de detecção Haar."""

    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    clahe_image = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray_image)
    gamma = 1.2
    inverse_gamma = 1.0 / gamma
    gamma_table = np.array([((value / 255.0) ** inverse_gamma) * 255 for value in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(gray_image, gamma_table)
    return gray_image, equalized_image, clahe_image, gamma_corrected


def detect_faces_haar(
    image_bgr: np.ndarray,
    max_detection_width: int = 640,
    min_face_size: tuple[int, int] = (24, 24),
) -> list[list[int]]:
    """Detecta faces com Haar Cascade no frame inteiro."""

    image_height, image_width = image_bgr.shape[:2]
    detection_scale = 1.0
    detection_image = image_bgr
    if image_width > max_detection_width:
        detection_scale = max_detection_width / image_width
        detection_height = int(image_height * detection_scale)
        detection_image = cv2.resize(detection_image, (max_detection_width, detection_height), interpolation=cv2.INTER_AREA)

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
        for gray_variant in preprocess_for_haar(detection_image):
            faces = cascade.detectMultiScale(
                gray_variant,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=min_face_size,
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            for face_left, face_top, face_width, face_height in faces:
                scaled_box = [
                    int(face_left / detection_scale),
                    int(face_top / detection_scale),
                    int((face_left + face_width) / detection_scale),
                    int((face_top + face_height) / detection_scale),
                ]
                face_boxes.append(expand_box(scaled_box, image_width, image_height, scale=1.20))

    return merge_face_boxes(face_boxes)


def detect_faces_haar_in_roi(
    image_bgr: np.ndarray,
    max_detection_width: int = 640,
    min_face_size: tuple[int, int] = (24, 24),
) -> list[list[int]]:
    """Detecta faces com Haar Cascade dentro de uma região já recortada."""

    return detect_faces_haar(image_bgr, max_detection_width=max_detection_width, min_face_size=min_face_size)


def detect_faces(image_bgr: np.ndarray, yolo_detector=None, roi_tracker=None) -> list[list[int]]:
    """Executa detecção híbrida: YOLO-Face, Haar em ROI e Haar no frame inteiro."""

    image_height, image_width = image_bgr.shape[:2]
    yolo_boxes: list[list[int]] = []
    if yolo_detector is not None:
        try:
            yolo_boxes = [detection["box"] for detection in yolo_detector.detect(image_bgr)]
        except Exception:
            yolo_boxes = []

    if yolo_boxes:
        expanded_yolo_boxes = [expand_box(box, image_width, image_height, scale=1.20) for box in yolo_boxes]
        merged_yolo_boxes = merge_face_boxes(expanded_yolo_boxes)
        if roi_tracker is not None:
            roi_tracker.update_from_detections(merged_yolo_boxes)
        return merged_yolo_boxes

    if roi_tracker is not None and roi_tracker.rois:
        roi_boxes = roi_tracker.detect_faces_in_rois(image_bgr, detector_function=detect_faces_haar_in_roi)
        if roi_boxes:
            expanded_roi_boxes = [expand_box(box, image_width, image_height, scale=1.20) for box in roi_boxes]
            return merge_face_boxes(expanded_roi_boxes)

    return detect_faces_haar(image_bgr)


def build_face_mask(image_shape: tuple[int, int, int], face_boxes: list[list[int]], feather: int = 51) -> np.ndarray:
    """Cria uma máscara suave para as regiões faciais detectadas."""

    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for left, top, right, bottom in face_boxes:
        cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)
    feather = max(3, feather if feather % 2 == 1 else feather + 1)
    return cv2.GaussianBlur(mask, (feather, feather), 0)


def draw_face_overlay(image_bgr: np.ndarray, face_boxes: list[list[int]]) -> np.ndarray:
    """Desenha as caixas faciais detectadas no frame."""

    output = image_bgr.copy()
    for index, (left, top, right, bottom) in enumerate(face_boxes, start=1):
        cv2.rectangle(output, (left, top), (right, bottom), (255, 80, 80), 3)
        cv2.putText(
            output,
            f"face {index}",
            (left, max(24, top - 8)),
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
    """Aplica blur e pixelização nas regiões faciais controladas pela máscara."""

    mask = build_face_mask(image_bgr.shape, face_boxes)
    blur_kernel = max(3, blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1)
    blurred_image = cv2.GaussianBlur(image_bgr, (blur_kernel, blur_kernel), 0)
    pixelated_image = image_bgr.copy()

    for left, top, right, bottom in face_boxes:
        roi = pixelated_image[top:bottom, left:right]
        if roi.size == 0:
            continue
        roi_height, roi_width = roi.shape[:2]
        small_width = max(1, roi_width // pixelation_factor)
        small_height = max(1, roi_height // pixelation_factor)
        reduced_roi = cv2.resize(roi, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
        pixelated_roi = cv2.resize(reduced_roi, (roi_width, roi_height), interpolation=cv2.INTER_NEAREST)
        pixelated_image[top:bottom, left:right] = pixelated_roi

    anonymized_image = cv2.addWeighted(blurred_image, 0.95, pixelated_image, 0.05, 0)
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    output = (alpha * anonymized_image + (1.0 - alpha) * image_bgr).astype(np.uint8)
    return output, mask


def build_privacy_composition(image_bgr: np.ndarray, face_boxes: list[list[int]]) -> FacePrivacyResult:
    """Gera visualização de detecção, máscara e resultado anonimizado."""

    face_overlay = draw_face_overlay(image_bgr, face_boxes)
    privacy_result, face_mask = blur_faces_with_mask(image_bgr, face_boxes)
    return FacePrivacyResult(
        detected_faces=face_boxes,
        face_overlay=face_overlay,
        privacy_result=privacy_result,
        face_mask=face_mask,
    )
