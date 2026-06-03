from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessBundle:
    """Resultados do pré-processamento clássico aplicado ao frame-chave."""

    gray: np.ndarray
    gaussian_blur: np.ndarray
    threshold_adaptive: np.ndarray
    threshold_otsu: np.ndarray
    primary_threshold: np.ndarray


@dataclass
class EdgeBundle:
    """Resultados dos métodos clássicos de bordas usados para análise."""

    sobel_magnitude: np.ndarray
    canny: np.ndarray
    primary_edges: np.ndarray


def preprocess_image(image_bgr: np.ndarray) -> PreprocessBundle:
    """Converte o frame para cinza, suaviza ruído e calcula thresholds comparativos."""

    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gaussian_blur = cv2.GaussianBlur(gray_image, (7, 7), 1.2)
    threshold_adaptive = cv2.adaptiveThreshold(
        gaussian_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        3,
    )
    _, threshold_otsu = cv2.threshold(
        gaussian_blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return PreprocessBundle(
        gray=gray_image,
        gaussian_blur=gaussian_blur,
        threshold_adaptive=threshold_adaptive,
        threshold_otsu=threshold_otsu,
        primary_threshold=threshold_otsu,
    )


def detect_edges(image_bgr: np.ndarray, gray_image: np.ndarray) -> EdgeBundle:
    """Calcula Sobel como borda principal e Canny como método comparativo."""

    if len(gray_image.shape) == 3:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

    denoised_gray = cv2.GaussianBlur(gray_image, (5, 5), 1.0)
    median_intensity = float(np.median(denoised_gray))
    lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
    upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))
    if lower_threshold >= upper_threshold:
        lower_threshold, upper_threshold = 50, 150

    sobel_x = cv2.Sobel(denoised_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(denoised_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
    sobel_magnitude = np.uint8(np.clip(sobel_magnitude, 0, 255))
    canny_edges = cv2.Canny(
        denoised_gray,
        lower_threshold,
        upper_threshold,
        apertureSize=3,
        L2gradient=True,
    )

    return EdgeBundle(
        sobel_magnitude=sobel_magnitude,
        canny=canny_edges,
        primary_edges=sobel_magnitude,
    )
