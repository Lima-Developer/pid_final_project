from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import YOLO_FACE_MODEL_PATH, YOLO_FACE_MODEL_URLS
from .utils import download_file, ensure_dir


def ensure_yolo_face_model(skip_downloads: bool = False) -> Path:
    """Garante que o modelo YOLO-Face ONNX esteja disponível para o OpenCV DNN."""

    if YOLO_FACE_MODEL_PATH.exists() and YOLO_FACE_MODEL_PATH.stat().st_size > 5_000_000:
        return YOLO_FACE_MODEL_PATH

    ensure_dir(YOLO_FACE_MODEL_PATH.parent)
    if skip_downloads:
        raise FileNotFoundError(f"Modelo YOLO-Face não encontrado em {YOLO_FACE_MODEL_PATH}")

    last_error: Exception | None = None
    for url in YOLO_FACE_MODEL_URLS:
        try:
            print(f"Baixando modelo YOLO-Face: {url}")
            download_file(url, YOLO_FACE_MODEL_PATH)
            if YOLO_FACE_MODEL_PATH.stat().st_size <= 5_000_000:
                raise RuntimeError("Arquivo ONNX baixado com tamanho inesperado.")
            return YOLO_FACE_MODEL_PATH
        except Exception as error:
            last_error = error

    raise FileNotFoundError(f"Não foi possível obter o modelo YOLO-Face: {last_error}")


class YOLOFaceOnnxDetector:
    """Executa inferência do YOLO-Face ONNX pelo módulo OpenCV DNN."""

    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.3,
        nms_threshold: float = 0.4,
        input_size: int = 640,
    ) -> None:
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

    def detect(self, image_bgr: np.ndarray) -> list[dict]:
        """Retorna caixas, scores e classe para faces detectadas no frame."""

        image_height, image_width = image_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image_bgr,
            scalefactor=1.0 / 255.0,
            size=(self.input_size, self.input_size),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        predictions = self.net.forward()
        return self._postprocess(predictions, image_width, image_height)

    def _postprocess(self, output: np.ndarray, image_width: int, image_height: int) -> list[dict]:
        """Converte a saída ONNX do YOLO-Face em bounding boxes no frame original."""

        output_data = np.squeeze(output)
        if output_data.ndim != 2:
            raise ValueError(f"Saída YOLO-Face inesperada com shape {output.shape}")

        if output_data.shape[0] == 5:
            output_data = output_data.T
        if output_data.shape[1] != 5:
            raise ValueError(f"Saída YOLO-Face incompatível com shape {output_data.shape}")

        width_factor = image_width / self.input_size
        height_factor = image_height / self.input_size
        boxes: list[list[int]] = []
        confidences: list[float] = []

        for row in output_data:
            center_x, center_y, box_width, box_height, confidence = row
            confidence = float(confidence)
            if confidence < self.confidence_threshold:
                continue

            left = int((center_x - box_width / 2) * width_factor)
            top = int((center_y - box_height / 2) * height_factor)
            right = int((center_x + box_width / 2) * width_factor)
            bottom = int((center_y + box_height / 2) * height_factor)
            left = max(0, left)
            top = max(0, top)
            right = min(image_width, right)
            bottom = min(image_height, bottom)
            if right > left and bottom > top:
                boxes.append([left, top, right - left, bottom - top])
                confidences.append(confidence)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.confidence_threshold,
            self.nms_threshold,
        )
        detections: list[dict] = []
        for index in np.array(indices).reshape(-1):
            left, top, width, height = boxes[int(index)]
            detections.append(
                {
                    "box": [left, top, left + width, top + height],
                    "score": float(confidences[int(index)]),
                    "label": "face",
                    "method": "yolo-face",
                }
            )
        return detections


def draw_yolo_face_detections(image_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Desenha as caixas retornadas pelo YOLO-Face."""

    output = image_bgr.copy()
    for detection in detections:
        left, top, right, bottom = detection["box"]
        score = detection["score"]
        cv2.rectangle(output, (left, top), (right, bottom), (0, 255, 128), 2)
        cv2.putText(
            output,
            f"YOLO-Face {score:.2f}",
            (left, max(20, top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 128),
            2,
            cv2.LINE_AA,
        )
    return output
