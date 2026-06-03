from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import COCO_CLASSES, YOLO_MODEL_PATH, YOLO_MODEL_URLS
from .utils import download_file, ensure_dir


def ensure_yolo_model(skip_downloads: bool = False) -> Path:
    if YOLO_MODEL_PATH.exists():
        return YOLO_MODEL_PATH

    ensure_dir(YOLO_MODEL_PATH.parent)
    if skip_downloads:
        raise FileNotFoundError(
            f"Modelo YOLO não encontrado. Coloque um arquivo ONNX em {YOLO_MODEL_PATH}"
        )

    last_error: Exception | None = None
    for url in YOLO_MODEL_URLS:
        try:
            download_file(url, YOLO_MODEL_PATH)
            return YOLO_MODEL_PATH
        except Exception as error:
            last_error = error

    raise FileNotFoundError(f"Não foi possível baixar o modelo YOLO: {last_error}")


class YOLOOnnxDetector:
    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.25,  # Reduzido de 0.35
        nms_threshold: float = 0.45,
        input_size: int = 640,
    ) -> None:
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

    def detect(self, image_bgr: np.ndarray) -> list[dict]:
        height, width = image_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image_bgr,
            scalefactor=1.0 / 255.0,
            size=(self.input_size, self.input_size),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        predictions = self.net.forward()
        detections = self._postprocess(predictions, width, height)
        return detections

    def _postprocess(self, output: np.ndarray, image_width: int, image_height: int) -> list[dict]:
        data = np.squeeze(output)

        if data.ndim != 2:
            raise ValueError(f"Saída YOLO inesperada com shape {output.shape}")

        if data.shape[0] in (84, 85):
            data = data.T

        boxes: list[list[int]] = []
        confidences: list[float] = []
        class_ids: list[int] = []

        x_factor = image_width / self.input_size
        y_factor = image_height / self.input_size

        for row in data:
            if len(row) < 6:
                continue

            if len(row) == 85:
                objectness = float(row[4])
                class_scores = row[5:]
                class_id = int(np.argmax(class_scores))
                confidence = objectness * float(class_scores[class_id])
                start_index = 0
            else:
                class_scores = row[4:]
                class_id = int(np.argmax(class_scores))
                confidence = float(class_scores[class_id])
                start_index = 0

            if class_id >= len(COCO_CLASSES):
                continue
            
            # Agora detecta tanto "person" quanto potencialmente outras classes relevantes
            # Para blur facial, "person" é suficiente como contexto, mas não substitui detecção facial
            if COCO_CLASSES[class_id] not in ("person",):
                continue
                
            if confidence < self.confidence_threshold:
                continue

            cx, cy, w, h = row[start_index:start_index + 4]
            x1 = int((cx - w / 2) * x_factor)
            y1 = int((cy - h / 2) * y_factor)
            x2 = int((cx + w / 2) * x_factor)
            y2 = int((cy + h / 2) * y_factor)

            boxes.append([x1, y1, max(0, x2 - x1), max(0, y2 - y1)])
            confidences.append(confidence)
            class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        results: list[dict] = []
        if len(indices) == 0:
            return results

        flat_indices = np.array(indices).reshape(-1)
        for index in flat_indices:
            x, y, w, h = boxes[int(index)]
            results.append(
                {
                    "box": [int(x), int(y), int(x + w), int(y + h)],
                    "score": float(confidences[int(index)]),
                    "label": COCO_CLASSES[class_ids[int(index)]],
                    "method": "yolo",
                }
            )
        return results


def draw_yolo_detections(image_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    output = image_bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        score = detection["score"]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(
            output,
            f"YOLO person {score:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
    return output