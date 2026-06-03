from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import FACE_CLASSES, YOLO_MODEL_PATH, YOLO_MODEL_URLS
from .utils import download_file, ensure_dir


def ensure_yolo_face_model(skip_downloads: bool = False) -> Path:
    """Garante que o modelo YOLO-Face ONNX está disponível."""
    if YOLO_MODEL_PATH.exists():
        # Verifica se o arquivo não está corrompido (> 5MB)
        if YOLO_MODEL_PATH.stat().st_size > 5_000_000:
            return YOLO_MODEL_PATH

    ensure_dir(YOLO_MODEL_PATH.parent)
    if skip_downloads:
        raise FileNotFoundError(
            f"Modelo YOLO-Face não encontrado. Execute primeiro:\n"
            f"  python export_yolo_face.py\n"
            f"Ou coloque o arquivo em: {YOLO_MODEL_PATH}"
        )

    last_error: Exception | None = None
    for url in YOLO_MODEL_URLS:
        try:
            print(f"Baixando modelo YOLO-Face de: {url[:60]}...")
            download_file(url, YOLO_MODEL_PATH)
            # Verifica integridade
            if YOLO_MODEL_PATH.stat().st_size < 5_000_000:
                raise RuntimeError("Download incompleto (arquivo muito pequeno)")
            print(f"✅ Modelo baixado: {YOLO_MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB")
            return YOLO_MODEL_PATH
        except Exception as error:
            print(f"  ❌ Falha: {error}")
            last_error = error

    raise FileNotFoundError(
        f"Não foi possível baixar o modelo YOLO-Face: {last_error}\n"
        f"\nSolução: Execute 'python export_yolo_face.py' para gerar o modelo localmente."
    )


class YOLOFaceOnnxDetector:
    """
    Detector de faces usando YOLO-Face (lindevs) otimizado para OpenCV DNN.
    
    Formato de saída do modelo exportado: (1, 5, 8400)
    - 4 primeiros valores: x_center, y_center, width, height
    - 5º valor: confidence score (apenas 1 classe = face)
    
    NOTA: O modelo foi exportado com:
      - dynamic=False (shapes estáticos)
      - opset=12 (compatível com OpenCV 4.x)
      - simplify=True (grafo simplificado)
    """
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
        """
        Processa saída do modelo lindevs yolov8n-face.
        
        Formato de entrada: (1, 5, 8400) — [x_center, y_center, width, height, conf]
        Precisamos transpor para (8400, 5) para iterar por anchor
        """
        data = np.squeeze(output)
        
        # Verifica e corrige formato da saída
        if data.ndim != 2:
            raise ValueError(f"Dimensão inesperada: {data.ndim}, shape: {data.shape}")
        
        # O modelo exporta como (5, 8400) — precisamos transpor
        if data.shape[0] == 5 and data.shape[1] == 8400:
            data = data.T  # (5, 8400) -> (8400, 5)
        elif data.shape[0] == 8400 and data.shape[1] == 5:
            pass  # Já está correto
        else:
            raise ValueError(
                f"Shape inesperado: {data.shape}. "
                f"Esperado (5, 8400) ou (8400, 5)"
            )

        boxes: list[list[int]] = []
        confidences: list[float] = []

        x_factor = image_width / self.input_size
        y_factor = image_height / self.input_size

        for row in data:
            if len(row) < 5:
                continue

            # Formato: [x_center, y_center, width, height, confidence]
            cx, cy, w, h, confidence = row
            
            confidence = float(confidence)
            
            if confidence < self.confidence_threshold:
                continue

            # Converte para coordenadas da imagem original
            x1 = int((cx - w / 2) * x_factor)
            y1 = int((cy - h / 2) * y_factor)
            x2 = int((cx + w / 2) * x_factor)
            y2 = int((cy + h / 2) * y_factor)

            # Garante coordenadas válidas
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image_width, x2), min(image_height, y2)

            if x2 > x1 and y2 > y1:
                # Formato [x, y, w, h] para NMS
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                confidences.append(confidence)

        # NMS
        if not boxes:
            return []
            
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.confidence_threshold, self.nms_threshold
        )
        
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
                    "label": "face",
                    "method": "yolo-face",
                }
            )
        return results


def draw_yolo_face_detections(image_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Desenha bounding boxes das faces detectadas pelo YOLO-Face."""
    output = image_bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        score = detection["score"]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 128), 2)
        cv2.putText(
            output,
            f"YOLO-Face {score:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 128),
            2,
            cv2.LINE_AA,
        )
    return output