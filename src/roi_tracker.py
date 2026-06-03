from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FaceROI:
    """Região de interesse para um rosto específico."""
    name: str           # "mulher_esquerda", "homem_direita", etc.
    x1: int
    y1: int
    x2: int
    y2: int
    active: bool = True
    frames_missing: int = 0
    max_frames_missing: int = 90  # ~3 segundos a 30fps
    
    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)
    
    def contains(self, box: list[int]) -> bool:
        """Verifica se uma bounding box está contida ou intersecta fortemente esta ROI."""
        bx1, by1, bx2, by2 = box
        # Interseção
        ix1 = max(self.x1, bx1)
        iy1 = max(self.y1, by1)
        ix2 = min(self.x2, bx2)
        iy2 = min(self.y2, by2)
        
        if ix2 <= ix1 or iy2 <= iy1:
            return False
        
        inter_area = (ix2 - ix1) * (iy2 - iy1)
        box_area = (bx2 - bx1) * (by2 - by1)
        return inter_area / box_area > 0.3  # 30% de overlap mínimo
    
    def expand(self, factor: float = 1.2) -> tuple[int, int, int, int]:
        """Expande a ROI para dar margem de movimento."""
        cx = (self.x1 + self.x2) / 2
        cy = (self.y1 + self.y2) / 2
        w = (self.x2 - self.x1) * factor
        h = (self.y2 - self.y1) * factor
        return (
            int(cx - w / 2),
            int(cy - h / 2),
            int(cx + w / 2),
            int(cy + h / 2),
        )


class ROIFaceTracker:
    """
    Tracker que usa ROIs pré-definidas para guiar a detecção.
    
    Ideal para vídeos de entrevista com posições fixas de câmera,
    onde os rostos aparecem em regiões previsíveis do frame.
    """
    def __init__(
        self,
        rois: list[FaceROI] | None = None,
        roi_expansion: float = 1.3,
    ) -> None:
        self.rois = rois or []
        self.roi_expansion = roi_expansion
        self.frame_count = 0
    
    def add_roi(self, name: str, x1: int, y1: int, x2: int, y2: int) -> None:
        """Adiciona uma ROI manualmente."""
        self.rois.append(FaceROI(name=name, x1=x1, y1=y1, x2=x2, y2=y2))
    
    def detect_faces_in_rois(
        self,
        image_bgr: np.ndarray,
        detector_function,  # Função que recebe (image_roi) -> list[boxes]
    ) -> list[list[int]]:
        """
        Executa detecção facial apenas dentro das ROIs ativas.
        
        Args:
            image_bgr: Frame completo
            detector_function: Função de detecção (Haar, YOLO, etc.)
        
        Returns:
            Lista de bounding boxes [x1, y1, x2, y2] no espaço da imagem original
        """
        all_boxes: list[list[int]] = []
        height, width = image_bgr.shape[:2]
        
        for roi in self.rois:
            if not roi.active:
                continue
            
            # Expande ROI para dar margem de movimento
            rx1, ry1, rx2, ry2 = roi.expand(self.roi_expansion)
            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2, ry2 = min(width, rx2), min(height, ry2)
            
            if rx2 <= rx1 or ry2 <= ry1:
                continue
            
            # Extrai ROI da imagem
            roi_image = image_bgr[ry1:ry2, rx1:rx2]
            if roi_image.size == 0:
                continue
            
            # Detecta faces DENTRO da ROI
            roi_boxes = detector_function(roi_image)
            
            # Converte coordenadas da ROI para coordenadas da imagem original
            for box in roi_boxes:
                bx1, by1, bx2, by2 = box
                global_box = [
                    rx1 + bx1,
                    ry1 + by1,
                    rx1 + bx2,
                    ry1 + by2,
                ]
                all_boxes.append(global_box)
            
            # Atualiza estado da ROI
            if roi_boxes:
                roi.frames_missing = 0
            else:
                roi.frames_missing += 1
                if roi.frames_missing > roi.max_frames_missing:
                    roi.active = False  # Desativa se não encontrar por muito tempo
        
        self.frame_count += 1
        return all_boxes
    
    def update_from_detections(self, detections: list[list[int]]) -> None:
        """
        Atualiza as ROIs baseado em detecções bem-sucedidas.
        Isso permite que as ROIs 'sigam' os rostos levemente.
        """
        for roi in self.rois:
            for box in detections:
                if roi.contains(box):
                    # Atualiza ROI para centralizar no rosto detectado
                    bx1, by1, bx2, by2 = box
                    cx = (bx1 + bx2) // 2
                    cy = (by1 + by2) // 2
                    w = roi.x2 - roi.x1
                    h = roi.y2 - roi.y1
                    
                    roi.x1 = cx - w // 2
                    roi.y1 = cy - h // 2
                    roi.x2 = cx + w // 2
                    roi.y2 = cy + h // 2
                    roi.frames_missing = 0
                    roi.active = True
                    break
    
    def draw_rois(self, image_bgr: np.ndarray) -> np.ndarray:
        """Desenha as ROIs na imagem para debug/visualização."""
        output = image_bgr.copy()
        colors = {
            "mulher": (255, 128, 0),   # Laranja
            "homem": (0, 128, 255),    # Azul
            "default": (128, 255, 128), # Verde
        }
        
        for roi in self.rois:
            color = colors.get(roi.name.split("_")[0], colors["default"])
            thickness = 2 if roi.active else 1
            style = cv2.LINE_AA if roi.active else cv2.LINE_8
            
            cv2.rectangle(output, (roi.x1, roi.y1), (roi.x2, roi.y2), color, thickness, style)
            cv2.putText(
                output,
                f"{roi.name} {'ON' if roi.active else 'OFF'}",
                (roi.x1, max(20, roi.y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        
        return output


def create_interview_rois(frame_width: int, frame_height: int) -> list[FaceROI]:
    """
    Cria ROIs padrão para entrevista com dois participantes.
    
    Layout típico:
    - Mulher: lado esquerdo da tela (20% a 55% da largura)
    - Homem: lado direito da tela (45% a 85% da largura)
    """
    rois = []
    
    # ROI da mulher (esquerda)
    rois.append(FaceROI(
        name="mulher_esquerda",
        x1=int(frame_width * 0.05),
        y1=int(frame_height * 0.05),
        x2=int(frame_width * 0.55),
        y2=int(frame_height * 0.85),
    ))
    
    # ROI do homem (direita)
    rois.append(FaceROI(
        name="homem_direita",
        x1=int(frame_width * 0.45),
        y1=int(frame_height * 0.05),
        x2=int(frame_width * 0.95),
        y2=int(frame_height * 0.85),
    ))
    
    return rois