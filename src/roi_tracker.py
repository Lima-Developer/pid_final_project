from __future__ import annotations

from dataclasses import dataclass

import cv2


@dataclass
class FaceROI:
    """Região de interesse usada para guiar a busca de rostos no vídeo."""

    name: str
    left: int
    top: int
    right: int
    bottom: int
    active: bool = True
    frames_missing: int = 0
    max_frames_missing: int = 90

    def contains(self, box: list[int]) -> bool:
        """Verifica se uma caixa detectada intersecta significativamente a ROI."""

        box_left, box_top, box_right, box_bottom = box
        intersection_left = max(self.left, box_left)
        intersection_top = max(self.top, box_top)
        intersection_right = min(self.right, box_right)
        intersection_bottom = min(self.bottom, box_bottom)
        if intersection_right <= intersection_left or intersection_bottom <= intersection_top:
            return False

        intersection_area = (intersection_right - intersection_left) * (intersection_bottom - intersection_top)
        detected_area = max(1, (box_right - box_left) * (box_bottom - box_top))
        return intersection_area / detected_area > 0.3

    def expand(self, factor: float = 1.2) -> tuple[int, int, int, int]:
        """Expande a ROI para acomodar pequenos movimentos da pessoa."""

        center_x = (self.left + self.right) / 2
        center_y = (self.top + self.bottom) / 2
        width = (self.right - self.left) * factor
        height = (self.bottom - self.top) * factor
        return (
            int(center_x - width / 2),
            int(center_y - height / 2),
            int(center_x + width / 2),
            int(center_y + height / 2),
        )


class ROIFaceTracker:
    """Mantém regiões de interesse previsíveis em vídeos de entrevista."""

    def __init__(self, rois: list[FaceROI] | None = None, roi_expansion: float = 1.3) -> None:
        self.rois = rois or []
        self.roi_expansion = roi_expansion
        self.frame_count = 0

    def detect_faces_in_rois(self, image_bgr, detector_function) -> list[list[int]]:
        """Executa o detector facial apenas dentro das ROIs ativas."""

        detections: list[list[int]] = []
        image_height, image_width = image_bgr.shape[:2]

        for roi in self.rois:
            if not roi.active:
                continue

            roi_left, roi_top, roi_right, roi_bottom = roi.expand(self.roi_expansion)
            roi_left = max(0, roi_left)
            roi_top = max(0, roi_top)
            roi_right = min(image_width, roi_right)
            roi_bottom = min(image_height, roi_bottom)
            if roi_right <= roi_left or roi_bottom <= roi_top:
                continue

            roi_image = image_bgr[roi_top:roi_bottom, roi_left:roi_right]
            if roi_image.size == 0:
                continue

            local_boxes = detector_function(roi_image)
            for local_left, local_top, local_right, local_bottom in local_boxes:
                detections.append(
                    [
                        roi_left + local_left,
                        roi_top + local_top,
                        roi_left + local_right,
                        roi_top + local_bottom,
                    ]
                )

            if local_boxes:
                roi.frames_missing = 0
            else:
                roi.frames_missing += 1
                if roi.frames_missing > roi.max_frames_missing:
                    roi.active = False

        self.frame_count += 1
        return detections

    def update_from_detections(self, detections: list[list[int]]) -> None:
        """Reposiciona uma ROI quando uma detecção confiável cai dentro dela."""

        for roi in self.rois:
            for box in detections:
                if not roi.contains(box):
                    continue

                box_left, box_top, box_right, box_bottom = box
                center_x = (box_left + box_right) // 2
                center_y = (box_top + box_bottom) // 2
                width = roi.right - roi.left
                height = roi.bottom - roi.top
                roi.left = center_x - width // 2
                roi.top = center_y - height // 2
                roi.right = center_x + width // 2
                roi.bottom = center_y + height // 2
                roi.frames_missing = 0
                roi.active = True
                break

    def draw_rois(self, image_bgr):
        """Desenha as ROIs ativas para inspeção visual."""

        output = image_bgr.copy()
        colors = {
            "mulher": (255, 128, 0),
            "homem": (0, 128, 255),
            "default": (128, 255, 128),
        }
        for roi in self.rois:
            color = colors.get(roi.name.split("_")[0], colors["default"])
            thickness = 2 if roi.active else 1
            line_type = cv2.LINE_AA if roi.active else cv2.LINE_8
            cv2.rectangle(output, (roi.left, roi.top), (roi.right, roi.bottom), color, thickness, line_type)
            cv2.putText(
                output,
                f"{roi.name} {'ON' if roi.active else 'OFF'}",
                (roi.left, max(20, roi.top - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return output


def create_interview_rois(frame_width: int, frame_height: int) -> list[FaceROI]:
    """Cria ROIs iniciais para os dois participantes do vídeo de entrevista."""

    return [
        FaceROI(
            name="mulher_esquerda",
            left=int(frame_width * 0.05),
            top=int(frame_height * 0.05),
            right=int(frame_width * 0.55),
            bottom=int(frame_height * 0.85),
        ),
        FaceROI(
            name="homem_direita",
            left=int(frame_width * 0.45),
            top=int(frame_height * 0.05),
            right=int(frame_width * 0.95),
            bottom=int(frame_height * 0.85),
        ),
    ]
