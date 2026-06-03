from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .face_privacy import blur_faces_with_mask, detect_faces, draw_face_overlay, expand_box, iou
from .roi_tracker import ROIFaceTracker, create_interview_rois
from .utils import ensure_dir


@dataclass
class VideoMetadata:
    """Metadados básicos do vídeo de entrada."""

    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


@dataclass
class FramePrivacyResult:
    """Artefatos gerados para um frame processado."""

    frame_index: int
    original_frame: np.ndarray
    anonymized_frame: np.ndarray
    face_overlay: np.ndarray
    face_mask: np.ndarray
    face_boxes: list[list[int]]


@dataclass
class VideoPrivacySummary:
    """Resumo da execução do pipeline de privacidade em vídeo."""

    input_path: Path
    output_path: Path
    metadata: VideoMetadata
    processed_frames: int = 0
    frames_with_faces: int = 0
    total_face_detections: int = 0
    max_faces_in_frame: int = 0
    sample_result: FramePrivacyResult | None = None
    sampled_face_counts: list[int] = field(default_factory=list)
    roi_tracker: ROIFaceTracker | None = None


class TrackedFace:
    """Representa uma face acompanhada temporalmente entre frames."""

    def __init__(self, box: list[int], face_id: int) -> None:
        self.box = box
        self.face_id = face_id
        self.history: list[list[int]] = [box]
        self.missed_frames = 0
        self.velocity = [0, 0, 0, 0]

    def update(self, new_box: list[int] | None) -> None:
        """Atualiza a posição da face com detecção nova ou predição por velocidade."""

        if new_box is not None:
            for coordinate_index in range(4):
                delta = new_box[coordinate_index] - self.box[coordinate_index]
                self.velocity[coordinate_index] = int(0.7 * self.velocity[coordinate_index] + 0.3 * delta)
            self.box = new_box
            self.history.append(new_box)
            if len(self.history) > 5:
                self.history.pop(0)
            self.missed_frames = 0
            return

        self.missed_frames += 1
        self.box = [
            self.box[coordinate_index] + self.velocity[coordinate_index]
            for coordinate_index in range(4)
        ]

    def get_smoothed_box(self, alpha: float = 0.3) -> list[int]:
        """Retorna a média temporal recente da caixa facial."""

        if len(self.history) < 2:
            return self.box

        smoothed_box = []
        for coordinate_index in range(4):
            value = self.history[-1][coordinate_index] * (1 - alpha) + self.history[-2][coordinate_index] * alpha
            if len(self.history) > 2:
                value = value * 0.7 + self.history[-3][coordinate_index] * 0.3
            smoothed_box.append(int(value))
        return smoothed_box


class TemporalFaceTracker:
    """Suaviza caixas faciais e evita falhas visuais entre frames consecutivos."""

    def __init__(
        self,
        hold_frames: int = 90,
        smoothing_alpha: float = 0.6,
        privacy_scale: float = 1.20,
        iou_match_threshold: float = 0.25,
    ) -> None:
        self.hold_frames = hold_frames
        self.smoothing_alpha = smoothing_alpha
        self.privacy_scale = privacy_scale
        self.iou_match_threshold = iou_match_threshold
        self.tracked_faces: list[TrackedFace] = []
        self.next_face_id = 0

    def smooth(self, detected_boxes: list[list[int]], frame_shape: tuple[int, int, int]) -> list[list[int]]:
        """Atualiza rastros temporais e retorna caixas expandidas para privacidade."""

        frame_height, frame_width = frame_shape[:2]
        if detected_boxes:
            self._update_tracks(detected_boxes)
            return self._privacy_boxes(frame_width, frame_height)

        active_boxes = []
        for tracked_face in self.tracked_faces:
            if tracked_face.missed_frames < self.hold_frames:
                tracked_face.update(None)
                active_boxes.append(tracked_face.get_smoothed_box(self.smoothing_alpha))

        if active_boxes:
            return self._privacy_boxes_from_raw(active_boxes, frame_width, frame_height)

        self.tracked_faces = []
        return []

    def _update_tracks(self, detected_boxes: list[list[int]]) -> None:
        matched_detections: set[int] = set()
        matched_tracks: set[int] = set()
        cost_matrix = [
            [1.0 - iou(detected_box, tracked_face.box) for tracked_face in self.tracked_faces]
            for detected_box in detected_boxes
        ]

        if cost_matrix and self.tracked_faces:
            cost_array = np.array(cost_matrix)
            while cost_array.size and cost_array.min() <= (1.0 - self.iou_match_threshold):
                detection_index, track_index = np.unravel_index(cost_array.argmin(), cost_array.shape)
                if detection_index in matched_detections or track_index in matched_tracks:
                    cost_array[detection_index, track_index] = 2.0
                    continue

                self.tracked_faces[track_index].update(detected_boxes[detection_index])
                matched_detections.add(detection_index)
                matched_tracks.add(track_index)
                cost_array[detection_index, :] = 2.0
                cost_array[:, track_index] = 2.0

        for detection_index, detected_box in enumerate(detected_boxes):
            if detection_index not in matched_detections:
                self.tracked_faces.append(TrackedFace(detected_box, self.next_face_id))
                self.next_face_id += 1

        self.tracked_faces = [
            tracked_face
            for tracked_face in self.tracked_faces
            if tracked_face.missed_frames < self.hold_frames or tracked_face.face_id in matched_tracks
        ]

    def _privacy_boxes(self, frame_width: int, frame_height: int) -> list[list[int]]:
        boxes = [tracked_face.get_smoothed_box(self.smoothing_alpha) for tracked_face in self.tracked_faces]
        return self._privacy_boxes_from_raw(boxes, frame_width, frame_height)

    def _privacy_boxes_from_raw(self, boxes: list[list[int]], frame_width: int, frame_height: int) -> list[list[int]]:
        return [expand_box(box, frame_width, frame_height, scale=self.privacy_scale) for box in boxes]


def read_video_metadata(video_path: Path) -> VideoMetadata:
    """Lê resolução, FPS, quantidade de frames e duração do vídeo."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo em {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    duration_seconds = frame_count / fps if fps > 0 else 0.0
    return VideoMetadata(width=width, height=height, fps=fps, frame_count=frame_count, duration_seconds=duration_seconds)


def process_frame(
    frame_bgr: np.ndarray,
    frame_index: int,
    tracker: TemporalFaceTracker | None = None,
    yolo_detector=None,
    roi_tracker=None,
    run_detection: bool = True,
) -> FramePrivacyResult:
    """Processa um frame: detecção facial, máscara e anonimização."""

    face_boxes = detect_faces(frame_bgr, yolo_detector=yolo_detector, roi_tracker=roi_tracker) if run_detection else []
    if tracker is not None:
        face_boxes = tracker.smooth(face_boxes, frame_bgr.shape)

    anonymized_frame, face_mask = blur_faces_with_mask(frame_bgr, face_boxes)
    face_overlay = draw_face_overlay(frame_bgr, face_boxes)
    return FramePrivacyResult(
        frame_index=frame_index,
        original_frame=frame_bgr,
        anonymized_frame=anonymized_frame,
        face_overlay=face_overlay,
        face_mask=face_mask,
        face_boxes=face_boxes,
    )


def process_video(
    input_path: Path,
    output_path: Path,
    yolo_detector=None,
    use_roi: bool = True,
    sample_second: float | None = None,
    max_frames: int | None = None,
    detection_stride: int = 1,
    progress_interval: int = 30,
) -> VideoPrivacySummary:
    """Executa o pipeline completo de privacidade frame a frame."""

    metadata = read_video_metadata(input_path)
    ensure_dir(output_path.parent)
    roi_tracker = ROIFaceTracker(create_interview_rois(metadata.width, metadata.height), roi_expansion=1.3) if use_roi else None
    if roi_tracker is not None:
        print(f"ROI Tracker: {len(roi_tracker.rois)} regiões configuradas")

    capture = cv2.VideoCapture(str(input_path))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        metadata.fps,
        (metadata.width, metadata.height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Não foi possível criar o vídeo de saída em {output_path}")

    target_second = sample_second if sample_second is not None else metadata.duration_seconds / 2
    target_sample_index = int(target_second * metadata.fps)
    target_sample_index = max(0, min(metadata.frame_count - 1, target_sample_index))
    summary = VideoPrivacySummary(input_path=input_path, output_path=output_path, metadata=metadata, roi_tracker=roi_tracker)
    tracker = TemporalFaceTracker()
    frame_index = 0

    while True:
        success, frame = capture.read()
        if not success:
            break
        if max_frames is not None and frame_index >= max_frames:
            break

        active_tracks = len(tracker.tracked_faces)
        run_detection = frame_index % max(1, detection_stride) == 0 or active_tracks > 0
        result = process_frame(
            frame,
            frame_index,
            tracker,
            yolo_detector=yolo_detector,
            roi_tracker=roi_tracker,
            run_detection=run_detection,
        )
        writer.write(result.anonymized_frame)

        face_count = len(result.face_boxes)
        summary.processed_frames += 1
        summary.total_face_detections += face_count
        summary.sampled_face_counts.append(face_count)
        summary.frames_with_faces += int(face_count > 0)
        summary.max_faces_in_frame = max(summary.max_faces_in_frame, face_count)
        if summary.sample_result is None or frame_index == target_sample_index:
            summary.sample_result = result

        if progress_interval > 0 and frame_index % progress_interval == 0:
            print(f"Processando frame {frame_index + 1}/{metadata.frame_count}...")

        frame_index += 1

    capture.release()
    writer.release()
    if summary.sample_result is None:
        raise RuntimeError("Nenhum frame foi processado.")
    return summary
