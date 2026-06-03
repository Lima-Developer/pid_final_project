from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .face_privacy import (
    blur_faces_with_mask,
    detect_faces,
    draw_face_overlay,
    expand_box,
    iou,
)
from .roi_tracker import ROIFaceTracker, create_interview_rois
from .utils import ensure_dir


@dataclass
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


@dataclass
class FramePrivacyResult:
    frame_index: int
    original_frame: np.ndarray
    anonymized_frame: np.ndarray
    face_overlay: np.ndarray
    face_mask: np.ndarray
    face_boxes: list[list[int]]


@dataclass
class VideoPrivacySummary:
    input_path: Path
    output_path: Path
    metadata: VideoMetadata
    processed_frames: int = 0
    frames_with_faces: int = 0
    total_face_detections: int = 0
    max_faces_in_frame: int = 0
    sample_result: FramePrivacyResult | None = None
    sampled_face_counts: list[int] = field(default_factory=list)
    roi_tracker: ROIFaceTracker | None = None  # NOVO


class TrackedFace:
    def __init__(self, box: list[int], face_id: int):
        self.box = box
        self.face_id = face_id
        self.history: list[list[int]] = [box]
        self.missed_frames = 0
        self.velocity = [0, 0, 0, 0]
    
    def update(self, new_box: list[int] | None):
        if new_box is not None:
            for i in range(4):
                self.velocity[i] = int(0.7 * self.velocity[i] + 0.3 * (new_box[i] - self.box[i]))
            self.box = new_box
            self.history.append(new_box)
            if len(self.history) > 5:
                self.history.pop(0)
            self.missed_frames = 0
        else:
            self.missed_frames += 1
            self.box = [
                self.box[0] + self.velocity[0],
                self.box[1] + self.velocity[1],
                self.box[2] + self.velocity[2],
                self.box[3] + self.velocity[3],
            ]
    
    def get_smoothed_box(self, alpha: float = 0.3) -> list[int]:
        if len(self.history) < 2:
            return self.box
        smoothed = []
        for i in range(4):
            val = self.history[-1][i] * (1 - alpha) + self.history[-2][i] * alpha
            if len(self.history) > 2:
                val = val * 0.7 + self.history[-3][i] * 0.3
            smoothed.append(int(val))
        return smoothed


class TemporalFaceTracker:
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
        height, width = frame_shape[:2]
        
        if detected_boxes:
            self._update_tracks(detected_boxes)
            return self._privacy_boxes(width, height)

        active_boxes = []
        for face in self.tracked_faces:
            if face.missed_frames < self.hold_frames:
                face.update(None)
                active_boxes.append(face.get_smoothed_box(self.smoothing_alpha))
        
        if active_boxes:
            return self._privacy_boxes_from_raw(active_boxes, width, height)
        
        self.tracked_faces = []
        return []

    def _update_tracks(self, detected_boxes: list[list[int]]):
        matched_detections: set[int] = set()
        matched_tracks: set[int] = set()
        
        cost_matrix = []
        for det_idx, det in enumerate(detected_boxes):
            row = []
            for track_idx, track in enumerate(self.tracked_faces):
                score = iou(det, track.box)
                row.append(1.0 - score)
            cost_matrix.append(row)
        
        if cost_matrix and self.tracked_faces:
            cost_array = np.array(cost_matrix)
            while True:
                if cost_array.size == 0 or cost_array.min() > (1.0 - self.iou_match_threshold):
                    break
                det_idx, track_idx = np.unravel_index(cost_array.argmin(), cost_array.shape)
                if det_idx in matched_detections or track_idx in matched_tracks:
                    cost_array[det_idx, track_idx] = 2.0
                    continue
                
                self.tracked_faces[track_idx].update(detected_boxes[det_idx])
                matched_detections.add(det_idx)
                matched_tracks.add(track_idx)
                cost_array[det_idx, :] = 2.0
                cost_array[:, track_idx] = 2.0
        
        for det_idx, det in enumerate(detected_boxes):
            if det_idx not in matched_detections:
                new_face = TrackedFace(det, self.next_face_id)
                self.next_face_id += 1
                self.tracked_faces.append(new_face)
        
        self.tracked_faces = [
            f for f in self.tracked_faces 
            if f.missed_frames < self.hold_frames or f.face_id in matched_tracks
        ]

    def _privacy_boxes(self, width: int, height: int) -> list[list[int]]:
        boxes = [f.get_smoothed_box(self.smoothing_alpha) for f in self.tracked_faces]
        return self._privacy_boxes_from_raw(boxes, width, height)

    def _privacy_boxes_from_raw(self, boxes: list[list[int]], width: int, height: int) -> list[list[int]]:
        return [expand_box(box, width, height, scale=self.privacy_scale) for box in boxes]


def read_video_metadata(video_path: Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo em {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    duration_seconds = frame_count / fps if fps > 0 else 0.0
    return VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=duration_seconds,
    )


def process_frame(
    frame_bgr: np.ndarray,
    frame_index: int,
    tracker: TemporalFaceTracker | None = None,
    yolo_detector=None,
    roi_tracker=None,
    run_detection: bool = True,
) -> FramePrivacyResult:
    face_boxes = detect_faces(
        frame_bgr,
        yolo_detector=yolo_detector,
        roi_tracker=roi_tracker,
    ) if run_detection else []
    
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
    metadata = read_video_metadata(input_path)
    ensure_dir(output_path.parent)

    # Inicializa ROI tracker
    roi_tracker = None
    if use_roi:
        roi_tracker = ROIFaceTracker(
            rois=create_interview_rois(metadata.width, metadata.height),
            roi_expansion=1.3,
        )
        print(f"ROI Tracker: {len(roi_tracker.rois)} regiões configuradas")

    capture = cv2.VideoCapture(str(input_path))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        metadata.fps,
        (metadata.width, metadata.height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Não foi possível criar o vídeo de saída em {output_path}")

    target_sample_index = int((sample_second if sample_second is not None else metadata.duration_seconds / 2) * metadata.fps)
    target_sample_index = max(0, min(metadata.frame_count - 1, target_sample_index))
    summary = VideoPrivacySummary(
        input_path=input_path,
        output_path=output_path,
        metadata=metadata,
        roi_tracker=roi_tracker,
    )
    tracker = TemporalFaceTracker()

    frame_index = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        if max_frames is not None and frame_index >= max_frames:
            break

        active_tracks = len(tracker.tracked_faces)
        run_detection = (frame_index % max(1, detection_stride) == 0) or (active_tracks > 0)
        
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
        if face_count > 0:
            summary.frames_with_faces += 1
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
