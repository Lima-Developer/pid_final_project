from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .utils import ensure_dir


def to_bgr(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def add_title(image_bgr: np.ndarray, title: str) -> np.ndarray:
    canvas = image_bgr.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 40), (20, 20, 20), -1)
    cv2.putText(
        canvas,
        title,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def resize_to(image_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image_bgr, (width, height))


def build_panel(images_with_titles: list[tuple[str, np.ndarray]], output_path: Path, cell_width: int = 420, cell_height: int = 280) -> None:
    cells: list[np.ndarray] = []
    for title, image in images_with_titles:
        framed = add_title(resize_to(to_bgr(image), cell_width, cell_height), title)
        cells.append(framed)

    rows: list[np.ndarray] = []
    for index in range(0, len(cells), 3):
        row_cells = cells[index:index + 3]
        while len(row_cells) < 3:
            row_cells.append(np.zeros_like(cells[0]))
        rows.append(np.hstack(row_cells))

    panel = np.vstack(rows)
    ensure_dir(output_path.parent)
    cv2.imwrite(str(output_path), panel)


def draw_ground_truth(image_bgr: np.ndarray, boxes: list[list[int]]) -> np.ndarray:
    output = image_bgr.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 100, 0), 2)
    return output


def build_metrics_dashboard(summary_metrics: dict[str, dict[str, float]], output_path: Path) -> None:
    canvas = np.full((820, 1400, 3), 245, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (1400, 110), (22, 28, 45), -1)
    cv2.putText(canvas, "Face Privacy Blur - Detector Metrics", (45, 68), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, "Candidate detection supports the target-search case", (47, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (200, 215, 255), 2, cv2.LINE_AA)

    metric_names = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1-score"),
        ("average_iou", "Avg IoU"),
    ]

    colors = {
        "hog": (86, 94, 219),
        "yolo": (0, 180, 255),
    }

    left_x = 240
    right_x = 1320
    label_x = 60
    top_y = 190
    group_height = 135
    bar_height = 26
    max_bar_width = 920

    cv2.rectangle(canvas, (1000, 38), (1030, 68), colors["hog"], -1)
    cv2.putText(canvas, "HOG", (1045, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (1120, 38), (1150, 68), colors["yolo"], -1)
    cv2.putText(canvas, "YOLO", (1165, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)

    for index, (metric_key, metric_label) in enumerate(metric_names):
        base_y = top_y + index * group_height
        cv2.putText(canvas, metric_label, (label_x, base_y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (35, 35, 35), 2, cv2.LINE_AA)
        cv2.line(canvas, (left_x, base_y + 85), (right_x, base_y + 85), (210, 210, 210), 2)

        for line_index, method in enumerate(["hog", "yolo"]):
            value = float(summary_metrics.get(method, {}).get(metric_key, 0.0))
            y = base_y + 12 + line_index * 42
            cv2.rectangle(canvas, (left_x, y), (left_x + max_bar_width, y + bar_height), (230, 230, 230), -1)
            bar_width = int(max_bar_width * max(0.0, min(1.0, value)))
            cv2.rectangle(canvas, (left_x, y), (left_x + bar_width, y + bar_height), colors[method], -1)
            cv2.putText(canvas, f"{value:.3f}", (left_x + 900, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (45, 45, 45), 2, cv2.LINE_AA)

    footer_y = 760
    cv2.rectangle(canvas, (45, footer_y - 36), (1355, footer_y + 18), (255, 255, 255), -1)
    cv2.putText(
        canvas,
        "Takeaway: detection is only the first step; the main case ranks candidates by visual similarity to the selected target.",
        (58, footer_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )

    ensure_dir(output_path.parent)
    cv2.imwrite(str(output_path), canvas)
