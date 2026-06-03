from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .utils import ensure_dir


def to_bgr(image: np.ndarray) -> np.ndarray:
    """Garante que uma imagem em escala de cinza seja convertida para BGR."""

    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def add_title(image_bgr: np.ndarray, title: str) -> np.ndarray:
    """Adiciona uma faixa superior com título a uma imagem."""

    canvas = image_bgr.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (20, 20, 20), -1)
    cv2.putText(
        canvas,
        title,
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def resize_to(image_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Redimensiona a imagem para o tamanho informado."""

    return cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)


def build_panel(
    images_with_titles: list[tuple[str, np.ndarray]],
    output_path: Path,
    cell_width: int = 480,
    cell_height: int = 270,
    columns: int = 3,
) -> None:
    """Cria um painel visual em grade a partir de imagens tituladas."""

    if not images_with_titles:
        raise ValueError("O painel precisa receber pelo menos uma imagem.")

    cells = [
        add_title(resize_to(to_bgr(image), cell_width, cell_height), title)
        for title, image in images_with_titles
    ]
    rows = []
    for start_index in range(0, len(cells), columns):
        row_cells = cells[start_index:start_index + columns]
        rows.append(np.hstack(row_cells))

    ensure_dir(output_path.parent)
    cv2.imwrite(str(output_path), np.vstack(rows))
