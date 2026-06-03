from __future__ import annotations

import json
import ssl
import urllib.request
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url: str, destination: Path) -> Path:
    ensure_dir(destination.parent)
    context = ssl.create_default_context()
    with urllib.request.urlopen(url, context=context) as response:
        destination.write_bytes(response.read())
    return destination


def save_json(data: dict, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))
