from __future__ import annotations

import json
import ssl
import urllib.request
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """Cria um diretório quando ele ainda não existe."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url: str, destination: Path) -> Path:
    """Baixa um arquivo para o caminho informado."""

    ensure_dir(destination.parent)
    context = ssl.create_default_context()
    with urllib.request.urlopen(url, context=context) as response:
        destination.write_bytes(response.read())
    return destination


def save_json(data: dict, path: Path) -> None:
    """Salva um dicionário em JSON UTF-8 indentado."""

    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    """Salva texto em UTF-8."""

    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")
