"""Resilient, resumable download for the complete Docling model set.

The helper first downloads the TableFormer-fast file with HTTP Range requests,
then delegates the rest of the official model set to ``docling-tools``. It is
intended for networks that intermittently close long Hugging Face downloads.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import requests


ROOT = Path(r"D:\rag001\docling-models")
LOG_PATH = ROOT / "download-full.log"
FAST_MODEL = (
    ROOT
    / "docling-project--docling-models"
    / "model_artifacts"
    / "tableformer"
    / "fast"
    / "tableformer_fast.safetensors"
)
# The official Hugging Face endpoint currently drops TLS streams on this
# machine.  This public mirror returns the same immutable revision; the SHA-256
# verification below makes the source route immaterial to file integrity.
FAST_URL = (
    "https://hf-mirror.com/docling-project/docling-models/resolve/v2.3.0/"
    "model_artifacts/tableformer/fast/tableformer_fast.safetensors?download=true"
)
FAST_SIZE = 145_453_276
FAST_SHA256 = "3119563aab5a7c96fda4d621119b63fd8806272b86c30936d15507616422f718"
DOCLING_TOOLS = Path(r"C:\Users\a1397\anaconda3\Scripts\docling-tools.exe")


def log(message: str) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_fast_tableformer() -> None:
    """Download TableFormer-fast safely despite interrupted HTTP streams."""
    FAST_MODEL.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    for attempt in range(1, 501):
        existing = FAST_MODEL.stat().st_size if FAST_MODEL.exists() else 0
        if existing == FAST_SIZE:
            break
        if existing > FAST_SIZE:
            FAST_MODEL.unlink()
            existing = 0

        try:
            headers = {"Range": f"bytes={existing}-"} if existing else {}
            response = session.get(
                FAST_URL,
                headers=headers,
                stream=True,
                timeout=(30, 90),
                allow_redirects=True,
            )
            if response.status_code not in (200, 206):
                raise RuntimeError(f"HTTP {response.status_code}")
            if existing and response.status_code == 200:
                raise RuntimeError("server did not honor the requested byte range")

            with FAST_MODEL.open("ab") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        handle.flush()
            response.close()
            log(
                f"fast-tableformer attempt={attempt} "
                f"bytes={FAST_MODEL.stat().st_size}/{FAST_SIZE}"
            )
        except Exception as exc:  # noqa: BLE001 - resume all network errors
            current = FAST_MODEL.stat().st_size if FAST_MODEL.exists() else 0
            log(
                f"fast-tableformer retry={attempt} bytes={current}/{FAST_SIZE} "
                f"error={type(exc).__name__}: {exc}"
            )
            time.sleep(3)
    else:
        raise RuntimeError("TableFormer-fast exceeded its retry limit")

    if FAST_MODEL.stat().st_size != FAST_SIZE:
        raise RuntimeError("TableFormer-fast size verification failed")
    if sha256(FAST_MODEL) != FAST_SHA256:
        raise RuntimeError("TableFormer-fast checksum verification failed")
    log("fast-tableformer verified")


def download_remaining_models() -> None:
    """Retry the official full Docling downloader until it succeeds."""
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_DISABLE_XET": "1",
            "HF_HUB_DOWNLOAD_TIMEOUT": "600",
            "HF_HUB_ETAG_TIMEOUT": "60",
        }
    )
    command = [
        str(DOCLING_TOOLS),
        "models",
        "download",
        "--all",
        "-o",
        str(ROOT),
        "--easyocr-lang",
        "ch_sim",
        "--easyocr-lang",
        "en",
        "--quiet",
    ]

    for attempt in range(1, 201):
        log(f"full-docling attempt={attempt}")
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=environment,
            check=False,
        )
        if result.returncode == 0:
            log("FULL DOCLING MODEL DOWNLOAD COMPLETE")
            return
        log(f"full-docling retry after exit={result.returncode}")
        time.sleep(20)

    raise RuntimeError("Docling model downloader exceeded its retry limit")


if __name__ == "__main__":
    try:
        download_fast_tableformer()
        download_remaining_models()
    except Exception as exc:  # noqa: BLE001 - process exit is logged for monitoring
        log(f"FATAL: {type(exc).__name__}: {exc}")
        sys.exit(1)
