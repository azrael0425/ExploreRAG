"""Resume and verify every Hugging Face repository selected by Docling --all.

Docling's normal CLI is retained for the non-Hugging-Face OCR assets.  This
helper covers the large Hugging Face repositories with bounded HTTP range
requests, which is reliable on networks that drop long Xet/CDN connections.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import time

import requests


ROOT = Path(r"D:\rag001\docling-models")
METADATA_ROOT = Path(r"D:\rag001\ExploreRAG\.docling-model-metadata")
LOG_PATH = ROOT / "download-all-hf.log"
RANGE_BYTES = 32 * 1024 * 1024

# These are exactly the additional Hugging Face repositories selected by
# ``docling-tools models download --all`` in Docling 2.117.0.
REPOSITORIES: dict[str, tuple[str, str]] = {
    "ibm-granite--granite-docling-258M": ("ibm-granite/granite-docling-258M", "982fe3b40f2fa73c365bdb1bcacf6c81b7184bfe"),
    "ibm-granite--granite-docling-258M-mlx": ("ibm-granite/granite-docling-258M-mlx", "e9939db25d2f296c8678d0491c4609a8c596c50a"),
    "docling-project--SmolDocling-256M-preview": ("docling-project/SmolDocling-256M-preview", "ce51f56c4ebe36e0b1c3a55f67b261ba22a50bf8"),
    "docling-project--SmolDocling-256M-preview-mlx-bf16": ("docling-project/SmolDocling-256M-preview-mlx-bf16", "4439622f5b6897153243712a640ddde4e33fb8a8"),
    "ibm-granite--granite-vision-3.3-2b": ("ibm-granite/granite-vision-3.3-2b", "bf66a8244401a8e26d2ff429aa82df8550a289e5"),
    "ibm-granite--granite-vision-3.3-2b-chart2csv-preview": ("ibm-granite/granite-vision-3.3-2b-chart2csv-preview", "298ed4d871264431ec555fa93901bc094579b005"),
    "ibm-granite--granite-vision-4.1-4b": ("ibm-granite/granite-vision-4.1-4b", "37d591f06319e8f1638b5adcf58bdf50e0f84f7a"),
    "nvidia--nemotron-ocr-v2": ("nvidia/nemotron-ocr-v2", "0e83e83f17943524b90afa6c0fd82ac2bc1a40ca"),
}

LFS_POINTER = re.compile(r"^oid sha256:([0-9a-f]{64})\s+size (\d+)$", re.MULTILINE)


def log(message: str) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def manifest(metadata_dir: Path) -> list[tuple[str, int, str | None]]:
    """Build a complete file/hash manifest from a Git-LFS metadata checkout."""
    items: list[tuple[str, int, str | None]] = []
    for source in sorted(metadata_dir.rglob("*")):
        if not source.is_file() or ".git" in source.parts:
            continue
        relative = source.relative_to(metadata_dir).as_posix()
        raw = source.read_bytes()
        pointer = LFS_POINTER.search(raw.decode("utf-8", errors="ignore"))
        if pointer:
            items.append((relative, int(pointer.group(2)), pointer.group(1)))
        else:
            # Config/assets are small and the mirror may serve a newer but
            # compatible text revision. Existing Docling-downloaded versions
            # are kept; only LFS weights require byte-for-byte validation.
            items.append((relative, len(raw), None))
    return items


def is_complete(path: Path, size: int, expected_hash: str | None) -> bool:
    if not path.is_file():
        return False
    if expected_hash is None:
        return path.stat().st_size > 0
    return path.stat().st_size == size and digest(path) == expected_hash


def fetch_file(session: requests.Session, repo: str, revision: str, relative: str, size: int, expected_hash: str | None, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if is_complete(destination, size, expected_hash):
        return

    for attempt in range(1, 1001):
        existing = destination.stat().st_size if destination.exists() else 0
        if existing > size:
            destination.unlink()
            existing = 0
        if expected_hash is not None and existing == size:
            # It has the right length but failed verification: restart that
            # individual weight file instead of treating it as a usable model.
            destination.unlink()
            existing = 0
        try:
            end = min(existing + RANGE_BYTES - 1, size - 1)
            headers = {"Range": f"bytes={existing}-{end}"} if expected_hash is not None else {}
            response = session.get(
                f"https://hf-mirror.com/{repo}/resolve/{revision}/{relative}",
                headers=headers,
                stream=True,
                timeout=(30, 120),
                allow_redirects=True,
            )
            if response.status_code not in (200, 206):
                raise RuntimeError(f"HTTP {response.status_code}")
            if existing and response.status_code == 200:
                raise RuntimeError("server did not honor requested range")
            with destination.open("ab") as handle:
                for block in response.iter_content(chunk_size=4 * 1024 * 1024):
                    if block:
                        handle.write(block)
                        handle.flush()
            response.close()
            if is_complete(destination, size, expected_hash):
                return
        except Exception as exc:  # Preserve partial data and retry from its byte count.
            current = destination.stat().st_size if destination.exists() else 0
            log(f"retry file={relative} attempt={attempt} bytes={current}/{size} error={type(exc).__name__}")
            time.sleep(3)
    raise RuntimeError(f"retry limit exceeded: {repo}/{relative}")


def main() -> None:
    session = requests.Session()
    for folder, (repo, revision) in REPOSITORIES.items():
        metadata_dir = METADATA_ROOT / folder
        if not metadata_dir.is_dir():
            raise FileNotFoundError(f"Missing manifest metadata: {metadata_dir}")
        files = manifest(metadata_dir)
        total = sum(size for _, size, _ in files)
        log(f"start repo={repo} files={len(files)} bytes={total}")
        model_dir = ROOT / folder
        for relative, size, expected_hash in files:
            fetch_file(
                session,
                repo,
                revision,
                relative,
                size,
                expected_hash,
                model_dir / relative,
            )
        log(f"complete repo={repo}")
    log("ALL HUGGINGFACE DOCLING MODELS COMPLETE")


if __name__ == "__main__":
    main()
