"""Download the PyTorch variant of BAAI/bge-m3 with resume and verification.

The application uses sentence-transformers/PyTorch, so the optional ONNX
runtime export is intentionally excluded.  This saves about 2.1 GiB without
changing the embedding model used by ExploreRAG.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import time

import requests


ROOT = Path(r"D:\rag001\bge-m3")
LOG_PATH = ROOT / "download.log"
BASE_URL = "https://hf-mirror.com/BAAI/bge-m3/resolve/main"

# Files required by SentenceTransformer("/models/bge-m3").  The SHA-256
# values for LFS files are taken from the repository's Git-LFS pointers.
FILES: dict[str, tuple[int | None, str | None]] = {
    "1_Pooling/config.json": (None, None),
    "colbert_linear.pt": (2_100_674, "19bfbae397c2b7524158c919d0e9b19393c5639d098f0a66932c91ed8f5f9abb"),
    "config.json": (None, None),
    "config_sentence_transformers.json": (None, None),
    "modules.json": (None, None),
    "pytorch_model.bin": (2_271_145_830, "b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38"),
    "sentence_bert_config.json": (None, None),
    "sentencepiece.bpe.model": (5_069_051, "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865"),
    "sparse_linear.pt": (3_516, "45c93804d2142b8f6d7ec6914ae23a1eee9c6a1d27d83d908a20d2afb3595ad9"),
    "special_tokens_map.json": (None, None),
    "tokenizer.json": (17_098_108, "21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08"),
    "tokenizer_config.json": (None, None),
}


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


def complete(path: Path, expected_size: int | None, expected_hash: str | None) -> bool:
    if not path.is_file():
        return False
    if expected_size is not None and path.stat().st_size != expected_size:
        return False
    return expected_hash is None or sha256(path) == expected_hash


def download(relative_path: str, expected_size: int | None, expected_hash: str | None) -> None:
    destination = ROOT / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if complete(destination, expected_size, expected_hash):
        log(f"verified {relative_path}")
        return

    session = requests.Session()
    for attempt in range(1, 501):
        existing = destination.stat().st_size if destination.exists() else 0
        if expected_size is not None and existing > expected_size:
            destination.unlink()
            existing = 0
        try:
            # Keep each CDN response bounded.  The network here can close a
            # very long response even when byte ranges themselves work well.
            if expected_size is not None:
                end = min(existing + 32 * 1024 * 1024 - 1, expected_size - 1)
                headers = {"Range": f"bytes={existing}-{end}"}
            else:
                headers = {"Range": f"bytes={existing}-"} if existing else {}
            response = session.get(
                f"{BASE_URL}/{relative_path}",
                headers=headers,
                stream=True,
                timeout=(30, 120),
                allow_redirects=True,
            )
            if response.status_code not in (200, 206):
                raise RuntimeError(f"HTTP {response.status_code}")
            if existing and response.status_code == 200:
                raise RuntimeError("server did not honor requested byte range")
            with destination.open("ab") as handle:
                for block in response.iter_content(chunk_size=4 * 1024 * 1024):
                    if block:
                        handle.write(block)
                        handle.flush()
            response.close()
            if complete(destination, expected_size, expected_hash):
                log(f"verified {relative_path}")
                return
            if expected_size is not None:
                # The next loop iteration requests the following 32 MiB range.
                continue
            if expected_size is None:
                # Small non-LFS configuration files have no pointer checksum.
                log(f"downloaded {relative_path}")
                return
        except Exception as exc:  # Retry without discarding validated partial data.
            current = destination.stat().st_size if destination.exists() else 0
            log(f"retry {relative_path} attempt={attempt} bytes={current} error={type(exc).__name__}")
            time.sleep(3)
    raise RuntimeError(f"retry limit exceeded for {relative_path}")


if __name__ == "__main__":
    try:
        for filename, (size, digest) in FILES.items():
            download(filename, size, digest)
        log("BGE-M3 PYTORCH DOWNLOAD COMPLETE")
    except Exception as exc:  # Keep a concise, secret-free record for monitoring.
        log(f"FATAL: {type(exc).__name__}: {exc}")
        raise
