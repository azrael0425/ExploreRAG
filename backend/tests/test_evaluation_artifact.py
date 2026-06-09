from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts.verify_evaluation_artifact import verify_artifact


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_artifact(path: Path, *, corrupt_raw: bool = False) -> None:
    raw = json.dumps(
        {
            "experiment_id": "experiment-1",
            "runs": [{"variant": variant} for variant in "ABCD"],
        }
    ).encode()
    report = b"# report\n"
    manifest = json.dumps(
        {
            "artifact_schema_version": 1,
            "experiment_id": "experiment-1",
            "code_snapshot": {"commit": "abc", "dirty": False},
            "files": {
                "raw.json": {"sha256": _digest(raw), "bytes": len(raw)},
                "report.md": {"sha256": _digest(report), "bytes": len(report)},
            },
        }
    ).encode()
    checksums = (
        f"{_digest(raw)}  raw.json\n"
        f"{_digest(report)}  report.md\n"
        f"{_digest(manifest)}  manifest.json\n"
    ).encode()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("raw.json", raw + (b"corrupt" if corrupt_raw else b""))
        archive.writestr("report.md", report)
        archive.writestr("manifest.json", manifest)
        archive.writestr("SHA256SUMS", checksums)


def test_verify_evaluation_artifact_accepts_complete_bundle(tmp_path: Path) -> None:
    artifact = tmp_path / "evaluation.zip"
    _write_artifact(artifact)

    manifest = verify_artifact(artifact)

    assert manifest["experiment_id"] == "experiment-1"


def test_verify_evaluation_artifact_rejects_modified_raw_data(tmp_path: Path) -> None:
    artifact = tmp_path / "evaluation.zip"
    _write_artifact(artifact, corrupt_raw=True)

    with pytest.raises(ValueError, match="Checksum mismatch for raw.json"):
        verify_artifact(artifact)
