"""Verify checksums and core provenance fields in an evaluation evidence ZIP."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


REQUIRED_FILES = {"raw.json", "report.md", "manifest.json", "SHA256SUMS"}


def _parse_checksums(value: str) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in value.splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or len(digest) != 64 or not name:
            raise ValueError(f"Invalid SHA256SUMS line: {line!r}")
        checksums[name] = digest
    return checksums


def verify_artifact(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        missing = REQUIRED_FILES - names
        if missing:
            raise ValueError(f"Missing artifact files: {sorted(missing)}")
        checksums = _parse_checksums(archive.read("SHA256SUMS").decode("utf-8"))
        if set(checksums) != REQUIRED_FILES - {"SHA256SUMS"}:
            raise ValueError("SHA256SUMS must cover raw.json, report.md, and manifest.json")
        for name, expected in checksums.items():
            actual = hashlib.sha256(archive.read(name)).hexdigest()
            if actual != expected:
                raise ValueError(f"Checksum mismatch for {name}")

        manifest = json.loads(archive.read("manifest.json"))
        raw = json.loads(archive.read("raw.json"))

    if manifest.get("artifact_schema_version") != 1:
        raise ValueError("Unsupported artifact schema")
    if manifest.get("experiment_id") != raw.get("experiment_id"):
        raise ValueError("Experiment id differs between manifest and raw data")
    variants = {run.get("variant") for run in raw.get("runs", [])}
    if variants != {"A", "B", "C", "D"}:
        raise ValueError(f"Expected A/B/C/D runs, got {sorted(variants)}")
    for name in ("raw.json", "report.md"):
        if (manifest.get("files") or {}).get(name, {}).get("sha256") != checksums[name]:
            raise ValueError(f"Manifest digest differs for {name}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    manifest = verify_artifact(args.artifact)
    snapshot = manifest.get("code_snapshot") or {}
    print(f"verified experiment={manifest['experiment_id']}")
    print(f"commit={snapshot.get('commit', 'unknown')} dirty={snapshot.get('dirty', 'unknown')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
