import subprocess

from app.core.config import settings
from app.evaluation import fingerprints


def test_git_snapshot_falls_back_to_container_build_metadata(monkeypatch) -> None:
    def unavailable(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(fingerprints.subprocess, "check_output", unavailable)
    monkeypatch.setattr(settings, "EXPLORERAG_BUILD_GIT_COMMIT", "abc123")
    monkeypatch.setattr(settings, "EXPLORERAG_BUILD_GIT_BRANCH", "main")
    monkeypatch.setattr(settings, "EXPLORERAG_BUILD_GIT_DIRTY", "false")

    assert fingerprints._git_snapshot() == {
        "commit": "abc123",
        "branch": "main",
        "dirty": False,
        "source": "build_metadata",
    }
