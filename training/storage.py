from __future__ import annotations

"""Paths for per-seed artifacts, with fallback for older flat result folders."""

from pathlib import Path


SUBDIRS = {".pt": "checkpoints", ".json": "metrics", ".jsonl": "progress"}


def artifact_path(folder: Path, stem: str, suffix: str, extension: str) -> Path:
    return folder / SUBDIRS[extension] / f"{stem}{suffix}{extension}"


def existing_artifact_path(folder: Path, stem: str, suffix: str,
                           extension: str) -> Path:
    organized = artifact_path(folder, stem, suffix, extension)
    return organized if organized.exists() else folder / f"{stem}{suffix}{extension}"


def prepare_artifact_dirs(folder: Path) -> None:
    for subdir in SUBDIRS.values():
        (folder / subdir).mkdir(parents=True, exist_ok=True)
