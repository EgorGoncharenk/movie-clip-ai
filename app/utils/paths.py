from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path

_UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_name(value: str) -> str:
    cleaned = _UNSAFE_FILENAME_CHARS.sub("_", value).strip(" .")
    return cleaned or "video"


def source_cache_key(source: Path) -> str:
    resolved = source.resolve()
    stat = resolved.stat()
    signature = f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]


def cache_directory(source: Path, temp_root: Path) -> Path:
    directory = temp_root / f"{safe_name(source.stem)}_{source_cache_key(source)}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def create_run_directory(source: Path, output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_root / f"{safe_name(source.stem)}_{timestamp}"
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = Path(f"{base}_{suffix}")
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate
