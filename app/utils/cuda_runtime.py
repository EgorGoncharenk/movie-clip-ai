from __future__ import annotations

import os
import sys
from pathlib import Path

_DLL_HANDLES: list[object] = []


def configure_cuda_dlls() -> list[Path]:
    """Expose NVIDIA wheel DLLs to CTranslate2 on Windows."""
    if os.name != "nt":
        return []

    nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    directories = [
        nvidia_root / component / "bin"
        for component in ("cuda_runtime", "cuda_nvrtc", "cublas", "cudnn")
        if (nvidia_root / component / "bin").is_dir()
    ]
    if not directories:
        return []

    current_path = os.environ.get("PATH", "")
    existing = current_path.split(os.pathsep) if current_path else []
    additions = [str(path) for path in directories if str(path) not in existing]
    if additions:
        os.environ["PATH"] = os.pathsep.join(additions + existing)

    if hasattr(os, "add_dll_directory"):
        registered = {
            str(getattr(handle, "path", ""))
            for handle in _DLL_HANDLES
        }
        for directory in directories:
            if str(directory) not in registered:
                _DLL_HANDLES.append(os.add_dll_directory(str(directory)))
    return directories


def cuda_runtime_files() -> dict[str, bool]:
    directories = configure_cuda_dlls()
    files = {
        "cuBLAS 12": "cublas64_12.dll",
        "cuDNN 9": "cudnn64_9.dll",
    }
    return {
        label: any((directory / filename).is_file() for directory in directories)
        for label, filename in files.items()
    }
