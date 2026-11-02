"""Make CUDA runtime wheels discoverable by Windows native DLL loaders."""
import os
from pathlib import Path
import site

_DLL_HANDLES = []


def configure_cuda_dlls():
    if os.name != "nt":
        return
    candidates = []
    for site_dir in site.getsitepackages():
        base = Path(site_dir) / "nvidia"
        for folder in (
            base / "cublas" / "bin",
            base / "cublas" / "lib",
            base / "cudnn" / "bin",
            base / "cudnn" / "lib",
        ):
            if folder.is_dir():
                candidates.append(str(folder))
    if not candidates:
        return
    for folder in candidates:
        try:
            _DLL_HANDLES.append(os.add_dll_directory(folder))
        except (OSError, AttributeError):
            pass
    os.environ["PATH"] = os.pathsep.join(candidates + [os.environ.get("PATH", "")])
