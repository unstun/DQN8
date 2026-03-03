"""PyTorch / CUDA / Matplotlib runtime setup.

Provides
--------
- configure_runtime()    Force non-interactive Matplotlib backend; OpenMP workaround on Windows.
- torch_runtime_info()   Detect CUDA availability, device count, device names.
- require_cuda()         Return CUDA device or raise with diagnostic message.
- select_device()        "auto" / "cuda" / "cpu" device selection.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass


def configure_runtime(*, matplotlib_backend: str = "Agg") -> None:
    """Best-effort runtime hardening for reproducible CLI runs.

    - Forces a non-interactive Matplotlib backend to avoid Qt thread shutdown
      issues when only saving figures.
    - Works around a common Windows crash when multiple OpenMP runtimes are
      loaded (e.g., torch + numpy/pandas/opencv).
    """

    os.environ.setdefault("MPLBACKEND", str(matplotlib_backend))

    if platform.system() == "Windows":
        # Workaround for: "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


@dataclass(frozen=True)
class TorchRuntimeInfo:
    torch_version: str
    cuda_available: bool
    torch_cuda_version: str | None
    device_count: int
    device_names: tuple[str, ...]


def torch_runtime_info() -> TorchRuntimeInfo:
    import torch

    device_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    device_names: tuple[str, ...] = tuple(
        torch.cuda.get_device_name(i) for i in range(device_count)
    )
    return TorchRuntimeInfo(
        torch_version=str(torch.__version__),
        cuda_available=bool(torch.cuda.is_available()),
        torch_cuda_version=None if torch.version.cuda is None else str(torch.version.cuda),
        device_count=device_count,
        device_names=device_names,
    )


def require_cuda(*, device_index: int = 0) -> "torch.device":
    """Return a CUDA torch.device, or raise with a helpful message."""
    import torch

    info = torch_runtime_info()
    if not info.cuda_available:
        raise RuntimeError(
            "CUDA is required but was not detected.\n"
            f"- Detected torch: {info.torch_version}\n"
            f"- torch.version.cuda: {info.torch_cuda_version}\n"
            "Install a CUDA-enabled PyTorch build, then re-run.\n"
            "- PyTorch install selector: https://pytorch.org/get-started/locally/"
        )

    idx = int(device_index)
    if idx < 0 or idx >= info.device_count:
        raise RuntimeError(
            f"Invalid CUDA device index {idx}. Available device_count={info.device_count}."
        )

    return torch.device(f"cuda:{idx}")


def select_device(*, device: str = "auto", cuda_device: int = 0) -> "torch.device":
    """Select a torch.device.

    device:
      - "auto": CUDA if available, else CPU
      - "cuda": require CUDA (raises if unavailable)
      - "cpu": force CPU
    """
    import torch

    choice = str(device).lower().strip()
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return require_cuda(device_index=int(cuda_device))
    if choice != "auto":
        raise ValueError(f"Unknown device {device!r}. Expected: auto|cpu|cuda")

    try:
        return require_cuda(device_index=int(cuda_device))
    except RuntimeError:
        return torch.device("cpu")
