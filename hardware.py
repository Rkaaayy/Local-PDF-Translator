"""Detect NVIDIA GPU availability and choose the compute device.

Detection happens in two layers:
  1. nvidia-smi  -> is an NVIDIA driver installed and which GPUs exist?
  2. PyTorch     -> can PyTorch actually use CUDA (i.e. is this a CUDA build)?

A GPU is only used when both layers agree. If a driver is present but PyTorch
cannot use CUDA, the report explains how to fix the installation.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field

CUDA_TORCH_INSTALL = (
    "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126"
)


@dataclass
class GPUInfo:
    name: str
    vram_mb: int
    driver: str | None = None


@dataclass
class HardwareReport:
    nvidia_driver_found: bool = False
    torch_installed: bool = False
    torch_version: str | None = None
    torch_cuda_version: str | None = None  # None means a CPU-only PyTorch build
    cuda_available: bool = False
    gpus: list[GPUInfo] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def device(self) -> str:
        return "cuda" if self.cuda_available else "cpu"


def _query_nvidia_smi() -> tuple[bool, list[GPUInfo]]:
    """Return (driver_found, gpus) as reported by nvidia-smi."""
    exe = shutil.which("nvidia-smi")  # handles nvidia-smi.exe on Windows
    if exe is None:
        return False, []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return True, []

    gpus: list[GPUInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            vram = int(float(parts[1]))
        except ValueError:
            vram = 0
        gpus.append(GPUInfo(name=parts[0], vram_mb=vram, driver=parts[2]))
    return True, gpus


def inspect_hardware() -> HardwareReport:
    """Collect everything we need to decide between GPU and CPU."""
    report = HardwareReport()
    report.nvidia_driver_found, smi_gpus = _query_nvidia_smi()

    try:
        import torch
    except ImportError:
        report.notes.append("PyTorch is not installed. See README.md for setup steps.")
        report.gpus = smi_gpus
        return report

    report.torch_installed = True
    report.torch_version = torch.__version__
    report.torch_cuda_version = torch.version.cuda
    report.cuda_available = bool(torch.cuda.is_available())

    if report.cuda_available:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            driver = smi_gpus[i].driver if i < len(smi_gpus) else None
            report.gpus.append(GPUInfo(
                name=props.name,
                vram_mb=int(props.total_memory // (1024 * 1024)),
                driver=driver,
            ))
        return report

    # No usable CUDA: explain why.
    report.gpus = smi_gpus
    if report.nvidia_driver_found and smi_gpus:
        if report.torch_cuda_version is None:
            report.notes.append(
                "An NVIDIA GPU was found, but this PyTorch build is CPU-only.\n"
                f"    Reinstall PyTorch with CUDA support:\n    {CUDA_TORCH_INSTALL}"
            )
        else:
            report.notes.append(
                "An NVIDIA GPU was found, but PyTorch could not initialise CUDA.\n"
                "    Update the NVIDIA driver, or check that the PyTorch CUDA build is "
                "compatible with it."
            )
    elif not report.nvidia_driver_found:
        report.notes.append("No NVIDIA driver detected (nvidia-smi not found). Using the CPU.")
    return report


def resolve_device(requested: str, report: HardwareReport) -> str:
    """Turn the --device option ('auto', 'cuda', 'cpu') into a concrete device."""
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not report.cuda_available:
            raise RuntimeError(
                "--device cuda was requested, but CUDA is not available.\n    "
                + "\n    ".join(report.notes or ["Unknown reason."])
            )
        return "cuda"
    return report.device  # auto


def default_batch_size(device: str, vram_mb: int = 0) -> int:
    """Pick a translation batch size that fits the available memory."""
    if device == "cpu":
        return 4
    if vram_mb >= 12000:
        return 64
    if vram_mb >= 8000:
        return 32
    if vram_mb >= 6000:
        return 16
    return 8


def format_report(report: HardwareReport, device: str | None = None) -> str:
    lines = ["Hardware check"]
    if report.torch_installed:
        cuda_label = report.torch_cuda_version or "none (CPU-only build)"
        lines.append(f"  PyTorch            : {report.torch_version} (CUDA: {cuda_label})")
    else:
        lines.append("  PyTorch            : not installed")
    lines.append(f"  NVIDIA driver      : {'found' if report.nvidia_driver_found else 'not found'}")
    lines.append(f"  CUDA usable        : {'yes' if report.cuda_available else 'no'}")
    for i, gpu in enumerate(report.gpus):
        lines.append(f"  GPU {i}             : {gpu.name}, {gpu.vram_mb} MB VRAM"
                     + (f", driver {gpu.driver}" if gpu.driver else ""))
    for note in report.notes:
        lines.append(f"  Note               : {note}")
    if device is not None:
        label = "GPU (CUDA)" if device == "cuda" else "CPU"
        lines.append(f"  => Processing on   : {label}")
    return "\n".join(lines)
