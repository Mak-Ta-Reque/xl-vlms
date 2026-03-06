"""Unified device configuration for xl-vlms.

Parses the DEVICE environment variable (or --device CLI arg) into a
HuggingFace-compatible ``device_map`` / ``max_memory`` dict that can be
passed directly to ``AutoModel.from_pretrained()``.

Supported formats
-----------------
  "cuda:1"       – single GPU (index 1), CPU overflow
  "cuda:0,2"     – GPUs 0 & 2, CPU overflow
  "cuda:[0,2]"   – same as above (bracket syntax)
  "auto"         – all available GPUs, CPU overflow
  "cpu"          – CPU only
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch

__all__ = ["DeviceConfig", "parse_device_config", "resolve_device_env"]

logger = logging.getLogger(__name__)

# Default CPU memory budget for overflow offloading (bytes).
_DEFAULT_CPU_MEMORY = "48GiB"
# Fraction of each GPU's total memory we allow the model to use.
_GPU_MEM_FRACTION = 0.90


# ---------------------------------------------------------------------------
# DeviceConfig dataclass
# ---------------------------------------------------------------------------
@dataclass
class DeviceConfig:
    """Holds everything ``from_pretrained`` needs for device placement.

    Attributes
    ----------
    device_map : str | dict | None
        Passed to ``from_pretrained(device_map=...)``.
        ``"auto"`` lets accelerate decide; ``"cpu"`` forces CPU.
    max_memory : dict | None
        Passed to ``from_pretrained(max_memory=...)``.
        Maps ``"cuda:N"`` / ``"cpu"`` to memory budget strings.
    primary_device : torch.device
        A *single* device used for non-model tensors (input batches,
        concept vectors, etc.).  Typically the first GPU in the list.
    gpu_ids : list[int]
        Sorted list of GPU indices that may be used (empty for CPU-only).
    raw : str
        Original string that was parsed.
    """

    device_map: Union[str, Dict[str, Any], None] = "auto"
    max_memory: Optional[Dict[str, str]] = None
    primary_device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    gpu_ids: List[int] = field(default_factory=list)
    raw: str = ""

    # Convenience -------------------------------------------------------
    @property
    def is_cpu_only(self) -> bool:
        return len(self.gpu_ids) == 0

    @property
    def is_single_gpu(self) -> bool:
        return len(self.gpu_ids) == 1

    @property
    def is_multi_gpu(self) -> bool:
        return len(self.gpu_ids) > 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gpu_memory_str(gpu_id: int) -> str:
    """Return a memory budget string for *gpu_id* (e.g. ``'22GiB'``)."""
    if not torch.cuda.is_available():
        return "0GiB"
    try:
        total = torch.cuda.get_device_properties(gpu_id).total_mem
        budget = int(total * _GPU_MEM_FRACTION)
        gib = budget / (1024 ** 3)
        return f"{gib:.0f}GiB"
    except Exception:
        return "24GiB"  # safe fallback


def _validate_gpu_ids(gpu_ids: List[int]) -> List[int]:
    """Filter *gpu_ids* to those that actually exist."""
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    valid = sorted(set(gid for gid in gpu_ids if 0 <= gid < n))
    if len(valid) < len(gpu_ids):
        missing = sorted(set(gpu_ids) - set(valid))
        logger.warning(
            "Requested GPU(s) %s not available (device_count=%d). Using %s.",
            missing, n, valid or "CPU",
        )
    return valid


def _build_max_memory(gpu_ids: List[int], cpu_mem: str = _DEFAULT_CPU_MEMORY) -> Dict[str, str]:
    mem: Dict[str, str] = {}
    for gid in gpu_ids:
        mem[gid] = _gpu_memory_str(gid)
    mem["cpu"] = cpu_mem
    return mem


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

_BRACKET_RE = re.compile(r"^cuda:\s*\[([^\]]+)\]$", re.IGNORECASE)
_COMMA_RE = re.compile(r"^cuda:\s*(\d[\d,\s]*)$", re.IGNORECASE)
_SINGLE_RE = re.compile(r"^cuda(?::(\d+))?$", re.IGNORECASE)


def parse_device_config(device_str: Optional[str] = None) -> DeviceConfig:
    """Parse a device specification string into a :class:`DeviceConfig`.

    Parameters
    ----------
    device_str : str or None
        One of the supported formats listed in the module docstring.
        ``None`` is treated as ``"auto"``.

    Returns
    -------
    DeviceConfig
    """
    if device_str is None:
        device_str = "auto"
    raw = device_str.strip()
    low = raw.lower()

    # --- CPU only ---
    if low == "cpu":
        return DeviceConfig(
            device_map="cpu",
            max_memory=None,
            primary_device=torch.device("cpu"),
            gpu_ids=[],
            raw=raw,
        )

    # --- auto ---
    if low == "auto":
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if n == 0:
            logger.info("DEVICE='auto' but no GPUs found – falling back to CPU.")
            return DeviceConfig(
                device_map="cpu",
                max_memory=None,
                primary_device=torch.device("cpu"),
                gpu_ids=[],
                raw=raw,
            )
        gpu_ids = list(range(n))
        return DeviceConfig(
            device_map="auto",
            max_memory=_build_max_memory(gpu_ids),
            primary_device=torch.device(f"cuda:{gpu_ids[0]}"),
            gpu_ids=gpu_ids,
            raw=raw,
        )

    # --- cuda:[0,2] bracket syntax ---
    m = _BRACKET_RE.match(raw)
    if m:
        ids = [int(x.strip()) for x in m.group(1).split(",") if x.strip().isdigit()]
        ids = _validate_gpu_ids(ids)
        if not ids:
            logger.warning("No valid GPUs in '%s' – falling back to CPU.", raw)
            return DeviceConfig(device_map="cpu", primary_device=torch.device("cpu"), gpu_ids=[], raw=raw)
        return DeviceConfig(
            device_map="auto",
            max_memory=_build_max_memory(ids),
            primary_device=torch.device(f"cuda:{ids[0]}"),
            gpu_ids=ids,
            raw=raw,
        )

    # --- cuda:0,2 comma syntax (no brackets) ---
    m = _COMMA_RE.match(raw)
    if m:
        ids = [int(x.strip()) for x in m.group(1).split(",") if x.strip().isdigit()]
        ids = _validate_gpu_ids(ids)
        if not ids:
            logger.warning("No valid GPUs in '%s' – falling back to CPU.", raw)
            return DeviceConfig(device_map="cpu", primary_device=torch.device("cpu"), gpu_ids=[], raw=raw)
        if len(ids) == 1:
            # Single-GPU shorthand: cuda:1
            return DeviceConfig(
                device_map="auto",
                max_memory=_build_max_memory(ids),
                primary_device=torch.device(f"cuda:{ids[0]}"),
                gpu_ids=ids,
                raw=raw,
            )
        return DeviceConfig(
            device_map="auto",
            max_memory=_build_max_memory(ids),
            primary_device=torch.device(f"cuda:{ids[0]}"),
            gpu_ids=ids,
            raw=raw,
        )

    # --- cuda  or  cuda:N ---
    m = _SINGLE_RE.match(raw)
    if m:
        idx = int(m.group(1)) if m.group(1) is not None else 0
        ids = _validate_gpu_ids([idx])
        if not ids:
            logger.warning("GPU %d not available – falling back to CPU.", idx)
            return DeviceConfig(device_map="cpu", primary_device=torch.device("cpu"), gpu_ids=[], raw=raw)
        return DeviceConfig(
            device_map="auto",
            max_memory=_build_max_memory(ids),
            primary_device=torch.device(f"cuda:{ids[0]}"),
            gpu_ids=ids,
            raw=raw,
        )

    # --- Unrecognised – try torch.device as-is ---
    logger.warning("Unrecognised DEVICE format '%s' – attempting torch.device().", raw)
    try:
        dev = torch.device(raw)
        return DeviceConfig(
            device_map=None,
            max_memory=None,
            primary_device=dev,
            gpu_ids=[dev.index] if dev.type == "cuda" and dev.index is not None else [],
            raw=raw,
        )
    except Exception:
        logger.error("Could not parse DEVICE='%s'. Falling back to CPU.", raw)
        return DeviceConfig(device_map="cpu", primary_device=torch.device("cpu"), gpu_ids=[], raw=raw)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def resolve_device_env() -> str:
    """Read the ``DEVICE`` env var, defaulting to ``'auto'``."""
    return os.environ.get("DEVICE", "auto")


def get_device_config(device_override: Optional[str] = None) -> DeviceConfig:
    """Parse ``device_override`` if given, else read the ``DEVICE`` env var."""
    return parse_device_config(device_override or resolve_device_env())
