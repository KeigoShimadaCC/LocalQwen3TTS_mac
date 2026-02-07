from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import torch


LOGGER = logging.getLogger(__name__)


DeviceType = Literal["cpu", "mps"]
DTypePref = Literal["auto", "float16", "float32"]


@dataclass
class DevicePlan:
    device: torch.device
    dtype: torch.dtype
    reason: str


def plan_device(
    device_pref: DeviceType | Literal["auto"],
    dtype_pref: DTypePref,
) -> DevicePlan:
    # Resolve device
    if device_pref == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        reason = "user forced mps"
    elif device_pref == "cpu":
        device = torch.device("cpu")
        reason = "user forced cpu"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        reason = "auto detected mps"
    else:
        device = torch.device("cpu")
        reason = "fallback to cpu"

    # Resolve dtype
    if dtype_pref == "float16":
        dtype = torch.float16
    elif dtype_pref == "float32":
        dtype = torch.float32
    else:
        dtype = torch.float16 if device.type == "mps" else torch.float32

    LOGGER.info(
        "Device plan resolved device=%s dtype=%s reason=%s", device, dtype, reason
    )
    return DevicePlan(device=device, dtype=dtype, reason=reason)
