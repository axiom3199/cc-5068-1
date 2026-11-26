"""Minimal WAV I/O on top of numpy (no soundfile dependency).

Supports 16/24/32-bit PCM and 32-bit float, mono or multi-channel, and
resampling with scipy.signal.resample_poly.
"""

from __future__ import annotations

import os
import struct
import wave
import numpy as np


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        nch = w.getnchannels()
        sw = w.getsampwidth()
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)

    if sw == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        data = (data - 128.0) / 128.0
    elif sw == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        val = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        val = np.where(val & 0x800000, val | ~0xFFFFFF, val)
        data = val.astype(np.float64) / 8388608.0
    elif sw == 4:
        # Could be int32 or float32; check WAVE format tag.
        data = _read_float_or_int32(path, raw)
    else:
        raise ValueError(f"unsupported sample width {sw}")

    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)
    return data, sr


def _read_float_or_int32(path: str, raw: bytes) -> np.ndarray:
    with open(path, "rb") as f:
        head = f.read(1024)
    # locate fmt chunk
    idx = head.find(b"fmt ")
    is_float = False
    if idx >= 0:
        fmt_tag = struct.unpack("<H", head[idx + 8: idx + 10])[0]
        is_float = fmt_tag == 3
    if is_float:
        return np.frombuffer(raw, dtype="<f4").astype(np.float64)
    return np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0


def write_wav(path: str, data: np.ndarray, sr: int, bit_depth: int = 16) -> None:
    data = np.asarray(data, dtype=np.float64)
    peak = np.max(np.abs(data)) if data.size else 0.0
    if bit_depth == 16:
        clipped = np.clip(data, -1.0, 1.0)
        ints = np.round(clipped * 32767.0).astype("<i2")
        sw, raw = 2, ints.tobytes()
    elif bit_depth == 32:
        clipped = np.clip(data, -1.0, 1.0)
        ints = np.round(clipped * 2147483647.0).astype("<i4")
        sw, raw = 4, ints.tobytes()
    else:
        raise ValueError("bit_depth must be 16 or 32")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(sw)
        w.setframerate(sr)
        w.writeframes(raw)
    return peak


def resample(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return data.astype(np.float64)
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    return resample_poly(data, up, down)
