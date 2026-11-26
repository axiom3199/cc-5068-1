"""Framing, spectra, spectral peaks and onset (attack) evidence.

This module produces *inspectable* evidence only.  It never picks a final
note.  Every spectral peak keeps its frequency, amplitude, bin and a
prominence measure so that later stages (and human reviewers) can see why a
candidate was generated or rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import AnalysisConfig


def frame_starts(n_samples: int, frame_len: int, hop: int) -> np.ndarray:
    if n_samples < frame_len:
        return np.array([], dtype=np.int64)
    n = 1 + (n_samples - frame_len) // hop
    return np.arange(n, dtype=np.int64) * hop


def get_window(name: str, size: int) -> np.ndarray:
    if name == "hann":
        # periodic Hann (fftbins=True)
        n = np.arange(size)
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / size)
    if name == "hamming":
        n = np.arange(size)
        return 0.54 - 0.46 * np.cos(2.0 * np.pi * n / size)
    if name == "rect":
        return np.ones(size)
    raise ValueError(f"unknown window {name}")


@dataclass
class Peak:
    freq_hz: float
    amp: float
    bin: int
    prominence_db: float

    def to_dict(self) -> dict:
        return {
            "freq_hz": round(self.freq_hz, 4),
            "amp": round(self.amp, 6),
            "bin": int(self.bin),
            "prominence_db": round(self.prominence_db, 2),
        }


@dataclass
class Spectrum:
    mag: np.ndarray          # amplitude spectrum (rfft, length nfft//2+1)
    power: np.ndarray
    freqs: np.ndarray
    rms: float
    rms_db: float
    peak_amp: float


class SpectralFrontEnd:
    """Shared by offline analysis and the streaming path."""

    def __init__(self, cfg: AnalysisConfig):
        self.cfg = cfg
        self.window = get_window(cfg.window, cfg.frame_len)
        self.win_sum = float(np.sum(self.window))
        self.nfft = cfg.nfft
        self.freqs = np.fft.rfftfreq(self.nfft, 1.0 / cfg.sample_rate)
        self.bin_hz = cfg.sample_rate / self.nfft

    def spectrum(self, frame: np.ndarray) -> Spectrum:
        w = frame * self.window
        spec = np.fft.rfft(w, n=self.nfft)
        mag = (2.0 / self.win_sum) * np.abs(spec)
        power = mag * mag
        rms = float(np.sqrt(np.mean(frame * frame))) if frame.size else 0.0
        rms_db = 20.0 * np.log10(max(rms, 1e-12))
        peak_amp = float(np.max(mag)) if mag.size else 0.0
        return Spectrum(mag, power, self.freqs, rms, rms_db, peak_amp)

    def stft(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (magnitude[n_frames, nbins], starts[n_frames])."""
        starts = frame_starts(len(x), self.cfg.frame_len, self.cfg.hop)
        if starts.size == 0:
            return np.zeros((0, self.nfft // 2 + 1)), starts
        frames = np.stack([x[s:s + self.cfg.frame_len] for s in starts])
        w = frames * self.window[None, :]
        spec = np.fft.rfft(w, n=self.nfft, axis=1)
        mag = (2.0 / self.win_sum) * np.abs(spec)
        return mag, starts

    # ------------------------------------------------------------------
    def pick_peaks(self, mag: np.ndarray) -> list[Peak]:
        """Local maxima with parabolic (log-magnitude) refinement.

        The lowest allowed frequency is a little below fmin so that a
        fundamental exactly at fmin is not lost to the bin grid.
        """
        cfg = self.cfg
        nbins = len(mag)
        lo_hz = cfg.fmin * 0.85
        hi_hz = min(cfg.max_harmonic_hz, cfg.sample_rate / 2.0)
        k_lo = max(2, int(np.ceil(lo_hz / self.bin_hz)))
        k_hi = min(nbins - 2, int(np.floor(hi_hz / self.bin_hz)))
        if k_hi <= k_lo:
            return []
        seg = mag[k_lo:k_hi + 1]
        gmax = float(np.max(seg))
        if gmax <= 0:
            return []
        abs_floor = 10.0 ** (cfg.peak_floor_db / 20.0)
        thr = max(abs_floor, cfg.peak_floor_rel * gmax)

        local_max = (seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:])
        idx = np.where(local_max)[0] + 1  # index inside seg
        if idx.size == 0:
            return []

        # keep strongest peaks only
        amps = seg[idx]
        order = np.argsort(amps)[::-1][: cfg.max_peaks]
        idx = idx[order]

        logmag = np.log(np.maximum(mag[k_lo:k_hi + 1], 1e-12))
        peaks: list[Peak] = []
        for i in idx:
            a = mag[k_lo + i]
            if a < thr:
                continue
            # parabolic interpolation on log magnitude
            y0, y1, y2 = logmag[i - 1], logmag[i], logmag[i + 1]
            denom = (y0 - 2.0 * y1 + y2)
            delta = 0.0 if abs(denom) < 1e-12 else 0.5 * (y0 - y2) / denom
            delta = float(np.clip(delta, -0.5, 0.5))
            k = k_lo + i
            freq = (k + delta) * self.bin_hz
            amp = float(a * np.exp(-0.25 * (y0 - y2) * delta))
            prom = self._prominence(mag, k_lo + i)
            peaks.append(Peak(float(freq), amp, int(k), prom))
        peaks.sort(key=lambda p: p.amp, reverse=True)
        return peaks

    def _prominence(self, mag: np.ndarray, k: int, half_width: int = 24) -> float:
        lo = max(0, k - half_width)
        hi = min(len(mag) - 1, k + half_width)
        left = __import__("numpy").min(mag[lo:k + 1]) if k > lo else mag[k]
        right = __import__("numpy").min(mag[k:hi + 1]) if k < hi else mag[k]
        base = max(left, right, 1e-12)
        return float(20.0 * np.log10(max(mag[k], 1e-12) / base))


def spectral_flux(mag: np.ndarray) -> np.ndarray:
    """Half-wave rectified spectral flux, normalised per frame by its norm."""
    if mag.shape[0] == 0:
        return np.zeros(0)
    diff = np.diff(mag, axis=0, prepend=mag[:1])
    pos = np.maximum(diff, 0.0)
    flux = np.sum(pos, axis=1)
    norm = np.sum(mag, axis=1) + 1e-9
    return flux / norm


def detect_attacks(flux: np.ndarray, cfg: AnalysisConfig) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (is_attack, attack_hold, threshold).

    ``attack_hold`` marks the guard region around each onset where pitch
    estimation is allowed to fail (unstable attack transient).
    """
    n = len(flux)
    is_attack = np.zeros(n, dtype=bool)
    hold = np.zeros(n, dtype=bool)
    if n == 0:
        return is_attack, hold, 0.0
    if n < 5:
        return is_attack, hold, float(np.max(flux)) if n else 0.0
    # smooth a little to avoid single-bin flicker
    kern = np.array([0.25, 0.5, 0.25])
    smooth = np.convolve(flux, kern, mode="same")
    med = float(np.median(smooth)) + 1e-9
    pct = float(np.percentile(smooth, cfg.onset_flux_percentile))
    thr = max(pct, cfg.attack_flux_ratio * med)
    min_gap = max(1, int(round(cfg.onset_min_gap_ms / 1000.0 / cfg.hop_seconds)))
    last = -10 ** 9
    for i in range(1, n):
        if smooth[i] >= thr and smooth[i] >= smooth[i - 1] and (i - last) >= min_gap:
            is_attack[i] = True
            last = i
    guard = cfg.attack_guard_frames
    for i in np.where(is_attack)[0]:
        lo = max(0, i - guard)
        hi = min(n, i + guard + 1)
        hold[lo:hi] = True
    return is_attack, hold, float(thr)
