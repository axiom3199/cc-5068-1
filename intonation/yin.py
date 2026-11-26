"""YIN periodicity estimate (de Cheveigné & Kawahara, 2002).

YIN provides *period evidence*, independent of the harmonic comb score.  It is
used as a cross-check: a candidate that agrees with YIN is stronger, and YIN's
aperiodicity feeds the voicing decision.  YIN is not trusted blindly either —
for a signal with only even harmonics it reports the physically true period,
which can be an octave above the intended note.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class YinResult:
    f0: float
    aperiodicity: float      # CMNDF value at the chosen lag (0 = periodic)
    period: float            # in samples (fractional)
    threshold_met: bool
    candidates: list         # [(f0, aperiodicity)] local minima, inspectable


def yin_frame(
    frame: np.ndarray,
    sr: int,
    fmin: float,
    fmax: float,
    threshold: float = 0.30,
    max_period: int | None = None,
) -> YinResult:
    x = np.asarray(frame, dtype=np.float64)
    n = len(x)
    if n < 16:
        return YinResult(0.0, 1.0, 0.0, False, [])

    tau_min = max(2, int(np.floor(sr / fmax)))
    if max_period is None:
        max_period = int(np.ceil(sr / fmin))
    tau_max = min(max_period, n // 2)
    if tau_max <= tau_min + 1:
        return YinResult(0.0, 1.0, 0.0, False, [])

    # Integration window W so that x[j] and x[j+tau] exist for j < W, tau <= tau_max.
    W = n - tau_max
    if W < tau_max // 2:
        # not enough overlap; shrink tau_max
        tau_max = n // 2
        W = n - tau_max
        tau_min = min(tau_min, max(2, tau_max - 2))

    head = x[:W]
    seg = x[:W + tau_max]
    # r(tau) = sum_{j=0}^{W-1} x[j] x[j+tau]  (cross-correlation)
    nfft = 1 << int(np.ceil(np.log2(len(seg) + W)))
    corr = np.fft.irfft(
        np.fft.rfft(seg, nfft) * np.conj(np.fft.rfft(head, nfft)), nfft
    )
    # corr[k] = sum_j head[j]*seg[j+k]; we want r(tau)=corr[tau]
    r = corr[0: tau_max + 1]

    energy0 = float(np.sum(head * head))
    cs = np.concatenate(([0.0], np.cumsum(x * x)))
    taus = np.arange(tau_max + 1)
    energy_tau = cs[taus + W] - cs[taus]

    d = energy0 + energy_tau - 2.0 * r
    d = np.maximum(d, 0.0)
    d[0] = 0.0

    # Cumulative mean normalised difference
    cmndf = np.ones_like(d)
    running = 0.0
    for tau in range(1, tau_max + 1):
        running += d[tau]
        cmndf[tau] = d[tau] * tau / running if running > 0 else 1.0

    # candidate local minima in the search range
    lo = max(tau_min, 1)
    hi = tau_max
    cand_list = []
    for tau in range(lo, hi):
        if cmndf[tau] <= cmndf[tau - 1] and cmndf[tau] < cmndf[tau + 1]:
            cand_list.append(tau)
    cand_list.sort(key=lambda t: cmndf[t])

    def interp(tau: int) -> tuple[float, float]:
        if tau <= 0 or tau >= tau_max:
            return float(tau), float(cmndf[tau])
        a, b, c = cmndf[tau - 1], cmndf[tau], cmndf[tau + 1]
        denom = a - 2.0 * b + c
        delta = 0.0 if abs(denom) < 1e-12 else 0.5 * (a - c) / denom
        delta = float(np.clip(delta, -0.5, 0.5))
        val = b - 0.25 * (a - c) * delta
        return tau + delta, float(val)

    # First minimum under the absolute threshold (classic YIN), else global min.
    # Guard against spurious early dips that dip under the threshold: only
    # accept an early minimum if it is a *deep* minimum (<= 0.55*threshold),
    # otherwise use the global minimum which is far more robust for low f0.
    chosen = None
    for tau in range(lo, hi):
        if cmndf[tau] < threshold and cmndf[tau] <= cmndf[tau + 1]:
            if cmndf[tau] <= 0.55 * threshold:
                chosen = tau
            break
    threshold_met = chosen is not None
    if chosen is None:
        if not cand_list:
            return YinResult(0.0, 1.0, 0.0, False, [])
        chosen = cand_list[0]

    period, aper = interp(chosen)
    if period <= 0:
        return YinResult(0.0, 1.0, 0.0, False, [])
    f0 = sr / period
    # Tolerate sub-cent excursions past the nominal range caused by
    # fractional-period interpolation (e.g. 54.996 Hz for fmin = 55 Hz).
    if f0 < fmin * 0.97 or f0 > fmax * 1.03:
        inside = [t for t in cand_list if fmin * 0.97 <= sr / t <= fmax * 1.03]
        if not inside:
            return YinResult(0.0, min(aper, 1.0), period, threshold_met, [])
        chosen = inside[0]
        period, aper = interp(chosen)
        f0 = sr / period

    cands = [(sr / interp(t)[0], interp(t)[1]) for t in cand_list[:6]]
    return YinResult(float(f0), float(aper), float(period), threshold_met, cands)


def yin_frames(frames: np.ndarray, sr: int, cfg) -> list[YinResult]:
    return [
        yin_frame(
            fr,
            sr,
            cfg.fmin,
            cfg.fmax,
            cfg.yin_threshold,
            cfg.yin_max_period,
        )
        for fr in frames
    ]
