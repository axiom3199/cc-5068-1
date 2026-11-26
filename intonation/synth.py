"""Deterministic synthetic recordings and fixtures for evaluating the analyzer.

Design constraints honoured here:

* Ground truth (including note names) is written to a *separate* JSON file.
  Audio file names never encode pitch, so the analyzer cannot cheat by
  reading the file name.
* Two takes share an opening that contains only even harmonics (missing
  fundamental).  One take then *adds* a low fundamental (the true pitch of the
  opening is retrospectively the low F0), the other genuinely jumps up an
  octave (the true pitch changes).
* Additional fixtures cover known-frequency harmonic groups, vibrato, and
  short noisy excerpts, plus silence and pure noise regions where no reliable
  pitch must be reported.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import numpy as np

from . import wavio
from .config import hz_from_midi

SR = 22050


@dataclass
class Segment:
    """A time-varying monophonic tone specification."""
    t0: float
    t1: float
    f0: float
    harmonics: dict[int, float]    # harmonic number -> amplitude
    vibrato_hz: float = 0.0
    vibrato_cents: float = 0.0
    vibrato_delay: float = 0.0
    noise_amp: float = 0.0
    attack: float = 0.02
    release: float = 0.05
    phase: float = 0.0

    def freq_at(self, rel: float) -> float:
        """Instantaneous f0 at `rel` seconds *since the segment start*."""
        cents_mod = 0.0
        if self.vibrato_hz > 0 and rel >= self.vibrato_delay:
            vm_t = rel - self.vibrato_delay
            # ramp vibrato in over 100 ms
            ramp = min(1.0, vm_t / 0.1)
            cents_mod = self.vibrato_cents * ramp * np.sin(
                2.0 * np.pi * self.vibrato_hz * vm_t
            )
        return self.f0 * 2.0 ** (cents_mod / 1200.0)


def render_segments(segments: list[Segment], duration: float, sr: int = SR,
                    seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Render additive segments with optional per-segment attack/release.

    Returns (audio, t).  A very small pitch glide is applied inside each
    segment so that the signal is not perfectly stationary (more realistic),
    but the glide stays well below the cent tolerance used for scoring.
    """
    n = int(round(duration * sr))
    t = np.arange(n) / sr
    out = np.zeros(n)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)

    for seg in segments:
        i0 = int(round(seg.t0 * sr))
        i1 = int(round(seg.t1 * sr))
        i1 = min(i1, n)
        if i1 <= i0:
            continue
        tt = t[i0:i1] - seg.t0
        dur = (i1 - i0) / sr
        # instantaneous frequency with phase integration
        phase = np.zeros(len(tt))
        inst_f = np.array([seg.freq_at(x) for x in tt])
        # integrate frequency to phase
        phase = 2.0 * np.pi * np.cumsum(inst_f) / sr + seg.phase
        env = np.ones(len(tt))
        a = min(1.0, max(1e-4, seg.attack))
        r = min(1.0, max(1e-4, seg.release))
        na = max(1, int(a * sr))
        nr = max(1, int(r * sr))
        na = min(na, len(tt))
        nr = min(nr, len(tt))
        env[:na] *= np.linspace(0.0, 1.0, na)
        env[-nr:] *= np.linspace(1.0, 0.0, nr)
        seg_sig = np.zeros(len(tt))
        for k, amp in seg.harmonics.items():
            if amp <= 0:
                continue
            seg_sig += amp * np.sin(k * phase + 0.13 * k)
        if seg.noise_amp > 0:
            seg_sig += seg.noise_amp * noise[i0:i1]
        out[i0:i1] += env * seg_sig
    return out, t


def normalize_peak(x: np.ndarray, peak: float = 0.85) -> np.ndarray:
    m = np.max(np.abs(x)) if x.size else 0.0
    if m <= 0:
        return x
    return x / m * peak


# ---------------------------------------------------------------------------
# Main two-take scenario
# ---------------------------------------------------------------------------

OPEN_F0 = 196.00         # G3: hidden fundamental of the shared opening
# The opening radiates only even harmonics of OPEN_F0 (392, 784, 1176, ...).
# Per-frame this is *indistinguishable* from a complete 392 Hz tone, so the
# literal spectral reading is 2*OPEN_F0.  Whether the true pitch is OPEN_F0 or
# 2*OPEN_F0 can only be decided by later evidence.
EVEN_ONLY = {2: 0.55, 4: 0.42, 6: 0.30, 8: 0.17, 10: 0.10}


def build_two_take_scenario() -> dict:
    """Return audio, sr, truth for take A and take B.

    Shared opening: even harmonics of OPEN_F0 only (missing fundamental).
      -> per-frame the densest reading is 2*OPEN_F0; the hidden truth is OPEN_F0
         and the region is labelled *ambiguous*.

    Take A: the low fundamental and odd harmonics *fade in smoothly* with no
      re-articulation, i.e. the same note with the fundamental revealed.
      -> later evidence proves f0 = OPEN_F0; offline revision must relabel the
         opening frames from 2*OPEN_F0 to OPEN_F0.

    Take B: a genuine leap up to 4*OPEN_F0 (784 Hz, two octaves above the
      hidden fundamental, one octave above the opening's literal reading) with a
      fresh attack.  -> the opening must NOT be relabelled; the leap is real.
    """
    open_start = 0.20
    open_end = 1.40          # opening lasts 1.2 s
    cont_end = 2.75          # continuation lasts 1.35 s
    tail = 0.30
    total = cont_end + tail

    opening = Segment(
        t0=open_start, t1=open_end, f0=OPEN_F0, harmonics=dict(EVEN_ONLY),
        vibrato_hz=5.0, vibrato_cents=6.0, vibrato_delay=0.30,
        noise_amp=0.006, attack=0.05, release=0.03,
    )

    # Take A: fundamental + odds fade in; continuous, no re-attack.
    a_harm = {1: 0.48, 2: 0.55, 3: 0.38, 4: 0.42, 5: 0.26,
              6: 0.30, 7: 0.16, 8: 0.17, 9: 0.10}
    a_cont = Segment(
        t0=open_end - 0.06, t1=cont_end, f0=OPEN_F0, harmonics=a_harm,
        vibrato_hz=5.0, vibrato_cents=6.0, noise_amp=0.006,
        attack=0.25, release=0.05,   # slow fade-in, no transient
    )

    # Take B: genuine leap to 4*OPEN_F0 with a fresh attack.
    b_f = 4.0 * OPEN_F0
    b_harm = {1: 0.50, 2: 0.34, 3: 0.22, 4: 0.13, 5: 0.07}
    b_cont = Segment(
        t0=open_end - 0.03, t1=cont_end, f0=b_f, harmonics=b_harm,
        vibrato_hz=5.0, vibrato_cents=6.0, noise_amp=0.006,
        attack=0.005, release=0.05,  # sharp re-articulation
    )

    a_audio, _ = render_segments([opening, a_cont], total, seed=11)
    b_audio, _ = render_segments([opening, b_cont], total, seed=12)
    a_audio = normalize_peak(a_audio)
    b_audio = normalize_peak(b_audio)

    def opening_regions():
        return [
            {"t0": 0.0, "t1": open_start, "kind": "silence", "f0": None,
             "label": "lead-in silence"},
            {"t0": open_start, "t1": open_end, "kind": "tone", "f0": OPEN_F0,
             "ambiguous": True,
             "spectral_reading_f0": 2.0 * OPEN_F0,
             "label": "shared opening: even harmonics only (missing fundamental)"},
        ]

    a_truth = {
        "take": "A",
        "sample_rate": SR,
        "hidden_open_f0_hz": OPEN_F0,
        "description": "even-harmonic opening; low fundamental then fades in "
                       "for the same note -> opening should be relabelled to the "
                       "low fundamental by offline revision",
        "segments": opening_regions() + [
            {"t0": open_end, "t1": cont_end, "kind": "tone", "f0": OPEN_F0,
             "ambiguous": False, "label": "low fundamental + odd harmonics"},
            {"t0": cont_end, "t1": total, "kind": "silence", "f0": None,
             "label": "trailing silence"},
        ],
        "revision_expectation": {
            "region": [open_start, open_end],
            "from_f0_hz": 2.0 * OPEN_F0,
            "to_f0_hz": OPEN_F0,
            "reason": "fundamental and odd harmonics appear during a legato "
                      "continuation of the same note",
        },
    }

    b_truth = {
        "take": "B",
        "sample_rate": SR,
        "hidden_open_f0_hz": OPEN_F0,
        "description": "same even-harmonic opening; a genuine leap up to "
                       "4*hidden fundamental with a fresh attack -> opening must "
                       "NOT be relabelled",
        "segments": opening_regions() + [
            {"t0": open_end, "t1": cont_end, "kind": "tone", "f0": 4.0 * OPEN_F0,
             "ambiguous": False, "label": "genuine octave leap"},
            {"t0": cont_end, "t1": total, "kind": "silence", "f0": None,
             "label": "trailing silence"},
        ],
        "jump": {"t_s": open_end, "from_reading_f0_hz": 2.0 * OPEN_F0,
                 "to_f0_hz": 4.0 * OPEN_F0},
        "revision_expectation": {"region": None,
                                 "reason": "new attack and new pitch: no relabel"},
    }

    return {
        "a": {"audio": a_audio, "sr": SR, "truth": a_truth},
        "b": {"audio": b_audio, "sr": SR, "truth": b_truth},
    }


# ---------------------------------------------------------------------------
# Calibration / robustness fixtures
# ---------------------------------------------------------------------------

def build_harmonic_fixtures() -> list[dict]:
    """Known-frequency harmonic groups for cent-error scoring.

    Each fixture is a pure-ish tone with a known f0 and a mix of harmonic
    amplitudes, rendered at 22050 Hz with light noise.
    """
    specs = [
        # (name, f0, harmonics, duration, noise)
        ("fixture_h1", 98.00, {1: 0.6, 2: 0.4, 3: 0.25, 4: 0.15, 5: 0.1}, 1.0, 0.004),
        ("fixture_h2", 146.83, {1: 0.5, 2: 0.5, 3: 0.3, 4: 0.2}, 0.9, 0.004),
        ("fixture_h3", 220.00, {1: 0.5, 2: 0.35, 3: 0.25, 4: 0.15}, 0.8, 0.004),
        ("fixture_h4", 293.66, {1: 0.55, 2: 0.3, 3: 0.2}, 0.7, 0.005),
        ("fixture_h5", 440.00, {1: 0.6, 2: 0.25, 3: 0.12}, 0.7, 0.005),
        ("fixture_h6", 55.00, {1: 0.45, 2: 0.4, 3: 0.3, 4: 0.2,
                               5: 0.14, 6: 0.1, 7: 0.07}, 1.4, 0.004),
        ("fixture_h7", 587.33, {1: 0.5, 2: 0.2}, 0.6, 0.005),
        ("fixture_h8", 783.99, {1: 0.5, 2: 0.15}, 0.55, 0.006),
        # missing fundamental but *odd* harmonics present -> f0 still recoverable
        ("fixture_missing_fund", 174.61, {2: 0.2, 3: 0.45, 4: 0.25, 5: 0.3},
         1.0, 0.004),
        # strong 2nd harmonic dominates -> max-peak would be an octave off
        ("fixture_dom2", 130.81, {1: 0.10, 2: 0.6, 3: 0.1, 4: 0.25}, 1.0, 0.004),
    ]
    out = []
    for name, f0, harm, dur, noise in specs:
        seg = Segment(t0=0.10, t1=0.10 + dur, f0=f0, harmonics=harm,
                      noise_amp=noise, attack=0.05, release=0.08)
        audio, _ = render_segments([seg], 0.10 + dur + 0.20, seed=hash(name) % 1000)
        audio = normalize_peak(audio, 0.7)
        out.append({
            "name": name, "audio": audio, "sr": SR,
            "truth": {
                "kind": "harmonic_group",
                "f0_hz": f0,
                "harmonics": harm,
                "tone_t0": 0.10, "tone_t1": 0.10 + dur,
                "sample_rate": SR,
            },
        })
    return out


def build_vibrato_fixtures() -> list[dict]:
    specs = [
        ("fixture_vib_slow", 196.00, 4.0, 25.0, 1.4),
        ("fixture_vib_fast", 261.63, 6.5, 35.0, 1.2),
        ("fixture_vib_wide", 330.00, 5.0, 60.0, 1.2),
        ("fixture_vib_low", 110.00, 5.5, 30.0, 1.5),
    ]
    out = []
    for name, f0, vhz, vcents, dur in specs:
        seg = Segment(t0=0.10, t1=0.10 + dur, f0=f0,
                      harmonics={1: 0.5, 2: 0.35, 3: 0.2, 4: 0.12},
                      vibrato_hz=vhz, vibrato_cents=vcents,
                      vibrato_delay=0.15, noise_amp=0.004,
                      attack=0.06, release=0.1)
        audio, _ = render_segments([seg], 0.10 + dur + 0.25, seed=hash(name) % 1000)
        audio = normalize_peak(audio, 0.7)
        out.append({
            "name": name, "audio": audio, "sr": SR,
            "truth": {
                "kind": "vibrato",
                "f0_center_hz": f0,
                "vibrato_hz": vhz,
                "vibrato_cents": vcents,
                "vibrato_delay": 0.15,
                "vibrato_ramp_s": 0.1,
                "tone_t0": 0.10, "tone_t1": 0.10 + dur,
                "sample_rate": SR,
            },
        })
    return out


def build_noisy_short_fixtures() -> list[dict]:
    specs = [
        ("fixture_noisy_short", 246.94, 0.22, 0.10),   # 220 ms tone
        ("fixture_noisy_shorter", 392.00, 0.14, 0.14),
        ("fixture_noisy_staccato", 174.61, 0.10, 0.12),
    ]
    out = []
    for name, f0, dur, noise in specs:
        seg = Segment(t0=0.08, t1=0.08 + dur, f0=f0,
                      harmonics={1: 0.45, 2: 0.3, 3: 0.2, 4: 0.12},
                      noise_amp=noise, attack=0.012, release=0.02)
        audio, _ = render_segments([seg], 0.08 + dur + 0.15, seed=hash(name) % 1000)
        audio = normalize_peak(audio, 0.6)
        out.append({
            "name": name, "audio": audio, "sr": SR,
            "truth": {
                "kind": "noisy_short",
                "f0_hz": f0,
                "tone_t0": 0.08, "tone_t1": 0.08 + dur,
                "noise_amp": noise,
                "sample_rate": SR,
            },
        })
    return out


def build_silence_noise_fixture() -> dict:
    """Silence then noise then a short tone then silence.

    Used to prove the analyzer returns *unreliable* frames rather than
    inventing a note in silence/noise.
    """
    total = 2.2
    n = int(total * SR)
    rng = np.random.default_rng(7)
    x = np.zeros(n)
    # noise-only region 0.5..1.1
    i0, i1 = int(0.5 * SR), int(1.1 * SR)
    x[i0:i1] = 0.05 * rng.standard_normal(i1 - i0)
    # quiet tone 1.3..1.8 (below/at silence threshold deliberately low level)
    seg = Segment(t0=1.35, t1=1.80, f0=hz_from_midi(60), harmonics={1: 0.4, 2: 0.25},
                  noise_amp=0.03, attack=0.03, release=0.03)
    tone, _ = render_segments([seg], total, seed=3)
    x = x + 0.25 * tone
    x = normalize_peak(x, 0.6)
    return {
        "name": "fixture_silence_noise", "audio": x, "sr": SR,
        "truth": {
            "kind": "silence_noise",
            "silence_regions": [[0.0, 0.5], [1.05, 1.30], [1.85, total]],
            "noise_regions": [[0.5, 1.1]],
            "tone_regions": [[1.35, 1.80]],
            "tone_f0_hz": hz_from_midi(60),
            "sample_rate": SR,
        },
    }


def build_all() -> dict:
    return {
        "two_take": build_two_take_scenario(),
        "harmonic": build_harmonic_fixtures(),
        "vibrato": build_vibrato_fixtures(),
        "noisy": build_noisy_short_fixtures(),
        "silence_noise": build_silence_noise_fixture(),
    }
