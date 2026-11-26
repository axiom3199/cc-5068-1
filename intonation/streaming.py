"""Streaming (causal) intonation analysis with a one-beat accompaniment gate.

The front-end is identical to the offline path, so what is measured is the
same.  What differs is *policy*: the streaming analyzer may not look ahead, so

* a note is only admitted to the accompaniment once it is voiced, confident,
  sustained for a minimum time, and not still octave-ambiguous;
* an octave decision that is still unresolved at its one-beat deadline is
  resolved under an explicit policy ("strict" fails closed and sends no pitch;
  "best_effort" commits the highest-scoring candidate), and either way the
  deadline is reported so the delay is auditable.

This module is import-safe and has no audio-device dependency; live input is
fed in fixed-size blocks by :mod:`server.live` or the browser UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import collections
import numpy as np

from .config import AnalysisConfig, StreamingConfig, cents
from .spectrum import SpectralFrontEnd
from .yin import yin_frame
from .candidates import generate_candidates, octave_relationships


@dataclass
class StreamFrame:
    index: int
    t_start: float
    t_center: float
    t_end: float
    sample_start: int
    state: str
    f0_hz: float | None
    confidence: float
    yin_aperiodicity: float
    ambiguous: bool
    lower_f0: float | None
    upper_f0: float | None
    is_attack: bool
    # gate results
    admitted: bool = False
    admission_reason: str = "pending"
    deadline_t: float | None = None
    overdue: bool = False
    deadline_resolution: str | None = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "t_start": round(self.t_start, 6),
            "t_center": round(self.t_center, 6),
            "t_end": round(self.t_end, 6),
            "sample_start": int(self.sample_start),
            "state": self.state,
            "f0_hz": round(self.f0_hz, 3) if self.f0_hz is not None else None,
            "confidence": round(self.confidence, 4),
            "yin_aperiodicity": round(self.yin_aperiodicity, 4),
            "ambiguous": self.ambiguous,
            "lower_f0_hz": round(self.lower_f0, 3) if self.lower_f0 else None,
            "upper_f0_hz": round(self.upper_f0, 3) if self.upper_f0 else None,
            "is_attack": self.is_attack,
            "admitted": self.admitted,
            "admission_reason": self.admission_reason,
            "deadline_t": round(self.deadline_t, 4) if self.deadline_t else None,
            "overdue": self.overdue,
            "deadline_resolution": self.deadline_resolution,
        }


class StreamingAnalyzer:
    """Block-driven causal analyzer.

    Feed audio with :meth:`push`; each call returns the newly produced frames.
    ``committed`` is the pitch currently allowed to drive the accompaniment
    (``None`` when the gate is closed).
    """

    def __init__(self, cfg: AnalysisConfig | None = None,
                 scfg: StreamingConfig | None = None):
        self.cfg = cfg or AnalysisConfig()
        self.scfg = scfg or StreamingConfig(
            tempo_bpm=self.cfg.tempo_bpm,
            decision_deadline_beats=self.cfg.decision_deadline_beats,
            retention_ms=self.cfg.retention_ms,
            admission_policy="strict",
        )
        self.fe = SpectralFrontEnd(self.cfg)
        self.buffer = np.zeros(0)
        self.total_samples = 0
        self.frame_index = 0
        self.frames: list[StreamFrame] = []

        # onset/attack detection state
        self._prev_mag: np.ndarray | None = None
        self._flux_hist = collections.deque(maxlen=40)

        # gate / deferral state
        self.committed: float | None = None
        self.committed_state: str = "silence"
        self._voiced_run = 0
        self._last_pitch: float | None = None
        self._ambiguous_since: float | None = None
        self._ambiguous_pair: tuple[float, float] | None = None
        self._deferral_pitch: float | None = None

    # ------------------------------------------------------------------
    def _local_attack(self, mag: np.ndarray) -> bool:
        if self._prev_mag is None or self._prev_mag.shape != mag.shape:
            self._prev_mag = mag.copy()
            return False
        diff = np.maximum(mag - self._prev_mag, 0.0)
        flux = float(np.sum(diff))
        norm = float(np.sum(mag)) + 1e-9
        val = flux / norm
        self._flux_hist.append(val)
        self._prev_mag = mag.copy()
        if len(self._flux_hist) < 8:
            return False
        thr = 4.0 * float(np.median(self._flux_hist))
        return val >= thr and val > 0.05

    def _deadline_seconds(self) -> float:
        return self.scfg.deadline_seconds

    def push(self, block: np.ndarray) -> list[StreamFrame]:
        block = np.asarray(block, dtype=np.float64)
        self.buffer = np.concatenate([self.buffer, block])
        self.total_samples += len(block)
        out: list[StreamFrame] = []

        while len(self.buffer) >= self.cfg.frame_len:
            frame = self.buffer[: self.cfg.frame_len]
            start_sample = self.frame_index * self.cfg.hop
            sf = self._analyze_block_frame(frame, start_sample)
            out.append(sf)
            self.frames.append(sf)
            self.buffer = self.buffer[self.cfg.hop:]
            self.frame_index += 1
        return out

    def _analyze_block_frame(self, frame: np.ndarray, start_sample: int) -> StreamFrame:
        cfg = self.cfg
        spec = self.fe.spectrum(frame)
        mag = spec.mag
        is_attack = self._local_attack(mag)
        y = yin_frame(frame, cfg.sample_rate, cfg.fmin, cfg.fmax,
                      cfg.yin_threshold, cfg.yin_max_period)
        t_start = start_sample / cfg.sample_rate
        t_center = (start_sample + cfg.frame_len / 2) / cfg.sample_rate
        t_end = (start_sample + cfg.frame_len) / cfg.sample_rate

        sf = StreamFrame(
            index=self.frame_index, t_start=t_start, t_center=t_center,
            t_end=t_end, sample_start=start_sample, state="unset",
            f0_hz=None, confidence=0.0, yin_aperiodicity=y.aperiodicity,
            ambiguous=False, lower_f0=None, upper_f0=None, is_attack=is_attack,
        )

        if spec.rms_db < cfg.silence_rms_db:
            sf.state = "silence"
            self._close_gate("silence")
            sf.admission_reason = "silence"
            return sf

        peaks = self.fe.pick_peaks(mag)
        if not peaks:
            sf.state = "noise"
            self._close_gate("noise")
            sf.admission_reason = "noise"
            return sf

        cands = generate_candidates(peaks, y, cfg)
        if not cands:
            sf.state = "noise"
            self._close_gate("noise")
            sf.admission_reason = "noise"
            return sf

        top = cands[0]
        periodicity = max(0.0, 1.0 - y.aperiodicity) if y.f0 > 0 else 0.0
        conf = 0.45 * periodicity + 0.55 * top.coverage * (0.6 + 0.4 * min(1.0, top.n_harmonics / 4))
        if top.viable:
            conf += 0.08
        conf = float(np.clip(conf, 0.0, 1.0))
        sf.confidence = conf

        looks_noisy = (periodicity < 0.30 and top.coverage < 0.75) or \
                      (top.n_harmonics < 2 and periodicity < 0.45)
        reliable = conf >= self.scfg.min_confidence and top.coverage >= 0.55

        if looks_noisy and not reliable:
            sf.state = "noise"
            self._close_gate("aperiodic")
            sf.admission_reason = "aperiodic"
            return sf
        if not reliable:
            sf.state = "attack" if is_attack else "noise"
            self._close_gate("attack" if is_attack else "low_confidence")
            sf.admission_reason = "unstable_attack" if is_attack else "low_confidence"
            return sf

        oct_rel = octave_relationships(cands, cfg)
        frame_ambiguous = bool(oct_rel["has_pair"])
        sf.ambiguous = frame_ambiguous
        sf.lower_f0 = oct_rel["lower_f0"]
        sf.upper_f0 = oct_rel["upper_f0"]
        sf.f0_hz = top.f0

        # ---- continuity: an ambiguous frame that continues an already
        # committed pitch is not a new octave question.  This is how a
        # sustained note that is intrinsically even-harmonic keeps its
        # established pitch instead of being re-decided every frame.  A fresh
        # attack breaks continuity (a leap must be re-decided).
        if self._ambiguous_since is None and frame_ambiguous and \
                not is_attack and \
                self.committed is not None and \
                abs(cents(top.f0, self.committed)) <= 80.0:
            sf.state = "voiced"
            sf.ambiguous = True
            sf.admitted = True
            sf.admission_reason = "continuation"
            self.committed = top.f0
            self.committed_state = "voiced"
            self._last_pitch = top.f0
            self._voiced_run = max(1, self._voiced_run + 1)
            return sf

        # ---- fresh octave ambiguity: open a deferral -------------------
        if self._ambiguous_since is None and frame_ambiguous:
            self._ambiguous_since = t_center
            self._ambiguous_pair = (sf.lower_f0, sf.upper_f0)
            self._deferral_pitch = top.f0

        if self._ambiguous_since is not None:
            sf.deadline_t = self._ambiguous_since + self._deadline_seconds()
            lower_true = self._lower_evidence(cands, top, self._ambiguous_pair)
            if lower_true is not None:
                sf.state = "voiced"
                sf.ambiguous = False
                sf.f0_hz = lower_true
                self._resolve_deferral(sf, lower_true, "resolved_lower_evidence")
                # a resolved deferral starts a fresh sustain count; admit once
                # the minimum sustain time has elapsed
                sf.admitted = False
                sf.admission_reason = "resolved_octave_filling"
                return sf
            if t_end >= sf.deadline_t:
                self._resolve_at_deadline(sf)
                return sf
            # still ambiguous and before deadline: hold the gate closed, but
            # expose the literal reading so the UI can show "waiting".
            sf.state = "ambiguous"
            sf.admitted = False
            sf.admission_reason = "deferred_octave"
            self.committed = None
            self.committed_state = "ambiguous"
            self._voiced_run = 0
            return sf

        sf.state = "voiced"

        if self._last_pitch is not None and \
                abs(cents(top.f0, self._last_pitch)) <= 80.0:
            self._voiced_run += 1
        else:
            self._voiced_run = 1
        self._last_pitch = top.f0

        min_run = max(1, int(round(self.scfg.min_sustain_ms / 1000.0 / cfg.hop_seconds)))
        if self._voiced_run >= min_run:
            sf.admitted = True
            sf.admission_reason = "admitted"
            self.committed = top.f0
            self.committed_state = "voiced"
            sf.deadline_resolution = "n/a"
        else:
            sf.admitted = False
            sf.admission_reason = "sustain_filling"
            self.committed = None
            self.committed_state = "filling"
            sf.deadline_t = t_center + self._deadline_seconds()
        return sf

    def _close_gate(self, reason: str):
        self._voiced_run = 0
        self._last_pitch = None
        self.committed = None
        self.committed_state = reason
        self._ambiguous_since = None
        self._ambiguous_pair = None

    def _lower_evidence(self, cands, top, pair) -> float | None:
        """Causal disambiguation: does this frame show the *low* pitch is real?

        The lower octave is proven present only when a candidate at the lower
        pitch has a genuinely strong fundamental (not a stray formant/noise
        peak) relative to its own harmonics.  The mere presence of a lower
        candidate, or a sub-octave fold of the stronger upper reading, is not
        evidence.
        """
        if pair is None:
            return None
        lower_ref = pair[0]
        for c in cands:
            if abs(cents(c.f0, lower_ref)) > 45.0 or not c.viable:
                continue
            fund_amp = 0.0
            harm_amp = 0.0
            for m in c.matched:
                if m.k == 1:
                    fund_amp = m.peak_amp
                elif 2 <= m.k <= 5:
                    harm_amp = max(harm_amp, m.peak_amp)
            # the lower fundamental must be clearly radiated: at least ~40% of
            # its strongest low harmonic.  This excludes cases where k=1 is only
            # a weak noise/formant peak (an alias of the upper reading).
            if c.fundamental_present and harm_amp > 0 and \
                    fund_amp >= 0.40 * harm_amp and c.coverage >= 0.75:
                return c.f0
        return None

    def _resolve_deferral(self, sf: "StreamFrame", chosen: float, reason: str):
        sf.deadline_resolution = reason
        sf.admitted = False
        sf.admission_reason = f"{reason}_filling"
        self._ambiguous_since = None
        self._ambiguous_pair = None
        self._last_pitch = chosen
        self._voiced_run = 1

    def _resolve_at_deadline(self, sf: "StreamFrame"):
        """One-beat deadline expired with no causal disambiguating evidence."""
        deadline = self._ambiguous_since + self._deadline_seconds()
        sf.deadline_t = deadline
        sf.overdue = True
        if self.scfg.admission_policy == "best_effort":
            chosen = sf.f0_hz
            sf.deadline_resolution = "deadline_best_effort"
            sf.admitted = True
            sf.admission_reason = "deadline_best_effort"
            sf.state = "voiced"
            self.committed = chosen
            self.committed_state = "voiced_forced"
            self._last_pitch = chosen
            self._voiced_run = 1
        else:
            sf.deadline_resolution = "deadline_strict_fail_closed"
            sf.admitted = False
            sf.admission_reason = "deadline_strict_fail_closed"
            sf.state = "ambiguous"
            self.committed = None
            self.committed_state = "ambiguous_failed_closed"
            self._voiced_run = 0
            self._last_pitch = None
        self._ambiguous_since = None
        self._ambiguous_pair = None

    # ------------------------------------------------------------------
    def finish(self) -> dict:
        return {
            "config": self.cfg.to_dict(),
            "streaming_config": self.scfg.to_dict(),
            "frames": [f.to_dict() for f in self.frames],
            "n_frames": len(self.frames),
            "admitted_frames": sum(1 for f in self.frames if f.admitted),
            "deadline_overdue_frames": sum(1 for f in self.frames if f.overdue),
        }


def stream_analyze(x: np.ndarray, cfg: AnalysisConfig | None = None,
                   scfg: StreamingConfig | None = None,
                   block_size: int = 256) -> dict:
    """Run the causal analyzer over a whole buffer in blocks (for tests/UI)."""
    an = StreamingAnalyzer(cfg, scfg)
    n = len(x)
    for i in range(0, n, block_size):
        an.push(x[i:i + block_size])
    return an.finish()
