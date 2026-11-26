"""Per-frame evidence extraction and voicing classification.

Produces one :class:`Frame` per analysis hop.  A frame is *allowed to have no
reliable pitch*: silence, noise and unstable attack regions are labelled as
such rather than being forced into a note.  The raw per-frame f0 estimate is
never smoothed here — smoothing is only ever a display overlay, because a
smoothed curve looking stable is not evidence of accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import AnalysisConfig, cents
from .spectrum import SpectralFrontEnd, detect_attacks
from .yin import YinResult, yin_frame
from .candidates import Candidate, generate_candidates, octave_relationships


@dataclass
class Frame:
    index: int
    sample_start: int
    sample_center: float
    sample_end: int
    t_start: float
    t_center: float
    t_end: float

    rms_db: float
    state: str                  # silence | noise | attack | voiced | ambiguous
    f0_hz: float | None
    confidence: float

    yin_f0: float
    yin_aperiodicity: float
    yin_period: float
    yin_threshold_met: bool

    candidates: list[Candidate]
    selected_rank: int
    selected_source: str | None
    n_candidates_total: int
    rejection_summary: dict

    ambiguous_octave: bool
    lower_octave_f0: float | None
    upper_octave_f0: float | None
    lower_octave_ratio: float | None

    is_attack: bool
    attack_hold: bool

    peaks: list[dict]           # inspectable top spectral peaks

    # filled later by tracking / revision
    tracked_f0: float | None = None
    tracked_state: str | None = None
    decision: str | None = None        # committed | deferred | rejected | ...
    octave_corrected: bool = False
    revision_note: str | None = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "time": {
                "sample_start": self.sample_start,
                "sample_center": round(self.sample_center, 2),
                "sample_end": self.sample_end,
                "t_start": round(self.t_start, 6),
                "t_center": round(self.t_center, 6),
                "t_end": round(self.t_end, 6),
            },
            "rms_db": round(self.rms_db, 2),
            "state": self.state,
            "f0_hz": round(self.f0_hz, 3) if self.f0_hz is not None else None,
            "confidence": round(self.confidence, 4),
            "yin": {
                "f0_hz": round(self.yin_f0, 3),
                "aperiodicity": round(self.yin_aperiodicity, 4),
                "period_samples": round(self.yin_period, 3),
                "threshold_met": self.yin_threshold_met,
            },
            "selected_rank": self.selected_rank,
            "selected_source": self.selected_source,
            "n_candidates_total": self.n_candidates_total,
            "candidates": [c.to_dict() for c in self.candidates],
            "rejection_summary": self.rejection_summary,
            "octave": {
                "ambiguous": self.ambiguous_octave,
                "lower_f0_hz": (round(self.lower_octave_f0, 3)
                                if self.lower_octave_f0 else None),
                "upper_f0_hz": (round(self.upper_octave_f0, 3)
                                if self.upper_octave_f0 else None),
                "lower_ratio": (round(self.lower_octave_ratio, 4)
                                if self.lower_octave_ratio is not None else None),
            },
            "attack": {"onset": self.is_attack, "guard": self.attack_hold},
            "peaks": self.peaks,
            "tracked_f0_hz": (round(self.tracked_f0, 3)
                              if self.tracked_f0 is not None else None),
            "tracked_state": self.tracked_state,
            "decision": self.decision,
            "octave_corrected": self.octave_corrected,
            "revision_note": self.revision_note,
        }


def _confidence(cand: Candidate | None, yin: YinResult, cfg: AnalysisConfig) -> float:
    periodicity = max(0.0, 1.0 - yin.aperiodicity) if yin.f0 > 0 else 0.0
    if cand is None:
        return 0.15 * periodicity
    # comb reliability discounted by tuning residual
    mean_res = 0.0
    if cand.matched:
        wsum = 0.0
        for m in cand.matched:
            w = (m.peak_amp / max(1e-9, max(x.peak_amp for x in cand.matched))) / m.k
            mean_res += w * abs(m.cents_err)
            wsum += w
        if wsum > 0:
            mean_res = mean_res / wsum
    res_quality = max(0.0, 1.0 - mean_res / max(cfg.match_tolerance_cents, 1e-6))
    comb = cand.coverage * (0.55 + 0.45 * res_quality)
    comb = comb * (0.6 + 0.4 * min(1.0, cand.n_harmonics / 4.0))
    conf = 0.45 * periodicity + 0.55 * comb
    if cand.viable:
        conf += 0.08
    return float(np.clip(conf, 0.0, 1.0))


def analyze_frame(
    frame: np.ndarray,
    index: int,
    sample_start: int,
    fe: SpectralFrontEnd,
    cfg: AnalysisConfig,
    is_attack: bool,
    attack_hold: bool,
    noise_floor_db: float,
) -> Frame:
    spec = fe.spectrum(frame)
    y = yin_frame(frame, cfg.sample_rate, cfg.fmin, cfg.fmax,
                  cfg.yin_threshold, cfg.yin_max_period)
    tm = cfg.frame_time(sample_start)

    f = Frame(
        index=index,
        sample_start=tm["sample_start"],
        sample_center=tm["sample_center"],
        sample_end=tm["sample_end"],
        t_start=tm["t_start"], t_center=tm["t_center"], t_end=tm["t_end"],
        rms_db=spec.rms_db,
        state="unset", f0_hz=None, confidence=0.0,
        yin_f0=y.f0, yin_aperiodicity=y.aperiodicity,
        yin_period=y.period, yin_threshold_met=y.threshold_met,
        candidates=[], selected_rank=-1, selected_source=None,
        n_candidates_total=0, rejection_summary={},
        ambiguous_octave=False, lower_octave_f0=None, upper_octave_f0=None,
        lower_octave_ratio=None,
        is_attack=is_attack, attack_hold=attack_hold,
        peaks=[],
    )

    # --- silence gate -------------------------------------------------
    if spec.rms_db < cfg.silence_rms_db:
        f.state = "silence"
        f.decision = "silence"
        return f

    peaks = fe.pick_peaks(spec.mag)
    f.peaks = [p.to_dict() for p in peaks[:12]]
    if not peaks:
        f.state = "noise"
        f.decision = "no_peaks"
        return f

    cands = generate_candidates(peaks, y, cfg)
    f.candidates = cands
    f.n_candidates_total = len(cands)
    if not cands:
        f.state = "noise"
        f.decision = "no_candidates"
        return f

    top = cands[0]
    conf = _confidence(top, y, cfg)
    f.confidence = conf
    f.selected_rank = top.rank
    f.selected_source = top.source

    # --- voicing decision --------------------------------------------
    periodicity = max(0.0, 1.0 - y.aperiodicity) if y.f0 > 0 else 0.0
    looks_noisy = (periodicity < 0.30 and top.coverage < 0.75) or \
                  (top.n_harmonics < 2 and periodicity < 0.45)
    reliable = conf >= cfg.confidence_min and top.coverage >= 0.55

    if looks_noisy and not reliable:
        f.state = "noise"
        f.decision = "aperiodic"
        f.f0_hz = None
        return f

    if not reliable:
        # attack transient or weak/uncertain: do not force a note
        if attack_hold:
            f.state = "attack"
            f.decision = "attack_unstable"
        else:
            f.state = "noise"
            f.decision = "low_confidence"
        f.f0_hz = None
        return f

    # --- octave ambiguity --------------------------------------------
    oct_rel = octave_relationships(cands, cfg)
    f.ambiguous_octave = bool(oct_rel["has_pair"])
    f.lower_octave_f0 = oct_rel["lower_f0"]
    f.upper_octave_f0 = oct_rel["upper_f0"]
    f.lower_octave_ratio = oct_rel["lower_ratio"]

    f.f0_hz = top.f0
    if f.ambiguous_octave:
        f.state = "ambiguous"
        f.decision = "octave_deferred"
    else:
        f.state = "voiced"
        f.decision = "tentative"

    # rejection summary (inspectable)
    summary: dict[str, int] = {}
    for c in cands:
        if c.rejection:
            summary[c.rejection] = summary.get(c.rejection, 0) + 1
    f.rejection_summary = summary
    return f


def analyze_audio(x: np.ndarray, sr: int, cfg: AnalysisConfig | None = None,
                  progress=None) -> dict:
    """Full frame-level analysis.  Returns dict with frames + config metadata.

    This is the evidence layer; it makes only tentative per-frame decisions.
    Tracking, deferral and revision happen in :mod:`intonation.tracker`.
    """
    from .wavio import resample
    cfg = cfg or AnalysisConfig()
    if sr != cfg.sample_rate:
        x = resample(x, sr, cfg.sample_rate)
        sr = cfg.sample_rate
    x = np.asarray(x, dtype=np.float64)

    fe = SpectralFrontEnd(cfg)
    mag, starts = fe.stft(x)

    # noise floor estimate from the quietest 5% of frames
    if mag.shape[0] == 0:
        return {"frames": [], "config": cfg.to_dict(), "audio": {"n_samples": len(x),
                "sample_rate": sr, "duration_s": len(x) / sr}}

    frame_rms_db = np.array([
        20.0 * np.log10(max(1e-12, np.sqrt(np.mean(
            x[s:s + cfg.frame_len] ** 2)))) for s in starts
    ])
    noise_floor_db = float(np.percentile(frame_rms_db, 5))

    from .spectrum import spectral_flux
    flux = spectral_flux(mag)
    is_attack, attack_hold, onset_thr = detect_attacks(flux, cfg)

    frames: list[Frame] = []
    for i, s in enumerate(starts):
        if progress and (i % 50 == 0):
            progress(i, len(starts))
        frame = x[s:s + cfg.frame_len]
        fr = analyze_frame(frame, int(i), int(s), fe, cfg,
                           bool(is_attack[i]), bool(attack_hold[i]), noise_floor_db)
        frames.append(fr)

    return {
        "frames": frames,
        "config": cfg.to_dict(),
        "audio": {
            "n_samples": int(len(x)),
            "sample_rate": int(sr),
            "duration_s": float(len(x) / sr),
        },
        "onset": {
            "is_attack": [bool(v) for v in is_attack],
            "threshold": onset_thr,
            "flux": [round(float(v), 5) for v in flux],
        },
        "noise_floor_db": noise_floor_db,
    }
