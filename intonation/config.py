"""Explicit analysis configuration.

Every time-related parameter is expressed in both samples and seconds and is
reported in the exported track so a reader can audit frame alignment.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
import math


@dataclass(frozen=True)
class AnalysisConfig:
    # --- framing ---------------------------------------------------------
    sample_rate: int = 22050
    frame_len: int = 2048          # 92.88 ms at 22050 Hz
    hop: int = 256                 # 11.61 ms at 22050 Hz (86.13 frames/s)
    window: str = "hann"
    nfft: int = 8192               # zero-padded spectrum for peak interpolation

    # --- pitch search range ---------------------------------------------
    fmin: float = 55.0             # A1
    fmax: float = 1600.0           # about G6
    max_harmonic_hz: float = 5000.0

    # --- spectral peak picking ------------------------------------------
    max_peaks: int = 80
    peak_floor_db: float = -75.0   # absolute floor for peak picking
    peak_floor_rel: float = 1e-4   # relative to loudest peak

    # --- candidate generation -------------------------------------------
    hypothesis_peaks: int = 16     # top peaks used as harmonic hypotheses
    max_hypothesis_order: int = 16 # a peak may be the k-th harmonic, k <= 16
    max_harmonic_order: int = 16   # harmonics scored for a candidate f0
    candidate_keep: int = 8        # candidates retained per frame
    candidate_score_ratio: float = 0.18  # keep candidates above this * best
    cluster_cents: float = 30.0    # hypotheses closer than this are merged
    octave_pair_cents: float = 45.0  # |cents(hi, 2*lo)| under this = octave pair
    viable_promotion_margin: float = 0.85  # promote a viable candidate over
                                          # a non-viable leader within this score ratio

    # --- harmonic comb scoring ------------------------------------------
    harmonic_weight_power: float = 1.0   # w_k = 1 / k**power
    match_tolerance_cents: float = 70.0  # how far a peak may sit from k*f0
    match_tolerance_min_hz: float = 8.0  # ... but never narrower than this
    strong_peak_ratio: float = 0.12      # peak >= this * loudest = "strong"
    missing_harmonic_penalty: float = 0.55
    odd_even_margin: float = 0.12        # odd/even energy ratio decision margin

    # --- YIN periodicity -------------------------------------------------
    yin_threshold: float = 0.30
    yin_max_period: int = 900      # ~24.5 Hz floor for the difference function

    # --- voicing / confidence -------------------------------------------
    silence_rms_db: float = -55.0
    voicing_min_periodicity: float = 0.62   # 1 - aperiodicity
    confidence_min: float = 0.55
    attack_flux_ratio: float = 4.0  # spectral flux relative to local median
    attack_guard_frames: int = 2

    # --- candidate retention / real-time decision ------------------------
    retention_ms: float = 450.0     # a candidate survives this long awaiting evidence
    decision_deadline_beats: float = 1.0
    tempo_bpm: float = 100.0        # beat = 0.6 s; deadline given per recording
    octave_ambiguity_margin: float = 0.55  # score ratio that counts as ambiguous

    # --- onset detection --------------------------------------------------
    onset_flux_percentile: float = 92.0
    onset_min_gap_ms: float = 90.0

    # --- offline revision -------------------------------------------------
    revision_lookahead_beats: float = 2.0   # evidence window after ambiguity
    revision_min_odd_energy: float = 0.10   # odd/even ratio that confirms lower f0

    # --- note segmentation ------------------------------------------------
    note_min_duration_ms: float = 70.0
    note_pitch_tolerance_cents: float = 60.0

    # --- derived helpers ---------------------------------------------------
    @property
    def frame_seconds(self) -> float:
        return self.frame_len / self.sample_rate

    @property
    def hop_seconds(self) -> float:
        return self.hop / self.sample_rate

    @property
    def beat_seconds(self) -> float:
        return 60.0 / self.tempo_bpm

    @property
    def frame_center_offset_samples(self) -> float:
        return self.frame_len / 2.0

    def frame_time(self, start_sample: int) -> dict:
        """Return the explicit sample/second time labels for one frame."""
        return {
            "sample_start": int(start_sample),
            "sample_center": float(start_sample + self.frame_len / 2.0),
            "sample_end": int(start_sample + self.frame_len),
            "t_start": start_sample / self.sample_rate,
            "t_center": (start_sample + self.frame_len / 2.0) / self.sample_rate,
            "t_end": (start_sample + self.frame_len) / self.sample_rate,
        }

    def to_dict(self) -> dict:
        d = asdict(self)
        d["frame_seconds"] = self.frame_seconds
        d["hop_seconds"] = self.hop_seconds
        d["beat_seconds"] = self.beat_seconds
        d["frame_center_offset_samples"] = self.frame_center_offset_samples
        return d


@dataclass(frozen=True)
class StreamingConfig:
    """Parameters of the real-time path.

    The streaming front-end is identical to the offline front-end; the only
    difference is that it may not look into the future.  The accompaniment
    admission gate is defined here.
    """

    tempo_bpm: float = 100.0
    decision_deadline_beats: float = 1.0
    retention_ms: float = 450.0
    min_confidence: float = 0.60
    min_sustain_ms: float = 60.0
    # "strict" fails closed on unresolved octave ambiguity; "best_effort"
    # commits the highest scoring candidate when the deadline expires.
    admission_policy: str = "strict"
    block_size: int = 256

    @property
    def beat_seconds(self) -> float:
        return 60.0 / self.tempo_bpm

    @property
    def deadline_seconds(self) -> float:
        return self.decision_deadline_beats * self.beat_seconds

    def to_dict(self) -> dict:
        d = asdict(self)
        d["beat_seconds"] = self.beat_seconds
        d["deadline_seconds"] = self.deadline_seconds
        return d


def cents(f: float, ref: float) -> float:
    return 1200.0 * math.log2(f / ref)


def hz_from_midi(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


def midi_from_hz(f: float) -> float:
    return 69.0 + 12.0 * math.log2(f / 440.0)


DEFAULT_CONFIG = AnalysisConfig()
DEFAULT_STREAMING = StreamingConfig()
