"""Top-level offline analysis: audio -> evidence -> tracked pitch -> notes."""

from __future__ import annotations

import numpy as np

from .config import AnalysisConfig
from .analyzer import analyze_audio
from .tracker import track_and_segment


def analyze_signal(x: np.ndarray, sr: int, cfg: AnalysisConfig | None = None,
                   allow_revision: bool = True, progress=None) -> dict:
    cfg = cfg or AnalysisConfig()
    raw = analyze_audio(x, sr, cfg, progress=progress)
    frames = raw["frames"]
    tracked = track_and_segment(frames, cfg, allow_revision=allow_revision)

    return {
        "config": raw["config"],
        "audio": raw["audio"],
        "noise_floor_db": raw["noise_floor_db"],
        "onset": raw["onset"],
        "frames": [f.to_dict() for f in frames],
        "notes": tracked["notes"],
        "deferrals": tracked["deferrals"],
        "revisions": tracked["revisions"],
        "summary": summarize(frames, tracked),
    }


def summarize(frames, tracked) -> dict:
    n = len(frames)
    voiced = sum(1 for f in frames if f.tracked_f0 is not None)
    ambiguous = sum(1 for f in frames if f.ambiguous_octave)
    attack = sum(1 for f in frames if f.state == "attack")
    noise = sum(1 for f in frames if f.state == "noise")
    silence = sum(1 for f in frames if f.state == "silence")
    relabelled = sum(1 for f in frames if f.octave_corrected)
    return {
        "n_frames": n,
        "voiced_frames": voiced,
        "unvoiced_frames": n - voiced,
        "ambiguous_frames": ambiguous,
        "attack_frames": attack,
        "noise_frames": noise,
        "silence_frames": silence,
        "relabelled_frames": relabelled,
        "n_notes": len(tracked["notes"]),
        "n_deferrals": len(tracked["deferrals"]),
        "n_revisions": len(tracked["revisions"]),
        "valid_coverage": (voiced / n) if n else 0.0,
    }
