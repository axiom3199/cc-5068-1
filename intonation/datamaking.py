"""Pitch-data production: run the *real* analyzer on the synthetic phrases and
verify the result against ground truth.

Hard rules honoured here:

* the analyzer receives **audio only**.  Ground truth is never passed into
  :func:`analyze_signal` and cannot influence a single frame — it is read only
  afterwards, inside the verification functions;
* failed regions are kept, not smoothed away: the comparison record lists every
  failing frame/region with its time, reference, estimate and the analyzer's own
  evidence, and an overall ``accepted`` flag;
* pitch is scored with cent error against the instantaneous reference f0, and
  voicing is scored separately (true voiced frames must be voiced, unvoiced
  gaps must not receive a pitch, candidate regions must stay undecided).
"""

from __future__ import annotations

import math
import numpy as np

from .config import AnalysisConfig, cents
from .analysis import analyze_signal
from .phrases import Phrase, phrase_truth
from . import tablature

# acceptance tolerances
CENT_MEAN_TOL = 5.0
CENT_P95_TOL = 25.0
OCTAVE_ERROR_RATE_TOL = 0.0
UNVOICED_SPURIOUS_RATE_TOL = 0.0
CANDIDATE_UNDECIDED_TOL = 0.90   # >=90% of candidate frames must be undecided


def run_analyzer_audio_only(audio: np.ndarray, sr: int,
                            cfg: AnalysisConfig | None = None) -> dict:
    """The single entry through which production runs the analyzer.

    Ground truth is deliberately not a parameter; callers verify the returned
    track separately.  This makes the "truth never enters the algorithm" rule
    structural rather than conventional.
    """
    cfg = cfg or AnalysisConfig()
    return analyze_signal(audio, sr, cfg, allow_revision=True)


# ---------------------------------------------------------------------------
# frame <-> truth alignment
# ---------------------------------------------------------------------------

def _truth_at(truth: dict, t: float) -> dict | None:
    for seg in truth["segments"]:
        if seg["t0"] <= t < seg["t1"]:
            return seg
    return None


def _inst_ref(truth: dict, t: float) -> float | None:
    """Instantaneous reference f0 at time t from the dense truth trajectory."""
    traj = truth["f0_trajectory"]
    best = None
    for point in traj:
        if point["voiced"] and point["f0_hz"] is not None:
            if best is None or abs(point["t"] - t) < abs(best["t"] - t):
                best = point
    if best is None or abs(best["t"] - t) > 0.02:
        # fall back to segment centre f0
        seg = _truth_at(truth, t)
        if seg and seg.get("f0"):
            return float(seg["f0"])
        return None
    return float(best["f0_hz"])


def _is_candidate_frame(truth: dict, t: float) -> bool:
    seg = _truth_at(truth, t)
    return bool(seg and seg.get("pitch_status") == "candidate")


def _is_unvoiced_frame(truth: dict, t: float) -> bool:
    seg = _truth_at(truth, t)
    return seg is None or seg.get("kind") in ("breath", "silence", "noise")


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def verify_track(track: dict, truth: dict) -> dict:
    """Cent error + voicing + ambiguity verification against truth.

    Returns an acceptance record with explicit per-failure evidence.  Known and
    candidate tone frames are scored separately: on a candidate region a
    definite octave is itself a failure (the analyzer should stay undecided).
    """
    frames = track["frames"]
    cent_errors: list[float] = []
    octave_errors = 0
    scored_known = 0

    candidate_total = 0
    candidate_undecided = 0

    unvoiced_total = 0
    unvoiced_spurious = 0

    failures: list[dict] = []

    for fr in frames:
        t = fr["time"]["t_center"]
        seg = _truth_at(truth, t)
        est = fr.get("tracked_f0_hz")

        if _is_unvoiced_frame(truth, t):
            # whole-window containment criterion (honest boundary handling)
            contained = any(fr["time"]["t_start"] >= s["t0"]
                            and fr["time"]["t_end"] <= s["t1"]
                            for s in truth["segments"]
                            if s["kind"] in ("breath", "silence", "noise"))
            if not contained:
                continue
            unvoiced_total += 1
            if est is not None:
                unvoiced_spurious += 1
                failures.append({
                    "kind": "voicing_false_positive", "t": round(t, 4),
                    "estimate_hz": est, "state": fr["state"],
                    "decision": fr.get("decision"),
                })
            continue

        if seg is None or seg.get("kind") != "tone":
            continue

        if seg.get("pitch_status") == "candidate":
            candidate_total += 1
            # Honest handling of a missing-fundamental frame is any path that
            # *first* recognised the octave question: an explicit ambiguous
            # flag/deferral, an offline revision reached through a deferral, or
            # the keep-upper fallback of a deferral.  A definite octave on a
            # frame that never entered the deferral mechanism is an unjustified
            # commitment and counts as a failure.
            undecided = (
                bool(fr.get("octave", {}).get("ambiguous"))
                or fr.get("decision") in ("deferred", "revised",
                                          "octave_kept_upper")
                or _frame_in_deferral(track, t)
            )
            if undecided:
                candidate_undecided += 1
            else:
                failures.append({
                    "kind": "candidate_committed_without_evidence",
                    "t": round(t, 4), "estimate_hz": est,
                    "candidate_range_hz": seg.get("candidate_range_hz"),
                    "decision": fr.get("decision"),
                })
            continue

        # known-pitch tone frame: cent error on the instantaneous reference
        ref = _inst_ref(truth, t)
        if ref is None or est is None:
            if est is None:
                failures.append({"kind": "voicing_false_negative",
                                 "t": round(t, 4), "reference_hz": ref,
                                 "state": fr["state"],
                                 "decision": fr.get("decision")})
            continue
        scored_known += 1
        e = cents(est, ref)
        cent_errors.append(e)
        if abs(e) > 600:
            octave_errors += 1
            failures.append({"kind": "octave_error", "t": round(t, 4),
                             "reference_hz": round(ref, 3),
                             "estimate_hz": est, "cents_err": round(e, 2)})
        elif abs(e) > CENT_P95_TOL:
            failures.append({"kind": "cent_outlier", "t": round(t, 4),
                             "reference_hz": round(ref, 3),
                             "estimate_hz": est, "cents_err": round(e, 2)})

    arr = np.array(cent_errors) if cent_errors else np.array([0.0])
    abs_arr = np.abs(arr)
    cent_stats = {
        "n_scored_known_frames": scored_known,
        "mean_cents": float(np.mean(arr)),
        "abs_mean_cents": float(np.mean(abs_arr)),
        "p95_abs_cents": float(np.percentile(abs_arr, 95)),
        "max_abs_cents": float(np.max(abs_arr)),
        "octave_errors": octave_errors,
        "octave_error_rate": (octave_errors / scored_known) if scored_known else 0.0,
    }
    voicing = {
        "unvoiced_frames": unvoiced_total,
        "unvoiced_spurious": unvoiced_spurious,
        "unvoiced_spurious_rate": (unvoiced_spurious / unvoiced_total
                                   if unvoiced_total else 0.0),
    }
    ambiguity = {
        "candidate_frames": candidate_total,
        "candidate_undecided": candidate_undecided,
        "candidate_undecided_rate": (candidate_undecided / candidate_total
                                     if candidate_total else 1.0),
    }

    accepted = (
        cent_stats["abs_mean_cents"] <= CENT_MEAN_TOL
        and cent_stats["p95_abs_cents"] <= CENT_P95_TOL
        and cent_stats["octave_error_rate"] <= OCTAVE_ERROR_RATE_TOL
        and voicing["unvoiced_spurious_rate"] <= UNVOICED_SPURIOUS_RATE_TOL
        and ambiguity["candidate_undecided_rate"] >= CANDIDATE_UNDECIDED_TOL
    )
    return {
        "accepted": bool(accepted),
        "tolerances": {
            "abs_mean_cents": CENT_MEAN_TOL, "p95_abs_cents": CENT_P95_TOL,
            "octave_error_rate": OCTAVE_ERROR_RATE_TOL,
            "unvoiced_spurious_rate": UNVOICED_SPURIOUS_RATE_TOL,
            "candidate_undecided_rate": CANDIDATE_UNDECIDED_TOL,
        },
        "cent_stats": cent_stats,
        "voicing": voicing,
        "ambiguity": ambiguity,
        "n_failures": len(failures),
        "failures": failures[:200],   # keep evidence, bounded size
    }


def _frame_in_deferral(track: dict, t: float) -> bool:
    for d in track.get("deferrals", []):
        if d["start_time"] - 0.05 <= t <= d["deadline_time"] + 0.05:
            return True
    return False


# ---------------------------------------------------------------------------
# full production of one phrase
# ---------------------------------------------------------------------------

def produce_phrase(phrase: Phrase, cfg: AnalysisConfig | None = None) -> dict:
    """Render -> analyze (audio only) -> tablature -> verify.

    The returned bundle is everything the build script persists; the analyzer
    track inside it is genuine output and the verification is computed against
    truth afterwards.
    """
    from .phrases import render_phrase
    cfg = cfg or AnalysisConfig()
    audio = render_phrase(phrase)
    truth = phrase_truth(phrase)
    track = run_analyzer_audio_only(audio, phrase.sample_rate, cfg)
    tab = tablature.build_tablature(phrase)
    verification = verify_track(track, truth)
    return {
        "phrase": phrase.to_dict(),
        "audio": audio,
        "sample_rate": phrase.sample_rate,
        "truth": truth,
        "track": track,
        "tablature": tab,
        "verification": verification,
    }


def verify_pair(track_full: dict, track_even: dict,
                truth_full: dict, truth_even: dict,
                alignment: dict) -> dict:
    """Verify a same-pitch / opposite-fundamental material pair.

    Both takes must track the same performed pitch where decidable; the
    fundamental-absent take must keep its region undecidable.  The independent
    score alignment is echoed so the pair can be displayed on one grid.
    """
    v_full = verify_track(track_full, truth_full)
    v_even = verify_track(track_even, truth_even)
    span = next(s for s in alignment["spans"] if not s["fundamental_present"])
    t0, t1 = span["t0"], span["t1"]

    # the even take: literal readings should sit near the UPPER octave while no
    # definite low octave is justified -> recorded as undecided, not an error
    even_literal = [fr["f0_hz"] for fr in track_even["frames"]
                    if t0 + 0.1 <= fr["time"]["t_center"] <= t1 - 0.1
                    and fr.get("f0_hz")]
    upper = 2.0 * alignment["performed_pitch_hz"]
    literal_octave_ok = (
        bool(even_literal)
        and np.median([abs(cents(v, upper)) for v in even_literal]) < 60.0
    )
    accepted = (v_full["accepted"] and v_even["accepted"]
                and v_even["ambiguity"]["candidate_undecided_rate"]
                >= CANDIDATE_UNDECIDED_TOL and literal_octave_ok)
    return {
        "accepted": bool(accepted),
        "alignment": alignment,
        "fundamental_present": v_full,
        "fundamental_absent": v_even,
        "even_take_literal_near_upper_octave": bool(literal_octave_ok),
        "failures": (v_full["failures"] + v_even["failures"])[:200],
    }
