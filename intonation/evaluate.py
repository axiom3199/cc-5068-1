"""Quantitative evaluation against the synthetic ground truth.

Metrics reported (all computed on raw per-frame estimates, *not* on a smoothed
display curve):

* cent error distribution over frames that should be voiced,
* octave-error rate,
* valid coverage (fraction of truly-tone frames that receive a reliable
  pitch) and spurious coverage (fraction of silence/noise frames wrongly
  given a pitch),
* jump delay (how late a genuine pitch leap is declared relative to truth),
* revision correctness for the two-take scenario.
"""

from __future__ import annotations

import math
import numpy as np

from .config import cents


def _segment_f0(truth: dict, t: float) -> tuple[str, float | None, bool]:
    """Return (kind, f0, ambiguous) for time t from a truth segment list."""
    for seg in truth["segments"]:
        if seg["t0"] <= t < seg["t1"]:
            return seg["kind"], seg.get("f0"), bool(seg.get("ambiguous", False))
    return "outside", None, False


def _tone_frames(result: dict, truth: dict) -> list[dict]:
    out = []
    for fr in result["frames"]:
        kind, f0, amb = _segment_f0(truth, fr["time"]["t_center"])
        if kind == "tone":
            out.append({"frame": fr, "f0": f0, "ambiguous": amb})
    return out


def cent_stats(frames: list[dict], f0_ref: float | None,
               key: str = "f0_hz") -> dict:
    errs = []
    for item in frames:
        fr = item["frame"]
        v = fr.get(key)
        if v is None:
            continue
        ref = item["f0"] if f0_ref is None else f0_ref
        if ref is None:
            continue
        errs.append(cents(v, ref))
    if not errs:
        return {"n": 0}
    errs = np.array(errs)
    abs_errs = np.abs(errs)
    return {
        "n": int(len(errs)),
        "mean_cents": float(np.mean(errs)),
        "abs_mean_cents": float(np.mean(abs_errs)),
        "median_abs_cents": float(np.median(abs_errs)),
        "p95_abs_cents": float(np.percentile(abs_errs, 95)),
        "max_abs_cents": float(np.max(abs_errs)),
        "octave_errors": int(np.sum(abs_errs > 600)),
        "octave_error_rate": float(np.mean(abs_errs > 600)),
    }


def _frame_contained(fr: dict, a: float, b: float) -> bool:
    """True only if the whole analysis window lies inside [a, b].

    A frame whose window straddles a boundary genuinely contains energy from
    both sides, so it cannot be counted as "spurious" for the silence/noise
    side.  Restricting the metric to fully-contained frames is the honest
    criterion and removes the inherent frame-length boundary smear.
    """
    return fr["time"]["t_start"] >= a and fr["time"]["t_end"] <= b


def evaluate_two_take(result: dict, truth: dict, key: str = "tracked_f0_hz") -> dict:
    tone = _tone_frames(result, truth)
    # non-ambiguous tone frames only for the strict cent/octave metric
    strict = [it for it in tone if not it["ambiguous"]]
    amb = [it for it in tone if it["ambiguous"]]

    voiced = [it for it in tone if it["frame"].get(key) is not None]
    coverage = len(voiced) / len(tone) if tone else 0.0

    # spurious coverage: frames fully inside silence/noise that still get a pitch
    spurious = 0
    sp_total = 0
    for fr in result["frames"]:
        tc = fr["time"]["t_center"]
        kind, _, _ = _segment_f0(truth, tc)
        if kind not in ("silence", "noise"):
            continue
        seg = next(s for s in truth["segments"]
                   if s["t0"] <= tc < s["t1"] and s["kind"] in ("silence", "noise"))
        if not _frame_contained(fr, seg["t0"], seg["t1"]):
            continue
        sp_total += 1
        if fr.get(key) is not None:
            spurious += 1
    spurious_rate = spurious / sp_total if sp_total else 0.0

    strict_stats = cent_stats(strict, None, key)
    amb_stats = cent_stats(amb, None, key)

    # ambiguity handling: fraction of ambiguous frames flagged deferred/explicit
    amb_frames = [fr for fr in result["frames"]
                  if _segment_f0(truth, fr["time"]["t_center"])[2]]
    amb_flagged = sum(1 for fr in amb_frames
                      if fr["decision"] in ("deferred", "revised")
                      or fr["octave"]["ambiguous"])
    amb_flag_rate = amb_flagged / len(amb_frames) if amb_frames else 1.0

    # revision correctness
    rev_exp = truth.get("revision_expectation", {})
    revisions = result.get("revisions", [])
    relabel_ok = None
    if rev_exp.get("region") is not None:
        r0, r1 = rev_exp["region"]
        relabel_ok = any(
            rv["kind"] == "octave_relabel"
            and rv["region_t"][0] <= r1 and rv["region_t"][1] >= r0
            for rv in revisions
        )
    else:
        # must NOT relabel the opening
        relabel_ok = not any(rv["kind"] == "octave_relabel" for rv in revisions)

    return {
        "tone_frames": len(tone),
        "voiced_frames": len(voiced),
        "coverage": coverage,
        "spurious_voiced_frames": spurious,
        "spurious_rate": spurious_rate,
        "strict_cent_stats": strict_stats,
        "ambiguous_cent_stats": amb_stats,
        "ambiguity_flag_rate": amb_flag_rate,
        "revision_expectation_met": relabel_ok,
        "n_revisions": len(revisions),
        "n_deferrals": len(result.get("deferrals", [])),
    }


def jump_delay(result: dict, truth: dict, key: str = "tracked_f0_hz") -> dict | None:
    """For take B: time between the true leap and the first frame tracking the
    new pitch within 100 cents."""
    jump = truth.get("jump")
    if not jump:
        return None
    t_jump = jump["t_s"]
    target = jump["to_f0_hz"]
    prev = None
    for fr in result["frames"]:
        t = fr["time"]["t_center"]
        if t < t_jump:
            prev = fr
            continue
        v = fr.get(key)
        if v is not None and abs(cents(v, target)) <= 100.0:
            return {
                "true_jump_s": t_jump,
                "declared_s": t,
                "delay_ms": (t - t_jump) * 1000.0,
                "from_f0_hz": jump["from_reading_f0_hz"],
                "to_f0_hz": target,
            }
    return {"true_jump_s": t_jump, "declared_s": None, "delay_ms": None,
            "to_f0_hz": target}


def evaluate_vibrato(result: dict, truth: dict,
                     key: str = "tracked_f0_hz") -> dict:
    """Evaluate vibrato by comparing to the *instantaneous* reference pitch,
    not a fixed centre.  A correct tracker follows the modulation, so tracking
    error is measured against the true modulated f0; separately we check the
    recovered modulation rate and depth.
    """
    t0 = truth["tone_t0"]
    t1 = truth["tone_t1"]
    f_center = truth["f0_center_hz"]
    v_hz = truth["vibrato_hz"]
    v_cents = truth["vibrato_cents"]
    delay = truth.get("vibrato_delay", 0.15)
    ramp_s = truth.get("vibrato_ramp_s", 0.1)

    def inst_f0(t: float) -> float:
        rel = t - t0
        if rel < delay:
            return f_center
        vm_t = rel - delay
        ramp = min(1.0, vm_t / ramp_s)
        return f_center * 2.0 ** (v_cents * ramp * math.sin(2 * math.pi * v_hz * vm_t) / 1200.0)

    frames = []
    errs = []
    for fr in result["frames"]:
        t = fr["time"]["t_center"]
        if t0 + 0.30 <= t <= t1 - 0.06:  # after ramp-in
            v = fr.get(key)
            if v is not None:
                errs.append(cents(v, inst_f0(t)))
    coverage = 0.0
    total = sum(1 for fr in result["frames"]
                if t0 + 0.30 <= fr["time"]["t_center"] <= t1 - 0.06)
    if total:
        coverage = len(errs) / total

    # verify recovered modulation from the tracked curve
    ts = []
    vs = []
    for fr in result["frames"]:
        t = fr["time"]["t_center"]
        v = fr.get(key)
        if v is not None and t0 + 0.30 <= t <= t1 - 0.06:
            ts.append(t)
            vs.append(cents(v, f_center))
    rec_hz = None
    rec_depth = None
    if len(ts) > 20:
        ts = np.array(ts)
        vs = np.array(vs)
        c = vs - np.mean(vs)
        # recovered rate via mean-removed zero crossings (robust for few cycles)
        zc = int(np.sum(np.diff(np.sign(c)) != 0))
        span = float(ts[-1] - ts[0])
        rec_hz = (zc / 2.0) / span if span > 0 else None
        # true vibrato_cents is the *peak* deviation, so recovered depth is the
        # half peak-to-peak of the cents curve.
        rec_depth = float((np.percentile(vs, 95) - np.percentile(vs, 5)) / 2.0)

    errs = np.array(errs) if errs else np.array([0.0])
    abs_errs = np.abs(errs)
    return {
        "kind": "vibrato",
        "coverage": coverage,
        "tracking_abs_mean_cents": float(np.mean(abs_errs)),
        "tracking_p95_cents": float(np.percentile(abs_errs, 95)),
        "tracking_max_cents": float(np.max(abs_errs)),
        "octave_errors": int(np.sum(abs_errs > 600)),
        "reference_vibrato_hz": v_hz,
        "recovered_vibrato_hz": rec_hz,
        "reference_vibrato_cents": v_cents,
        "recovered_vibrato_cents_p5_p95": rec_depth,
        "tone_interval": [t0, t1],
    }


def fixture_stats(result: dict, truth: dict, key: str = "tracked_f0_hz") -> dict:
    """Cent / octave metrics restricted to the fixture's tone interval."""
    if truth.get("kind") == "vibrato":
        return evaluate_vibrato(result, truth, key)
    t0 = truth.get("tone_t0", 0.0)
    t1 = truth.get("tone_t1", 1e9)
    if truth.get("kind") == "harmonic_group":
        ref = truth["f0_hz"]
        frames = []
        for fr in result["frames"]:
            t = fr["time"]["t_center"]
            if t0 + 0.04 <= t <= t1 - 0.04:
                frames.append({"frame": fr, "f0": ref})
    elif truth.get("kind") == "noisy_short":
        ref = truth["f0_hz"]
        frames = []
        for fr in result["frames"]:
            t = fr["time"]["t_center"]
            if t0 + 0.03 <= t <= t1 - 0.03:
                frames.append({"frame": fr, "f0": ref})
    else:
        return {}

    voiced = [it for it in frames if it["frame"].get(key) is not None]
    coverage = len(voiced) / len(frames) if frames else 0.0
    stats = cent_stats(frames, None, key)
    stats["coverage"] = coverage
    stats["tone_interval"] = [t0, t1]
    return stats


def evaluate_silence_noise(result: dict, truth: dict,
                           key: str = "tracked_f0_hz") -> dict:
    sil = truth["silence_regions"]
    noi = truth["noise_regions"]
    spurious = {"silence": 0, "silence_total": 0,
                "noise": 0, "noise_total": 0}
    for fr in result["frames"]:
        v = fr.get(key)
        if any(_frame_contained(fr, a, b) for a, b in sil):
            spurious["silence_total"] += 1
            if v is not None:
                spurious["silence"] += 1
        if any(_frame_contained(fr, a, b) for a, b in noi):
            spurious["noise_total"] += 1
            if v is not None:
                spurious["noise"] += 1
    spurious["silence_rate"] = (spurious["silence"] / spurious["silence_total"]
                                if spurious["silence_total"] else 0.0)
    spurious["noise_rate"] = (spurious["noise"] / spurious["noise_total"]
                              if spurious["noise_total"] else 0.0)
    return spurious
