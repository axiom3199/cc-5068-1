"""Follow-along accompaniment synthesised from an intonation analysis result.

The synthesizer is driven *only* by pitch the kernel has committed: offline
tracks use the voiced notes, the streaming path uses frames whose admission
gate opened.  Deferred, gated or unreliable regions stay silent, so the
audible result is itself an inspectable readout of the admission decision:
you can hear exactly where the kernel was unwilling to play along.
"""

from __future__ import annotations

import math
import numpy as np

from .config import cents

# Soft timbres: the accompaniment should be audible without masking the note.
FOLLOW_HARMONICS = {1: 0.50, 2: 0.27, 3: 0.12}
BASS_HARMONICS = {1: 0.55, 2: 0.18}


def _runs_from_notes(notes: list[dict]) -> list[dict]:
    spans = []
    for n in notes:
        f0 = n.get("median_f0_hz")
        if f0 is None or n.get("state") != "voiced":
            continue
        spans.append({
            "t0": float(n["t_start"]), "t1": float(n["t_end"]),
            "f0": float(f0), "source": "note",
            "confidence": float(n.get("confidence", 0.0)),
        })
    return spans


def _runs_from_stream(frames: list[dict], gap_s: float = 0.05,
                      tol_cents: float = 60.0) -> list[dict]:
    spans: list[dict] = []
    cur: dict | None = None
    for f in frames:
        if not f.get("admitted") or not f.get("f0_hz"):
            if cur is not None:
                spans.append(cur)
                cur = None
            continue
        f0 = float(f["f0_hz"])
        t0, t1 = float(f["t_start"]), float(f["t_end"])
        if (cur is not None and abs(cents(f0, cur["f0"])) <= tol_cents
                and t0 - cur["t1"] <= gap_s):
            cur["t1"] = t1
            cur["vals"].append(f0)
            cur["confs"].append(float(f.get("confidence", 0.0)))
        else:
            if cur is not None:
                spans.append(cur)
            cur = {"t0": t0, "t1": t1, "f0": f0, "source": "stream",
                   "vals": [f0], "confs": [float(f.get("confidence", 0.0))]}
    if cur is not None:
        spans.append(cur)
    out = []
    for s in spans:
        out.append({
            "t0": s["t0"], "t1": s["t1"],
            "f0": float(np.median(s["vals"])) if s["vals"] else s["f0"],
            "source": s["source"],
            "confidence": float(np.mean(s["confs"])) if s["confs"] else 0.0,
        })
    return out


def admitted_spans(result: dict) -> list[dict]:
    """Pitch spans the kernel committed, from either analysis mode."""
    if result.get("mode") == "streaming" or "notes" not in result:
        return _runs_from_stream(result.get("frames", []))
    return _runs_from_notes(result["notes"])


def gated_spans(result: dict) -> list[dict]:
    """Regions the kernel refused to commit, with the reason, for display."""
    if result.get("mode") == "streaming" or "notes" not in result:
        out: list[dict] = []
        cur = None
        for f in result.get("frames", []):
            if f.get("admitted"):
                if cur is not None:
                    out.append(cur)
                    cur = None
                continue
            reason = f.get("admission_reason", "gated")
            if cur is not None and f.get("t_start", 0.0) - cur["t1"] <= 0.05:
                cur["t1"] = float(f.get("t_end", cur["t1"]))
                cur["reasons"].append(reason)
            else:
                if cur is not None:
                    out.append(cur)
                cur = {"t0": float(f.get("t_start", 0.0)),
                       "t1": float(f.get("t_end", 0.0)),
                       "reasons": [reason]}
        if cur is not None:
            out.append(cur)
        return [{"t0": s["t0"], "t1": s["t1"],
                 "reason": max(set(s["reasons"]), key=s["reasons"].count)}
                for s in out]

    out = []
    for n in result["notes"]:
        if n.get("median_f0_hz") is None and n.get("state") != "voiced":
            out.append({"t0": float(n["t_start"]), "t1": float(n["t_end"]),
                        "reason": n.get("state", "unvoiced")})
    return out


def _env(n: int, sr: int, attack: float = 0.03, release: float = 0.05) -> np.ndarray:
    e = np.ones(n)
    na = min(n, max(1, int(attack * sr)))
    nr = min(n, max(1, int(release * sr)))
    e[:na] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(na) / na)
    e[-nr:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(nr) / nr)
    return e


def _add_tone(out: np.ndarray, sr: int, t0: float, t1: float, f0: float,
              harmonics: dict[int, float], gain: float) -> None:
    n = len(out)
    i0 = max(0, int(round(t0 * sr)))
    i1 = min(n, int(round(t1 * sr)))
    if i1 <= i0:
        return
    nn = i1 - i0
    phase = 2.0 * np.pi * f0 * np.arange(nn) / sr
    seg = np.zeros(nn)
    for k, a in harmonics.items():
        if k * f0 >= sr / 2.0:
            continue
        seg += a * np.sin(k * phase + 0.13 * k)
    seg *= _env(nn, sr)
    out[i0:i1] += gain * seg


def render_spans(spans: list[dict], duration: float, sr: int,
                 gain_follow: float = 0.50,
                 gain_bass: float = 0.22) -> np.ndarray:
    """Render one following voice plus a soft sub-octave bass drone."""
    n = int(round(duration * sr))
    out = np.zeros(n)
    for s in spans:
        f0 = float(s["f0"])
        _add_tone(out, sr, s["t0"], s["t1"], f0, FOLLOW_HARMONICS, gain_follow)
        _add_tone(out, sr, s["t0"], s["t1"], f0 / 2.0, BASS_HARMONICS, gain_bass)
    peak = float(np.max(np.abs(out))) if n else 0.0
    if peak > 0:
        out = out / peak * 0.85
    return out


def spans_from_score(score: dict) -> list[dict]:
    spans = []
    for note in score.get("notes", []):
        if note.get("kind") != "note" or note.get("f0_hz") is None:
            continue
        spans.append({"t0": note["t_start"], "t1": note["t_end"],
                      "f0": float(note["f0_hz"]), "source": "score",
                      "confidence": float(note.get("confidence", 0.0))})
    return spans


def render_accompaniment(result: dict, sr: int, duration: float | None = None,
                         follow_score: dict | None = None) -> dict:
    """Return rendered audio plus the spans and gated regions that produced it."""
    if duration is None:
        duration = float(result.get("audio", {}).get("duration_s") or 0.0)
        if duration <= 0 and result.get("frames"):
            duration = float(result["frames"][-1].get("t_end", 0.0))
    spans = spans_from_score(follow_score) if follow_score else admitted_spans(result)
    audio = render_spans(spans, duration, sr)
    return {
        "audio": audio,
        "sample_rate": sr,
        "duration_s": duration,
        "spans": spans,
        "gated": gated_spans(result),
        "n_spans": len(spans),
        "note": "accompaniment is gated by the same admission evidence as the track",
    }
