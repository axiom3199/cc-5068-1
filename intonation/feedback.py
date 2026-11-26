"""Failure feedback: turn human score edits back into kernel critique.

The UI is driven by the kernel (curve, candidates, octave ambiguity, admission
gate).  When a musician overrides a kernel decision on the staff, that override
is not merely cosmetic: it is evidence the kernel was wrong.  This module
classifies each edit against the kernel's original decision, attaches the
kernel evidence that was available at that moment, and emits a feedback record.

Records are appended to ``data/feedback/<case>.json`` so they can be used to
tune thresholds/config without losing the audit trail.
"""

from __future__ import annotations

import json
import math
import os

from .config import cents


def _frames_in(result: dict, t0: float, t1: float) -> list[dict]:
    out = []
    for fr in result.get("frames", []):
        tc = fr.get("t_center", fr.get("time", {}).get("t_center"))
        if tc is None:
            continue
        if t0 <= tc <= t1:
            out.append(fr)
    return out


def classify_edits(original_score: dict, edited_score: dict, result: dict) -> list[dict]:
    """Compare edited score to the kernel's original decision note by note."""
    orig = original_score["notes"]
    edit_actions = edited_score.get("edits", [])
    by_index: dict[int, list[dict]] = {}
    for ed in edit_actions:
        by_index.setdefault(int(ed.get("index", -1)), []).append(ed)

    records: list[dict] = []
    for i in sorted(by_index):
        if i < 0 or i >= len(orig) or i >= len(edited_score["notes"]):
            continue
        o = orig[i]
        if o.get("kind") != "note":
            continue
        note = edited_score["notes"][i]
        t0, t1 = note["t_start"], note["t_end"]
        frames = _frames_in(result, t0, t1)
        ambiguous = any(
            f.get("octave", {}).get("ambiguous")
            or f.get("ambiguous") for f in frames
        )
        admission = sorted({f.get("admission_reason") for f in frames
                            if f.get("admission_reason")})

        if note.get("kind") == "deleted":
            records.append({
                "kind": "spurious_note",
                "kernel_midi": round(o["midi"], 2),
                "t_range": [round(t0, 3), round(t1, 3)],
                "evidence": {"ambiguous_octave": ambiguous,
                              "admission_reasons": admission,
                              "kernel_confidence": round(o.get("confidence", 0), 3)},
                "suggestion": "raise confidence_min or require more sustain here",
            })
            continue

        delta = cents(note["f0_hz"], o["f0_hz"])
        if abs(delta) >= 600:
            d_oct = round(delta / 1200.0)
            records.append({
                "kind": "octave_failure",
                "kernel_midi": round(o["midi"], 2),
                "user_midi": round(note["midi"], 2),
                "octaves_corrected": d_oct,
                "t_range": [round(t0, 3), round(t1, 3)],
                "evidence": {
                    "ambiguous_octave": ambiguous,
                    "admission_reasons": admission,
                    "kernel_octave_corrected": bool(o.get("octave_corrected")),
                },
                "suggestion": ("kernel kept the wrong octave; strengthen the "
                               "temporal/attack evidence or the sub-octave score margin"),
            })
        elif abs(delta) >= 20:
            records.append({
                "kind": "pitch_error",
                "kernel_midi": round(o["midi"], 2),
                "user_midi": round(note["midi"], 2),
                "delta_cents": round(delta, 2),
                "t_range": [round(t0, 3), round(t1, 3)],
                "evidence": {"ambiguous_octave": ambiguous,
                              "admission_reasons": admission},
                "suggestion": "refine f0 weighting or peak tolerance near this frame",
            })
    return records


def append_feedback(path: str, case: str, records: list[dict],
                    edited_score: dict) -> dict:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        try:
            with open(path) as f:
                log = json.load(f)
        except Exception:
            log = {"case": case, "entries": []}
    else:
        log = {"case": case, "entries": []}
    entry = {
        "n_edits": len(edited_score.get("edits", [])),
        "records": records,
        "score_note_count": edited_score.get("n_notes"),
    }
    log["entries"].append(entry)
    log["last"] = entry
    with open(path, "w") as f:
        json.dump(log, f, indent=1)
    return entry
