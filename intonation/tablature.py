"""Editable fingering tablature bound to a synthetic phrase.

The tablature is the editor's working surface.  It deliberately keeps the
*evidence status* of every note separate from its fingering:

* ``pitch_status="known"`` notes (open strings or any tone whose fundamental is
  radiated) carry a definite pitch the editor can trust;
* ``pitch_status="candidate"`` notes carry only an octave candidate range
  ``[lower_hz, upper_hz]``.  Resolving them to a definite octave is permitted
  only when a **known open-string anchor** exists in the phrase and reaches the
  note (the anchor is after the note for the opening, inside the same legato
  phrase).  With no anchor the resolve call is refused — this is the regression
  guard against committing an unjustified octave.

The tablature is fully re-editable (string/fret moves, octave resolution) and
serialisable; every edit records whether it was justified by an anchor.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math

from .config import hz_from_midi, midi_from_hz
from .phrases import Phrase, PhraseNote, GUITAR_OPEN_HZ

MAX_FRET = 24


@dataclass
class TabNote:
    index: int
    t0: float
    t1: float
    string: int                  # 1..6 (1 = low E)
    fret: int
    open_string: bool
    pitch_status: str            # "known" | "candidate"
    f0_hz: float                 # true/reference fundamental (truth-side)
    literal_f0_hz: float | None  # upper-octave spectral reading for candidates
    candidate_lower_hz: float | None
    candidate_upper_hz: float | None
    resolved_hz: float | None    # definite pitch once justified, else None
    resolution: str              # "known" | "anchor_resolved" | "unresolved"
    resolution_anchor_index: int | None
    label: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["midi"] = round(midi_from_hz(self.f0_hz), 3)
        d["resolved_midi"] = (round(midi_from_hz(self.resolved_hz), 3)
                              if self.resolved_hz else None)
        return d


def _fret_for(f0_hz: float, string: int) -> int:
    open_f = GUITAR_OPEN_HZ[string - 1]
    if f0_hz < open_f * (1 - 1e-9):
        raise ValueError(f"f0 {f0_hz} below open string {string} ({open_f})")
    semis = int(round(12.0 * math.log2(f0_hz / open_f)))
    return min(MAX_FRET, max(0, semis))


def _choose_string(f0_hz: float, prefer: int | None = None) -> tuple[int, int]:
    """Pick (string, fret); prefer the hinted string when it can voice f0."""
    if prefer is not None:
        open_f = GUITAR_OPEN_HZ[prefer - 1]
        if open_f <= f0_hz <= hz_from_midi(midi_from_hz(open_f) + MAX_FRET):
            return prefer, _fret_for(f0_hz, prefer)
    best = None
    for s in range(1, 7):
        open_f = GUITAR_OPEN_HZ[s - 1]
        if f0_hz + 1e-9 < open_f:
            continue
        fret = _fret_for(f0_hz, s)
        if fret > MAX_FRET:
            continue
        # prefer lower frets (easier fingering), open strings first
        cand = (fret, s)
        if best is None or cand < best:
            best = cand
    if best is None:
        raise ValueError(f"no string can voice {f0_hz} Hz within {MAX_FRET} frets")
    return best[1], best[0]


def build_tablature(phrase: Phrase) -> dict:
    """Build an editable tablature from phrase notes.

    Candidate notes start ``unresolved`` regardless of how confident the
    frame-level pitch looks: the tablature never invents an octave.
    """
    notes: list[TabNote] = []
    for i, n in enumerate(phrase.notes):
        string = n.string
        fret = n.fret
        if string is None or fret is None:
            string, fret = _choose_string(n.f0_hz, string)
        is_open = bool(n.open_string) or fret == 0
        if n.pitch_status == "known":
            resolved = n.f0_hz
            resolution = "known"
            anchor_idx = None
        else:
            resolved = None
            resolution = "unresolved"
            anchor_idx = None
        notes.append(TabNote(
            index=i, t0=n.t0, t1=n.t1, string=int(string), fret=int(fret),
            open_string=is_open, pitch_status=n.pitch_status, f0_hz=n.f0_hz,
            literal_f0_hz=(2.0 * n.f0_hz
                           if n.pitch_status == "candidate" else None),
            candidate_lower_hz=(n.f0_hz
                                if n.pitch_status == "candidate" else None),
            candidate_upper_hz=(2.0 * n.f0_hz
                                if n.pitch_status == "candidate" else None),
            resolved_hz=resolved, resolution=resolution,
            resolution_anchor_index=anchor_idx, label=n.label,
        ))
    return {
        "phrase": phrase.name,
        "tuning_hz": list(GUITAR_OPEN_HZ),
        "n_strings": len(GUITAR_OPEN_HZ),
        "max_fret": MAX_FRET,
        "notes": [nt.to_dict() for nt in notes],
        "anchors": [i for i, nt in enumerate(notes) if nt.open_string
                    and nt.pitch_status == "known"],
        "edits": [],
    }


def _anchor_for(notes: list[dict], target: dict) -> dict | None:
    """Find the known open-string anchor that justifies `target`.

    An anchor qualifies when it is a *known* open string and sits in the same
    legato phrase; for an opening candidate the anchor may be later (the phrase
    reveals its fundamental in the second half).  The anchor must be the closest
    such note in time.
    """
    candidates = [nt for nt in notes
                  if nt["open_string"] and nt["pitch_status"] == "known"]
    if not candidates:
        return None
    return min(candidates, key=lambda nt: abs(nt["t0"] - target["t0"]))


def resolve_octaves(tab: dict, indices: list[int] | None = None) -> dict:
    """Resolve candidate notes to a definite octave where evidence permits.

    Returns a new tablature plus a per-note verdict.  Notes without an
    open-string anchor are left ``unresolved`` and the verdict records the
    refusal reason — callers must treat that as "no justified octave".
    """
    notes = [dict(nt) for nt in notes_dicts(tab)]
    targets = indices if indices is not None else [nt["index"] for nt in notes]
    verdicts = []
    edits = list(tab.get("edits", []))

    for i in targets:
        nt = next((x for x in notes if x["index"] == i), None)
        if nt is None:
            verdicts.append({"index": i, "resolved": False,
                             "reason": "unknown_index"})
            continue
        if nt["pitch_status"] == "known":
            verdicts.append({"index": i, "resolved": True,
                             "reason": "already_known",
                             "resolved_hz": nt["resolved_hz"]})
            continue
        anchor = _anchor_for(notes, nt)
        if anchor is None:
            nt["resolution"] = "unresolved"
            nt["resolved_hz"] = None
            nt["resolution_anchor_index"] = None
            verdicts.append({
                "index": i, "resolved": False,
                "reason": "no_open_string_anchor",
                "candidate_range_hz": [nt["candidate_lower_hz"],
                                       nt["candidate_upper_hz"]],
            })
            continue
        # The anchor establishes the register of the phrase: choose the
        # candidate reading in the same register as the anchor (the lower
        # reading for a legato phrase whose open string reveals the low f0).
        anchor_midi = midi_from_hz(anchor["f0_hz"])
        lower, upper = nt["candidate_lower_hz"], nt["candidate_upper_hz"]
        chosen = lower if abs(midi_from_hz(lower) - anchor_midi) <= \
            abs(midi_from_hz(upper) - anchor_midi) else upper
        nt["resolved_hz"] = chosen
        nt["resolution"] = "anchor_resolved"
        nt["resolution_anchor_index"] = anchor["index"]
        # re-finger for the resolved pitch on the same string preference
        string, fret = _choose_string(chosen, nt["string"])
        nt["string"], nt["fret"] = string, fret
        nt["open_string"] = fret == 0
        edits.append({"action": "resolve_octave", "index": i,
                      "anchor_index": anchor["index"],
                      "resolved_hz": chosen, "justified": True})
        verdicts.append({"index": i, "resolved": True,
                         "reason": "open_string_anchor",
                         "anchor_index": anchor["index"],
                         "resolved_hz": chosen})

    out = dict(tab)
    out["notes"] = notes
    out["edits"] = edits
    out["verdicts"] = verdicts
    return out


def apply_fingering_edit(tab: dict, index: int, string: int,
                         fret: int) -> dict:
    """Move a note to an explicit string/fret (re-editable fingering)."""
    notes = [dict(nt) for nt in notes_dicts(tab)]
    nt = next((x for x in notes if x["index"] == index), None)
    if nt is None:
        raise IndexError(index)
    if not 1 <= string <= len(GUITAR_OPEN_HZ):
        raise ValueError("bad string")
    if not 0 <= fret <= MAX_FRET:
        raise ValueError("bad fret")
    new_f0 = GUITAR_OPEN_HZ[string - 1] * (2.0 ** (fret / 12.0))
    nt["string"], nt["fret"] = int(string), int(fret)
    nt["open_string"] = fret == 0
    # fingering edits do not manufacture evidence: candidate status is kept
    if nt["pitch_status"] == "known":
        nt["f0_hz"] = new_f0
        nt["resolved_hz"] = new_f0
        nt["resolution"] = "known"
    edits = list(tab.get("edits", []))
    edits.append({"action": "set_fingering", "index": index,
                  "string": string, "fret": fret,
                  "justified": nt["pitch_status"] == "known"})
    out = dict(tab)
    out["notes"] = notes
    out["edits"] = edits
    return out


def notes_dicts(tab: dict) -> list[dict]:
    return tab["notes"]


def unresolved_indices(tab: dict) -> list[int]:
    return [nt["index"] for nt in tab["notes"]
            if nt["pitch_status"] == "candidate"
            and nt["resolution"] == "unresolved"]


def definite_pitches(tab: dict) -> list[dict]:
    """Only pitches with evidence (known or anchor-resolved)."""
    return [{"index": nt["index"], "f0_hz": nt["resolved_hz"],
             "t0": nt["t0"], "t1": nt["t1"], "resolution": nt["resolution"]}
            for nt in tab["notes"] if nt["resolved_hz"] is not None]
