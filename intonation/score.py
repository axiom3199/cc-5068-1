"""Editable score derived from the intonation track.

The score is the layer a musician can correct.  It is built from the kernel's
committed notes (including the offline octave revision), laid out on a beat
grid, exported to MusicXML / MIDI, and *re-editable*: the browser sends edits
(octave shifts, pitch corrections, deletions) back to the server, which applies
them, re-renders the accompaniment and re-exports.  Each edit the user makes is
also recorded as failure feedback for the kernel, so a wrong kernel decision is
visible rather than smoothed away.
"""

from __future__ import annotations

import math
import os
import struct

# MIDI pitch -> (step, alter, octave).  Sharps are used for black keys.
_STEP = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"]
_ALTER = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]


def midi_to_name(midi: float) -> str:
    m = int(round(midi))
    return f"{_STEP[m % 12]}{'#' if _ALTER[m % 12] else ''}{m // 12 - 1}"


def midi_to_musicxml_pitch(midi: float) -> dict:
    m = int(round(midi))
    return {"step": _STEP[m % 12], "alter": _ALTER[m % 12], "octave": m // 12 - 1}


def hz_to_midi(f0: float) -> float:
    return 69.0 + 12.0 * math.log2(f0 / 440.0)


def _quantize_beats(x: float, grid: float) -> float:
    return round(x / grid) * grid


def build_score(result: dict, tempo_bpm: float = 100.0,
                beats_per_measure: int = 4, grid: float = 0.25) -> dict:
    """Turn committed notes into a beat-aligned, re-editable score.

    ``grid`` is the rhythmic quantization in beats (0.25 = a sixteenth note).
    Pitch is never quantized here; only timing is snapped, so a cent-level
    pitch track stays visible on the staff as its true pitch.
    """
    sec_per_beat = 60.0 / tempo_bpm
    raw = []
    for i, n in enumerate(result.get("notes", [])):
        f0 = n.get("median_f0_hz")
        if f0 is None or n.get("state") != "voiced":
            raw.append({"kind": "rest", "t_start": n["t_start"],
                        "t_end": n["t_end"], "f0_hz": None,
                        "kernel_index": i})
            continue
        raw.append({
            "kind": "note", "t_start": n["t_start"], "t_end": n["t_end"],
            "f0_hz": float(f0), "confidence": float(n.get("confidence", 0.0)),
            "midi": hz_to_midi(f0),
            "octave_corrected": bool(n.get("octave_corrected")),
            "revision": n.get("revision"),
            "kernel_index": i,
        })

    notes = []
    for item in raw:
        beat_start = _quantize_beats(item["t_start"] / sec_per_beat, grid)
        beat_end = max(beat_start + grid,
                       _quantize_beats(item["t_end"] / sec_per_beat, grid))
        base = {
            "beat_start": beat_start,
            "beats": beat_end - beat_start,
            "t_start": item["t_start"], "t_end": item["t_end"],
            "kind": item["kind"],
            "kernel_index": item["kernel_index"],
            "edited": False,
        }
        if item["kind"] == "note":
            base.update({
                "f0_hz": item["f0_hz"],
                "midi": item["midi"],
                "pitch_name": midi_to_name(item["midi"]),
                "confidence": item["confidence"],
                "octave_corrected": item["octave_corrected"],
                "revision": item["revision"],
                "source": "kernel",
            })
        notes.append(base)

    # merge adjacent rests for a cleaner staff
    merged = []
    for nt in notes:
        if (merged and nt["kind"] == "rest" and merged[-1]["kind"] == "rest"
                and nt["beat_start"] <= merged[-1]["beat_start"] + merged[-1]["beats"]):
            prev = merged[-1]
            prev["beats"] = (nt["beat_start"] + nt["beats"]) - prev["beat_start"]
            prev["t_end"] = nt["t_end"]
        else:
            merged.append(nt)

    return {
        "tempo_bpm": tempo_bpm,
        "beats_per_measure": beats_per_measure,
        "grid_beats": grid,
        "sec_per_beat": sec_per_beat,
        "notes": merged,
        "n_notes": sum(1 for n in merged if n["kind"] == "note"),
        "n_rests": sum(1 for n in merged if n["kind"] == "rest"),
        "revisions": result.get("revisions", []),
    }


def apply_edits(score: dict, edits: list[dict]) -> dict:
    """Apply user edits in place-ish (returns a new score dict).

    Supported actions:
      {"action": "octave", "index": k, "dir": 1|-1}
      {"action": "set_f0", "index": k, "f0_hz": f}
      {"action": "set_midi", "index": k, "midi": m}
      {"action": "delete", "index": k}
    Edits are recorded under ``score["edits"]`` for feedback to the kernel.
    """
    notes = [dict(n) for n in score["notes"]]
    applied = []
    for ed in edits:
        k = int(ed.get("index", -1))
        if k < 0 or k >= len(notes):
            continue
        nt = notes[k]
        action = ed.get("action")
        if action == "delete":
            nt["kind"] = "deleted"
            nt["edited"] = True
            applied.append({**ed, "applied": True})
        elif nt["kind"] != "note":
            applied.append({**ed, "applied": False, "reason": "not_a_note"})
            continue
        elif action == "octave":
            d = int(ed.get("dir", 0))
            nt["midi"] = nt["midi"] + 12 * d
            nt["f0_hz"] = nt["f0_hz"] * (2.0 ** d)
            nt["octave_corrected"] = True
            nt["edited"] = True
            applied.append({**ed, "applied": True})
        elif action == "set_f0":
            nt["f0_hz"] = float(ed["f0_hz"])
            nt["midi"] = hz_to_midi(nt["f0_hz"])
            nt["edited"] = True
            applied.append({**ed, "applied": True})
        elif action == "set_midi":
            nt["midi"] = float(ed["midi"])
            nt["f0_hz"] = 440.0 * 2.0 ** ((nt["midi"] - 69.0) / 12.0)
            nt["edited"] = True
            applied.append({**ed, "applied": True})
        else:
            applied.append({**ed, "applied": False, "reason": "unknown_action"})
        if nt.get("kind") == "note":
            nt["pitch_name"] = midi_to_name(nt["midi"])
            nt["source"] = "edited"

    out = dict(score)
    out["notes"] = notes
    out["n_notes"] = sum(1 for n in notes if n["kind"] == "note")
    out["edits"] = score.get("edits", []) + applied
    return out


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def to_musicxml(score: dict) -> str:
    divisions = 480
    bpm = score.get("tempo_bpm", 100.0)
    bpm_measure = score.get("beats_per_measure", 4)

    # lay notes into measures by beat position
    notes = [n for n in score["notes"] if n["kind"] != "deleted"]
    total_beats = max((n["beat_start"] + n["beats"] for n in notes), default=0.0)
    n_measures = max(1, int(math.ceil(total_beats / bpm_measure)))

    measures: list[list[str]] = [[] for _ in range(n_measures)]
    for n in notes:
        m = min(n_measures - 1, int(n["beat_start"] // bpm_measure))
        dur = int(round(n["beats"] * divisions))
        if dur <= 0:
            dur = divisions // 4
        if n["kind"] == "rest":
            measures[m].append(
                f'<note><rest/><duration>{dur}</duration>'
                f'<voice>1</voice><type>quarter</type></note>')
        else:
            p = midi_to_musicxml_pitch(n["midi"])
            measures[m].append(
                f'<note><pitch><step>{p["step"]}</step>'
                f'<alter>{p["alter"]}</alter><octave>{p["octave"]}</octave></pitch>'
                f'<duration>{dur}</duration><voice>1</voice>'
                f'<type>{_type_name(n["beats"])}</type></note>')
        if n.get("edited") or n.get("octave_corrected"):
            measures[m].append(
                '<direction><direction-type><words>revised</words>'
                '</direction-type></direction>')

    body = []
    for i, entries in enumerate(measures):
        num = i + 1
        if i == 0:
            attrs = (f'<attributes><divisions>{divisions}</divisions>'
                     '<key><fifths>0</fifths></key>'
                     f'<time><beats>{bpm_measure}</beats><beat-type>4</beat-type></time>'
                     '<clef><sign>G</sign><line>2</line></clef></attributes>')
            tempo = ('<direction><direction-type><metronome>'
                     f'<beat-unit>quarter</beat-unit><per-minute>{bpm:g}</per-minute>'
                     '</metronome></direction-type></direction>')
        else:
            attrs, tempo = "", ""
        body.append(f'<measure number="{num}">{attrs}{tempo}{"".join(entries)}</measure>')

    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 '
            'Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">\n'
            '<score-partwise version="3.1">'
            '<part-list><score-part id="P1"><part-name>Intonation</part-name>'
            '</score-part></part-list>'
            '<part id="P1">' + "".join(body) + '</part></score-partwise>\n')


def _type_name(beats: float) -> str:
    table = {0.25: "16th", 0.5: "eighth", 0.75: "eighth.",
             1.0: "quarter", 1.5: "quarter.", 2.0: "half",
             3.0: "half.", 4.0: "whole"}
    return table.get(round(beats, 3), "quarter")


def to_midi_bytes(score: dict) -> bytes:
    """Standard MIDI file (format 0) with tempo and note on/off events."""
    tpb = 480
    bpm = score.get("tempo_bpm", 100.0)
    events: list[tuple[int, int, bytes]] = []  # (tick, order, data)

    us_per_quarter = int(round(1e6 * 60.0 / bpm))
    events.append((0, 0, b"\xff\x51\x03" + us_per_quarter.to_bytes(3, "big")))

    for n in score["notes"]:
        if n["kind"] != "note":
            continue
        on = int(round(n["beat_start"] * tpb))
        off = int(round((n["beat_start"] + n["beats"]) * tpb))
        key = max(0, min(127, int(round(n["midi"]))))
        events.append((on, 1, b"\x90" + bytes([key, 80])))
        events.append((off, 2, b"\x80" + bytes([key, 0])))

    events.sort(key=lambda e: (e[0], e[1]))

    track = bytearray()
    prev = 0
    for tick, _, data in events:
        delta = tick - prev
        prev = tick
        track += _vlq(delta) + data
    track += _vlq(0) + b"\xff\x2f\x00"

    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, tpb)
    trk = b"MTrk" + struct.pack(">I", len(track)) + bytes(track)
    return header + trk


def _vlq(value: int) -> bytes:
    value = int(max(0, value))
    out = bytearray([value & 0x7F])
    value >>= 7
    while value:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(out))


def write_exports(score: dict, outdir: str, name: str = "score") -> dict:
    os.makedirs(outdir, exist_ok=True)
    xml_path = os.path.join(outdir, f"{name}.musicxml")
    midi_path = os.path.join(outdir, f"{name}.mid")
    with open(xml_path, "w") as f:
        f.write(to_musicxml(score))
    with open(midi_path, "wb") as f:
        f.write(to_midi_bytes(score))
    return {"musicxml": xml_path, "midi": midi_path}
