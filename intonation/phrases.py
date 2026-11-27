"""Synthetic *phrases* with known f0, vibrato and voicing changes, bound to an
editable fingering tablature.

This layer sits on top of :mod:`intonation.synth` and adds three things the
plain fixtures do not have:

1. **Editable phrase parameters** — every note is a plain serialisable dataclass
   (pitch, timbre, vibrato, noise, articulation), so a phrase can be rebuilt bit
   for bit from a saved parameter file.
2. **Explicit pitch status** — each note records whether its pitch is a *known*
   pitch (``pitch_status="known"``, e.g. an open string whose fundamental is
   radiated) or only a *candidate range* (``pitch_status="candidate"``, an
   even-harmonic / missing-fundamental texture whose octave cannot be settled
   from its own spectrum).  The status is propagated into the ground truth and
   the tablature; nothing downstream is allowed to promote a candidate range to
   a definite octave without evidence.
3. **Tablature binding** — :mod:`intonation.tablature` turns the phrase into
   string/fret positions; the phrase keeps the open-string anchor an editor
   needs to resolve an otherwise undecidable octave.

Ground truth (including note names and the known/candidate status) is always
written to a *separate* JSON file; audio file names never encode pitch.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
import numpy as np

from . import synth
from .config import hz_from_midi, midi_from_hz

SR = synth.SR

# ---------------------------------------------------------------------------
# Timbre presets.  "even" textures radiate only even harmonics of the note f0,
# so per frame they are spectrally identical to a complete tone an octave up.
# ---------------------------------------------------------------------------
TIMBRES: dict[str, dict[int, float]] = {
    # full spectrum, fundamental clearly radiated -> known pitch
    "full": {1: 0.50, 2: 0.42, 3: 0.30, 4: 0.22, 5: 0.14, 6: 0.10},
    # open-string-like: strong fundamental plus a bright even series
    "open_string": {1: 0.62, 2: 0.40, 3: 0.26, 4: 0.30, 5: 0.16, 6: 0.18},
    # dominant second harmonic (max-peak would read an octave high), f0 present
    "dom2": {1: 0.12, 2: 0.60, 3: 0.12, 4: 0.26, 5: 0.08},
    # even harmonics only -> missing fundamental, octave genuinely undecidable
    "even": {2: 0.55, 4: 0.42, 6: 0.30, 8: 0.17, 10: 0.10},
    # breathy: weak, sparse partials over broadband aspiration noise
    "breathy": {1: 0.16, 2: 0.10, 3: 0.06},
}

# timbres whose own spectrum cannot determine the octave (missing fundamental)
CANDIDATE_TIMBRES = frozenset({"even"})


@dataclass
class PhraseNote:
    """One editable note of a phrase.

    ``f0_hz`` is the *true* fundamental (used only for synthesis and ground
    truth, never fed to the analyzer).  ``pitch_status`` says whether that
    pitch is directly evidenced in the sound or merely a candidate range.
    """

    f0_hz: float
    t0: float
    t1: float
    timbre: str = "full"
    vibrato_hz: float = 0.0
    vibrato_cents: float = 0.0
    vibrato_delay: float = 0.15
    noise_amp: float = 0.006
    attack: float = 0.03
    release: float = 0.05
    pitch_status: str = "known"          # "known" | "candidate"
    label: str = ""
    # fingering hint for the tablature binding; None = let the tab choose
    string: int | None = None
    fret: int | None = None
    open_string: bool = False

    @property
    def midi(self) -> float:
        return midi_from_hz(self.f0_hz)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["midi"] = round(self.midi, 3)
        d["harmonics"] = dict(TIMBRES[self.timbre])
        return d


@dataclass
class Phrase:
    name: str
    notes: list[PhraseNote]
    duration: float
    sample_rate: int = SR
    # unvoiced gaps [t0, t1, kind] where no pitch must be reported
    unvoiced: list[list] = field(default_factory=list)
    description: str = ""
    seed: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "description": self.description,
            "seed": self.seed,
            "unvoiced": [list(g) for g in self.unvoiced],
            "notes": [n.to_dict() for n in self.notes],
        }


def _status_for(timbre: str) -> str:
    return "candidate" if timbre in CANDIDATE_TIMBRES else "known"


def render_phrase(phrase: Phrase) -> np.ndarray:
    """Render phrase notes (additive synth) plus unvoiced noise regions."""
    segments = []
    for n in phrase.notes:
        segments.append(synth.Segment(
            t0=n.t0, t1=n.t1, f0=n.f0_hz, harmonics=dict(TIMBRES[n.timbre]),
            vibrato_hz=n.vibrato_hz, vibrato_cents=n.vibrato_cents,
            vibrato_delay=n.vibrato_delay, noise_amp=n.noise_amp,
            attack=n.attack, release=n.release,
        ))
    audio, _ = synth.render_segments(segments, phrase.duration,
                                     sr=phrase.sample_rate, seed=phrase.seed)

    # add broadband aspiration into declared unvoiced gaps (no periodic partials)
    if phrase.unvoiced:
        rng = np.random.default_rng(phrase.seed + 991)
        n_total = int(round(phrase.duration * phrase.sample_rate))
        for g in phrase.unvoiced:
            t0, t1, kind = g[0], g[1], g[2]
            i0 = int(round(t0 * phrase.sample_rate))
            i1 = min(n_total, int(round(t1 * phrase.sample_rate)))
            if i1 <= i0:
                continue
            level = 0.10 if kind == "breath" else 0.05
            # shaped noise burst with soft edges
            burst = rng.standard_normal(i1 - i0) * level
            na = max(1, int(0.02 * phrase.sample_rate))
            env = np.ones(len(burst))
            env[:na] = np.linspace(0.0, 1.0, na)
            env[-na:] = np.linspace(1.0, 0.0, na)
            audio[i0:i1] += burst * env
    return synth.normalize_peak(audio, 0.8)


def phrase_truth(phrase: Phrase) -> dict:
    """Ground-truth segments + instantaneous f0 trajectory (evaluation only).

    The truth file keeps, per note, whether the pitch is *known* or a
    *candidate range*, and for candidate notes the literal (upper-octave)
    spectral reading so acceptance code can check the analyzer stayed
    undecided rather than inventing an octave.
    """
    segments = []
    for n in phrase.notes:
        seg = {
            "t0": n.t0, "t1": n.t1, "kind": "tone",
            "f0": n.f0_hz,
            "midi": round(n.midi, 3),
            "pitch_status": n.pitch_status,
            "timbre": n.timbre,
            "label": n.label,
            "open_string": n.open_string,
            "ambiguous": n.pitch_status == "candidate",
        }
        if n.pitch_status == "candidate":
            seg["spectral_reading_f0"] = 2.0 * n.f0_hz
            seg["candidate_range_hz"] = [n.f0_hz, 2.0 * n.f0_hz]
        if n.vibrato_hz > 0:
            seg["vibrato_hz"] = n.vibrato_hz
            seg["vibrato_cents"] = n.vibrato_cents
            seg["vibrato_delay"] = n.vibrato_delay
        segments.append(seg)
    for g in phrase.unvoiced:
        segments.append({"t0": g[0], "t1": g[1], "kind": g[2], "f0": None,
                         "label": "unvoiced gap"})
    segments.sort(key=lambda s: s["t0"])
    return {
        "phrase": phrase.name,
        "sample_rate": phrase.sample_rate,
        "duration": phrase.duration,
        "description": phrase.description,
        "pitch_status_legend": {
            "known": "f0 directly evidenced by the spectrum (e.g. open string)",
            "candidate": "missing-fundamental texture; only an octave candidate "
                         "range is justified, no definite octave without evidence",
        },
        "segments": segments,
        "f0_trajectory": _f0_trajectory(phrase),
        "open_string_anchors": [
            {"f0_hz": n.f0_hz, "t0": n.t0, "t1": n.t1, "label": n.label}
            for n in phrase.notes if n.open_string
        ],
    }


def _f0_trajectory(phrase: Phrase, hop_s: float = 0.01) -> list[dict]:
    """Dense instantaneous-f0 reference at a fixed 10 ms grid (truth only)."""
    out = []
    t = 0.0
    while t < phrase.duration:
        for n in phrase.notes:
            if n.t0 <= t < n.t1:
                rel = t - n.t0
                cents_mod = 0.0
                if n.vibrato_hz > 0 and rel >= n.vibrato_delay:
                    vm_t = rel - n.vibrato_delay
                    ramp = min(1.0, vm_t / 0.1)
                    cents_mod = n.vibrato_cents * ramp * np.sin(
                        2.0 * np.pi * n.vibrato_hz * vm_t)
                inst = n.f0_hz * 2.0 ** (cents_mod / 1200.0)
                out.append({"t": round(t, 4), "f0_hz": float(inst),
                            "voiced": True,
                            "pitch_status": n.pitch_status})
                break
        else:
            out.append({"t": round(t, 4), "f0_hz": None, "voiced": False,
                        "pitch_status": None})
        t += hop_s
    return out


# ---------------------------------------------------------------------------
# Anchor phrase: first half has no fundamental (two octave readings fit); the
# second half opens a known open string, giving the editor the evidence to
# disambiguate.  Removing the open-string anchor (see build_no_anchor_variant)
# must leave every candidate note unresolvable.
# ---------------------------------------------------------------------------

# Standard guitar open strings used as the reference instrument.
GUITAR_OPEN_HZ = [82.41, 110.0, 146.83, 196.0, 246.94, 329.63]  # E2..E4
GUITAR_OPEN_NAMES = ["E2", "A2", "D3", "G3", "B3", "E4"]


def build_anchor_phrase(name: str | None = None,
                        with_anchor: bool = True) -> Phrase:
    """Four-note legato phrase.

    N1 (G3, even-only) and N4 (F3, even-only) are missing-fundamental tones:
    both octave readings fit their spectra.  N2 (A3, dominant-2nd) keeps a weak
    fundamental so its own pitch is known but the timbre is adversarial.  N3 is
    a known **open G string** (when ``with_anchor``); its radiated fundamental
    is the evidence that lets an editor resolve the candidate octaves.  When the
    anchor is removed, N3 becomes even-only too and nothing justifies a definite
    octave anywhere.
    """
    if name is None:
        name = "phrase_anchor" if with_anchor else "phrase_anchor_no_anchor"
    f_g3 = hz_from_midi(55)      # 196.00
    f_a3 = hz_from_midi(57)      # 220.00
    f_f3 = hz_from_midi(53)      # 174.61

    lead = 0.20
    notes: list[PhraseNote] = []

    # N1: candidate (missing fundamental), legato opening
    n1_t0, n1_t1 = lead, 1.10
    notes.append(PhraseNote(
        f0_hz=f_g3, t0=n1_t0, t1=n1_t1, timbre="even",
        vibrato_hz=5.0, vibrato_cents=20.0,
        pitch_status="candidate", label="N1 even-only G3 (octave undecidable)",
        string=3, fret=0))

    # N2: known pitch, adversarial dominant-2nd timbre
    n2_t0, n2_t1 = 1.16, 2.00
    notes.append(PhraseNote(
        f0_hz=f_a3, t0=n2_t0, t1=n2_t1, timbre="dom2",
        vibrato_hz=5.5, vibrato_cents=18.0,
        pitch_status="known", label="N2 dominant-2nd A3 (weak fundamental)",
        string=2, fret=7))

    # N3: the open-string anchor (or, without it, another candidate texture)
    n3_t0, n3_t1 = 2.06, 2.95
    if with_anchor:
        notes.append(PhraseNote(
            f0_hz=f_g3, t0=n3_t0, t1=n3_t1, timbre="open_string",
            vibrato_hz=5.0, vibrato_cents=25.0,
            pitch_status="known", open_string=True,
            label="N3 OPEN G string: known pitch, octave anchor",
            string=3, fret=0))
    else:
        notes.append(PhraseNote(
            f0_hz=f_g3, t0=n3_t0, t1=n3_t1, timbre="even",
            vibrato_hz=5.0, vibrato_cents=25.0,
            pitch_status="candidate", open_string=False,
            label="N3 anchor removed: even-only G3 (still undecidable)",
            string=3, fret=0))

    # N4: candidate again, resolvable only while the anchor exists
    n4_t0, n4_t1 = 3.07, 3.95
    notes.append(PhraseNote(
        f0_hz=f_f3, t0=n4_t0, t1=n4_t1, timbre="even",
        vibrato_hz=4.5, vibrato_cents=15.0,
        pitch_status="candidate", label="N4 even-only F3 (octave undecidable)",
        string=3, fret=10))

    duration = 4.25
    # a breath (unvoiced) gap inside the phrase and a short trailing breath
    unvoiced = [
        [2.00, 2.06, "breath"],
        [2.95, 3.07, "breath"],
        [3.95, duration, "breath"],
    ]
    desc = ("Legato phrase: first-half missing-fundamental notes have two "
            "consistent octave readings; the open G string in the second half "
            "provides the known-pitch anchor for octave resolution."
            if with_anchor else
            "Same phrase with the open-string anchor removed: no note justifies "
            "a definite octave, so candidate ranges must stay unresolved.")
    return Phrase(name=name, notes=notes, duration=duration,
                  unvoiced=unvoiced, description=desc,
                  seed=2024 if with_anchor else 2025)


# ---------------------------------------------------------------------------
# Material pair: same performed pitch, opposite fundamental presence, aligned
# on an independent (shared) score.  Decidable vs. undecidable time spans are
# marked explicitly.
# ---------------------------------------------------------------------------

def build_fundamental_pair(name: str = "pair_f3",
                           f0_hz: float = 174.61) -> tuple[Phrase, Phrase, dict]:
    """Two phrases with identical pitch/timing grid, opposite f0 presence.

    ``full`` take radiates the fundamental (pitch decidable frame by frame);
    ``even`` take suppresses it (only the octave candidate range is justified).
    The independent alignment grid and the decidable/undecidable spans are
    returned as the third element.
    """
    t0, t1 = 0.25, 2.40
    common = dict(t0=t0, t1=t1, f0_hz=f0_hz,
                  vibrato_hz=5.0, vibrato_cents=22.0, vibrato_delay=0.2)
    full_note = PhraseNote(timbre="full", pitch_status="known",
                           label=f"{name}: fundamental present (decidable)",
                           **common)
    even_note = PhraseNote(timbre="even", pitch_status="candidate",
                           label=f"{name}: fundamental absent (undecidable)",
                           **common)
    duration = 2.70
    p_full = Phrase(name=f"{name}_full", notes=[full_note], duration=duration,
                    description="Material pair take A: fundamental present.",
                    seed=3101)
    p_even = Phrase(name=f"{name}_even", notes=[even_note], duration=duration,
                    description="Material pair take B: missing fundamental.",
                    seed=3102)
    alignment = {
        "pair_name": name,
        "performed_pitch_hz": f0_hz,
        "performed_midi": round(midi_from_hz(f0_hz), 3),
        "independent_score": [
            {"t0": 0.0, "t1": t0, "kind": "rest"},
            {"t0": t0, "t1": t1, "kind": "note", "midi": round(midi_from_hz(f0_hz), 3)},
            {"t0": t1, "t1": duration, "kind": "rest"},
        ],
        "spans": [
            {"take": f"{name}_full", "t0": t0, "t1": t1,
             "fundamental_present": True, "decidability": "decidable"},
            {"take": f"{name}_even", "t0": t0, "t1": t1,
             "fundamental_present": False, "decidability": "undecidable",
             "candidate_range_hz": [f0_hz, 2.0 * f0_hz]},
        ],
    }
    return p_full, p_even, alignment


def build_breathy_phrase(name: str = "phrase_breathy") -> Phrase:
    """A pitched but breathy phrase plus fully unvoiced aspiration regions.

    Used to prove the analyzer keeps a weak pitch through broadband breath
    noise while reporting the purely noisy gaps as unvoiced.
    """
    f_d4 = hz_from_midi(62)      # 293.66
    f_c4 = hz_from_midi(60)      # 261.63
    notes = [
        PhraseNote(f0_hz=f_d4, t0=0.25, t1=1.35, timbre="breathy",
                   vibrato_hz=5.0, vibrato_cents=12.0, noise_amp=0.09,
                   attack=0.05, release=0.06, pitch_status="known",
                   label="breathy D4 (weak partials, strong aspiration)"),
        PhraseNote(f0_hz=f_c4, t0=1.55, t1=2.60, timbre="breathy",
                   vibrato_hz=4.5, vibrato_cents=10.0, noise_amp=0.11,
                   attack=0.05, release=0.06, pitch_status="known",
                   label="breathy C4"),
    ]
    duration = 2.90
    unvoiced = [
        [0.0, 0.25, "breath"],
        [1.35, 1.55, "breath"],
        [2.60, duration, "breath"],
    ]
    return Phrase(name=name, notes=notes, duration=duration,
                  unvoiced=unvoiced,
                  description="Breathy pitched phrase with unvoiced aspiration "
                              "gaps for voicing/cent verification.",
                  seed=4242)
