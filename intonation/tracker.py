"""Frame tracking, note segmentation, octave deferral and offline revision.

Two consumers share this module:

* the **offline** analyzer, which may look ahead and therefore may *revise*
  earlier frames using later evidence;
* the **streaming** analyzer, which may not look ahead and must decide within a
  beat deadline.

The design that makes both honest:

1. Each frame yields a tentative pitch (or no pitch).
2. When a frame's octave is genuinely ambiguous (an even-harmonic / missing
   fundamental timbre), the pitch is *deferred*, not committed.  The deferral
   records the candidate pair and a deadline.
3. A note is only emitted once it is either unambiguously supported or the
   deferral has been resolved by later evidence (offline) or the deadline has
   passed (streaming).
4. Offline, a resolved deferral can *relabel* earlier frames — this is how the
   opening which literally reads an octave high is corrected to the true low
   fundamental when the fundamental appears later under a legato continuation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np

from .config import AnalysisConfig, StreamingConfig, cents
from .analyzer import Frame


@dataclass
class NoteEvent:
    t_start: float
    t_end: float
    sample_start: int
    sample_end: int
    f0_hz: float | None
    median_f0_hz: float | None
    state: str                    # voiced | silence | noise | ambiguous
    confidence: float
    n_frames: int
    source: str = "auto"
    octave_corrected: bool = False
    revision: str | None = None
    onset: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.t_end - self.t_start)

    def to_dict(self) -> dict:
        return {
            "t_start": round(self.t_start, 6),
            "t_end": round(self.t_end, 6),
            "sample_start": int(self.sample_start),
            "sample_end": int(self.sample_end),
            "duration_s": round(self.duration, 6),
            "f0_hz": round(self.f0_hz, 3) if self.f0_hz is not None else None,
            "median_f0_hz": (round(self.median_f0_hz, 3)
                             if self.median_f0_hz is not None else None),
            "state": self.state,
            "confidence": round(self.confidence, 4),
            "n_frames": self.n_frames,
            "source": self.source,
            "octave_corrected": self.octave_corrected,
            "revision": self.revision,
            "onset": self.onset,
            "midi": (round(69.0 + 12.0 * math.log2(self.median_f0_hz / 440.0), 3)
                     if self.median_f0_hz else None),
        }


@dataclass
class Deferral:
    start_index: int
    end_index: int
    lower_f0: float
    upper_f0: float
    start_time: float
    deadline_time: float
    resolved: bool = False
    resolution: str | None = None
    chosen_f0: float | None = None

    def to_dict(self) -> dict:
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "start_time": round(self.start_time, 4),
            "deadline_time": round(self.deadline_time, 4),
            "lower_f0_hz": round(self.lower_f0, 3),
            "upper_f0_hz": round(self.upper_f0, 3),
            "resolved": self.resolved,
            "resolution": self.resolution,
            "chosen_f0_hz": (round(self.chosen_f0, 3)
                             if self.chosen_f0 is not None else None),
        }


def _tentative_f0(frame: Frame) -> float | None:
    if frame.state in ("voiced", "ambiguous"):
        return frame.f0_hz
    return None


def _sustained_region(frames: list[Frame], start: int, cfg: AnalysisConfig) -> tuple[int, int]:
    """Return [start, end) of the contiguous same-pitch region around `start`.

    Used to decide whether consecutive voiced frames belong to one note.
    """
    n = len(frames)
    end = start
    base = _tentative_f0(frames[start])
    while end + 1 < n:
        nxt = frames[end + 1]
        f = _tentative_f0(nxt)
        if f is None:
            break
        # allow octave-ambiguous frames to continue the same region
        if abs(cents(f, base)) <= cfg.note_pitch_tolerance_cents or \
                abs(cents(f, 2.0 * base)) <= cfg.note_pitch_tolerance_cents or \
                abs(cents(f, 0.5 * base)) <= cfg.note_pitch_tolerance_cents:
            end += 1
        else:
            break
    return start, end + 1


def _region_pitch(frames: list[Frame], lo: int, hi: int) -> tuple[float | None, float]:
    vals = []
    confs = []
    for i in range(lo, hi):
        f = _tentative_f0(frames[i])
        if f is not None:
            vals.append(f)
            confs.append(frames[i].confidence)
    if not vals:
        return None, 0.0
    vals = np.array(vals)
    confs = np.array(confs)
    # choose the value with the highest confidence, robust to octave flips
    order = np.argsort(confs)[::-1]
    ref = vals[order[0]]
    close = vals[np.abs(np.array([cents(v, ref) for v in vals])) < 120]
    med = float(np.median(close)) if close.size else float(ref)
    return med, float(np.mean(confs))


def _octave_evidence_after(frames: list[Frame], hi: int, low_f0: float,
                           cfg: AnalysisConfig, window_beats: float) -> dict:
    """Look at frames after `hi` for evidence of the *low* fundamental.

    Two independent pieces of evidence:

    * **odd-harmonic / fundamental return**: a later voiced region at the low
      f0 with a real fundamental strongly implies the earlier reading was the
      upper octave.
    * **continuation without re-articulation**: if no new attack occurs between
      the ambiguous region and the later low-f0 region, the same note simply
      revealed its fundamental -> relabel.

    Returns a dict with the evidence and a verdict.
    """
    n = len(frames)
    win_frames = max(1, int(round(window_beats * cfg.beat_seconds / cfg.hop_seconds)))
    start = hi
    stop = min(n, hi + win_frames)
    attacks_between = sum(1 for i in range(hi, stop) if frames[i].is_attack)

    low_region = []
    for i in range(start, stop):
        f = frames[i]
        fr = f.f0_hz
        if fr is None:
            continue
        if abs(cents(fr, low_f0)) <= 60.0:
            low_region.append(i)

    fundamental_seen = False
    odd_evidence = 0.0
    for i in low_region:
        for c in frames[i].candidates:
            if abs(cents(c.f0, low_f0)) <= 60.0:
                if c.fundamental_present:
                    fundamental_seen = True
                odd_evidence = max(odd_evidence, c.odd_even_ratio)
                break

    has_low = len(low_region) >= 3
    verdict = None
    if has_low and (fundamental_seen or odd_evidence >= cfg.revision_min_odd_energy):
        if attacks_between == 0:
            verdict = "relabel_low"
        else:
            # a new attack may mean a genuine leap that later returns; only
            # relabel if the low region follows immediately and is strong
            verdict = "relabel_low" if len(low_region) >= 6 else "keep"
    elif attacks_between == 0 and has_low:
        verdict = "relabel_low"

    return {
        "low_region_frames": [int(i) for i in low_region],
        "fundamental_seen": fundamental_seen,
        "odd_evidence": round(float(odd_evidence), 3),
        "attacks_between": int(attacks_between),
        "verdict": verdict,
        "lookahead_s": round((stop - hi) * cfg.hop_seconds, 3),
    }


def track_and_segment(frames: list[Frame], cfg: AnalysisConfig,
                      allow_revision: bool = True) -> dict:
    """Produce notes, deferrals and (offline) revisions from frame evidence.

    Returns {"notes": [...], "deferrals": [...], "revisions": [...],
             "frames": frames}.
    """
    n = len(frames)
    deferrals: list[Deferral] = []
    revisions: list[dict] = []

    # ---- pass 1: identify octave-ambiguous regions and defer them ----
    # A deferral region is a run of frames whose *literal* pitch (the upper
    # reading) is continuous, mirroring note segmentation.  Splitting on
    # literal pitch keeps the shared opening (392 Hz) separate from a
    # continuation that reads 196 Hz, even though both are octave-ambiguous.
    amb_idx = [i for i in range(n)
               if frames[i].state == "ambiguous"
               and frames[i].lower_octave_f0 and frames[i].upper_octave_f0]
    gap_tol = max(2, int(round(0.10 / cfg.hop_seconds)))
    pair_tol_cents = 120.0

    def _literal(i: int) -> float | None:
        return frames[i].f0_hz

    groups: list[list[int]] = []
    for i in amb_idx:
        if groups and (i - groups[-1][-1]) <= gap_tol:
            g = groups[-1]
            ref = _literal(g[-1])
            # continue only if the literal pitch and octave pair are continuous
            if (ref is not None and _literal(i) is not None
                    and abs(cents(_literal(i), ref)) <= pair_tol_cents
                    and abs(cents(frames[i].lower_octave_f0,
                                  frames[g[-1]].lower_octave_f0)) <= pair_tol_cents):
                g.append(i)
                continue
        groups.append([i])

    # Bridge isolated non-ambiguous frames *inside* a group whose literal pitch
    # matches the surrounding upper reading (transition frames where the
    # fundamental is momentarily too weak to form a viable lower candidate).
    for g in groups:
        lo_ref = float(np.median([frames[i].lower_octave_f0 for i in g]))
        hi_ref = float(np.median([frames[i].upper_octave_f0 for i in g]))

        def _is_bridge(j: int) -> bool:
            f = frames[j].f0_hz
            return f is not None and abs(cents(f, hi_ref)) <= pair_tol_cents

        i = g[0] - 1
        while i >= 0 and (g[0] - i) <= gap_tol and _is_bridge(i):
            g.insert(0, i)
            i -= 1
        j = g[-1] + 1
        while j < n and (j - g[-1]) <= gap_tol and _is_bridge(j):
            g.append(j)
            j += 1

    # Bridging can make adjacent groups overlap; merge any that now do so.
    merged: list[list[int]] = []
    for g in sorted(groups, key=lambda z: z[0]):
        if merged and g[0] <= merged[-1][-1]:
            merged[-1].extend(k for k in g if k > merged[-1][-1])
        else:
            merged.append(sorted(set(g)))
    groups = merged

    for g in groups:
        lo_i, hi_i = g[0], g[-1]
        low_vals = [frames[i].lower_octave_f0 for i in g
                    if frames[i].lower_octave_f0]
        up_vals = [frames[i].upper_octave_f0 for i in g
                   if frames[i].upper_octave_f0]
        if not low_vals or not up_vals:
            continue
        low_f0 = float(np.median(low_vals))
        up_f0 = float(np.median(up_vals))
        d = Deferral(
            start_index=lo_i, end_index=hi_i,
            lower_f0=low_f0, upper_f0=up_f0,
            start_time=frames[lo_i].t_center,
            deadline_time=frames[lo_i].t_center
            + cfg.decision_deadline_beats * cfg.beat_seconds,
        )
        deferrals.append(d)
        for i in g:
            frames[i].decision = "deferred"
            frames[i].lower_octave_f0 = low_f0
            frames[i].upper_octave_f0 = up_f0

    # ---- pass 2: resolve deferrals ----
    # offline: use later evidence inside the lookahead window
    for d in deferrals:
        if not allow_revision:
            continue
        ev = _octave_evidence_after(frames, d.end_index + 1, d.lower_f0, cfg,
                                    cfg.revision_lookahead_beats)
        d_dict = ev
        if ev["verdict"] == "relabel_low":
            d.resolved = True
            d.resolution = "later_fundamental_evidence"
            d.chosen_f0 = d.lower_f0
            # relabel the deferred frames
            for i in range(d.start_index, d.end_index + 1):
                fr = frames[i]
                fr.tracked_f0 = d.lower_f0
                fr.octave_corrected = True
                fr.revision_note = "relabelled to lower fundamental: " \
                                   "fundamental/odd harmonics appeared later " \
                                   "with no new attack"
                fr.state = "voiced"
                fr.decision = "revised"
            revisions.append({
                "kind": "octave_relabel",
                "region_t": [round(frames[d.start_index].t_start, 4),
                             round(frames[d.end_index].t_end, 4)],
                "from_f0_hz": round(d.upper_f0, 3),
                "to_f0_hz": round(d.lower_f0, 3),
                "evidence": d_dict,
            })
        else:
            d.resolved = True
            d.resolution = "no_later_evidence_keep_upper"
            d.chosen_f0 = d.upper_f0
            for i in range(d.start_index, d.end_index + 1):
                fr = frames[i]
                fr.tracked_f0 = fr.f0_hz or d.upper_f0
                fr.state = "voiced"
                fr.decision = "octave_kept_upper"

    # ---- pass 3: fill tracked f0 for non-deferred frames ----
    for i, fr in enumerate(frames):
        if fr.tracked_f0 is None and fr.f0_hz is not None:
            fr.tracked_f0 = fr.f0_hz
        if fr.tracked_state is None:
            if fr.tracked_f0 is not None:
                fr.tracked_state = "voiced"
            elif fr.state == "attack":
                fr.tracked_state = "attack"
            elif fr.state == "silence":
                fr.tracked_state = "silence"
            else:
                fr.tracked_state = "unvoiced"

    # ---- pass 4: segment into notes by pitch continuity ----
    notes = segment_notes(frames, cfg)
    return {
        "notes": [nt.to_dict() for nt in notes],
        "note_objects": notes,
        "deferrals": [d.to_dict() for d in deferrals],
        "revisions": revisions,
        "frames": frames,
    }


def segment_notes(frames: list[Frame], cfg: AnalysisConfig) -> list[NoteEvent]:
    n = len(frames)
    notes: list[NoteEvent] = []
    min_frames = max(1, int(round(cfg.note_min_duration_ms / 1000.0 / cfg.hop_seconds)))

    i = 0
    while i < n:
        fr = frames[i]
        if fr.tracked_f0 is None:
            # run of unvoiced frames -> silence or noise event
            j = i + 1
            while j < n and frames[j].tracked_f0 is None:
                j += 1
            states = [frames[k].tracked_state for k in range(i, j)]
            state = "silence" if states.count("silence") >= (j - i) / 2 else "noise"
            if state == "noise" and (j - i) < min_frames:
                # merge very short noise gaps into neighbours later; still record
                pass
            notes.append(NoteEvent(
                t_start=frames[i].t_start, t_end=frames[j - 1].t_end,
                sample_start=frames[i].sample_start,
                sample_end=frames[j - 1].sample_end,
                f0_hz=None, median_f0_hz=None, state=state,
                confidence=0.0, n_frames=j - i, source="auto",
            ))
            i = j
            continue

        # voiced run at one pitch
        base = fr.tracked_f0
        j = i + 1
        while j < n:
            nxt = frames[j]
            if nxt.tracked_f0 is None:
                break
            if abs(cents(nxt.tracked_f0, base)) > cfg.note_pitch_tolerance_cents:
                break
            j += 1
        region = frames[i:j]
        f0s = np.array([f.tracked_f0 for f in region])
        confs = np.array([f.confidence for f in region])
        median_f0 = float(np.median(f0s))
        corrected = any(f.octave_corrected for f in region)
        revision_note = next((f.revision_note for f in region if f.revision_note), None)
        onset = any(f.is_attack for f in region[:3])

        if (j - i) < min_frames:
            # too short: represent as a brief note but mark it
            state = "voiced"
        else:
            state = "voiced"
        notes.append(NoteEvent(
            t_start=region[0].t_start, t_end=region[-1].t_end,
            sample_start=region[0].sample_start, sample_end=region[-1].sample_end,
            f0_hz=median_f0, median_f0_hz=median_f0, state=state,
            confidence=float(np.mean(confs)), n_frames=j - i,
            octave_corrected=corrected, revision=revision_note, onset=onset,
        ))
        i = j
    return notes


def leading_silence(notes: list[dict]) -> float | None:
    for nt in notes:
        if nt["state"] == "voiced":
            return nt["t_start"]
    return None
