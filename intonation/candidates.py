"""Candidate F0 generation from inspectable harmonic / period evidence.

The generator never looks for "the largest spectral peak".  It forms
harmonic-comb hypotheses (a spectral peak may be the k-th harmonic of a much
lower fundamental) plus YIN period hypotheses, refines each hypothesis by
weighted least squares over the peaks it explains, and scores it by

  * how much peak energy it explains, weighted towards low harmonics,
  * how much strong peak energy it fails to explain,
  * which low harmonics are missing,
  * whether an independent period estimate (YIN) supports it.

Each candidate keeps its matched harmonics, missing harmonics, odd/even split,
coverage and source string, so a reviewer can see exactly why it scored what
it scored.  Retained-but-unused candidates keep their scores and a rejection
reason.

Octave handling is deliberately *not* decided here.  A lower-octave candidate
that explains the strong peaks but is missing its fundamental is reported as
``viable`` — it is the missing-fundamental hypothesis.  Whether it survives is
decided over time by the tracking / revision layers using later evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .config import AnalysisConfig, cents
from .spectrum import Peak
from .yin import YinResult


@dataclass
class MatchedHarmonic:
    k: int
    peak_index: int
    peak_freq: float
    peak_amp: float
    cents_err: float

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "peak_freq": round(self.peak_freq, 3),
            "peak_amp": round(self.peak_amp, 5),
            "cents_err": round(self.cents_err, 2),
        }


@dataclass
class Candidate:
    f0: float
    score: float
    explained_amp: float          # unweighted normalised amplitude explained
    explained_weighted: float     # sum a_k / k**p
    unexplained_amp: float
    subharmonic_amp: float        # strong peak energy below f0 (not explainable)
    coverage: float               # explained_amp / strong_peak_amp
    strong_peak_amp: float
    matched: list[MatchedHarmonic]
    missing: list[int]
    odd_energy: float
    even_energy: float
    fundamental_present: bool
    fundamental_cents: float | None
    yin_agree: bool
    viable: bool                  # explains essentially all strong peaks
    source: str
    n_harmonics: int = 0
    periodicity: float = 0.0

    rejection: str | None = None
    rank: int = -1

    @property
    def odd_even_ratio(self) -> float:
        if self.even_energy <= 1e-9:
            return float("inf") if self.odd_energy > 1e-9 else 0.0
        return self.odd_energy / self.even_energy

    def to_dict(self) -> dict:
        return {
            "f0_hz": round(self.f0, 3),
            "score": round(self.score, 4),
            "coverage": round(self.coverage, 4),
            "viable": self.viable,
            "explained_amp": round(self.explained_amp, 4),
            "explained_weighted": round(self.explained_weighted, 4),
            "unexplained_amp": round(self.unexplained_amp, 4),
            "subharmonic_amp": round(self.subharmonic_amp, 4),
            "n_harmonics": self.n_harmonics,
            "matched": [m.to_dict() for m in self.matched],
            "missing": list(self.missing),
            "odd_energy": round(self.odd_energy, 4),
            "even_energy": round(self.even_energy, 4),
            "odd_even_ratio": (round(self.odd_even_ratio, 3)
                               if math.isfinite(self.odd_even_ratio) else None),
            "fundamental_present": self.fundamental_present,
            "fundamental_cents": (round(self.fundamental_cents, 2)
                                  if self.fundamental_cents is not None else None),
            "yin_agree": self.yin_agree,
            "periodicity": round(self.periodicity, 4),
            "source": self.source,
            "rank": self.rank,
            "rejection": self.rejection,
        }


def _hypotheses(peaks: list[Peak], yin: YinResult, cfg: AnalysisConfig):
    hyps: list[tuple[float, str]] = []
    n_top = min(len(peaks), cfg.hypothesis_peaks)
    for i in range(n_top):
        p = peaks[i]
        if p.freq_hz <= 0:
            continue
        kmax = min(cfg.max_hypothesis_order,
                   int(math.floor(cfg.max_harmonic_hz / p.freq_hz)) + 1)
        for k in range(1, kmax + 1):
            f = p.freq_hz / k
            if cfg.fmin <= f <= cfg.fmax:
                hyps.append((f, f"peak{i + 1}/{k}"))
    if yin.f0 > 0 and yin.aperiodicity < 0.65:
        for mult, tag in ((1.0, "yin"), (0.5, "yin/2"), (2.0, "yin*2")):
            f = yin.f0 * mult
            if cfg.fmin <= f <= cfg.fmax:
                hyps.append((f, tag))
    return hyps


def _cluster(hyps, cfg):
    if not hyps:
        return []
    hyps = sorted(hyps, key=lambda t: t[0])
    out = []
    cf = [hyps[0][0]]
    cs = [hyps[0][1]]
    for f, src in hyps[1:]:
        if abs(cents(f, cf[-1])) <= cfg.cluster_cents:
            cf.append(f)
            cs.append(src)
        else:
            out.append((float(np.median(cf)), cs[0] + f"+{len(cs) - 1}"))
            cf = [f]
            cs = [src]
    out.append((float(np.median(cf)), cs[0] + f"+{len(cs) - 1}"))
    return out


def _match_peaks(f0: float, peaks: list[Peak], cfg: AnalysisConfig):
    """Match peaks to integer harmonics of f0.  Returns
    (matched, missing, explained, unexplained, subharmonic, strong_total)."""
    pf = np.array([p.freq_hz for p in peaks])
    pa = np.array([p.amp for p in peaks])
    top = float(pa.max()) if pa.size else 0.0
    if top <= 0:
        return [], [], 0.0, 0.0, 0.0, 0.0
    an = pa / top

    strong_mask = an >= cfg.strong_peak_ratio
    strong_total = float(np.sum(an[strong_mask])) if np.any(strong_mask) else top

    kmax = min(cfg.max_harmonic_order, int(math.floor(cfg.max_harmonic_hz / f0)))
    if kmax < 1:
        return [], [], 0.0, 0.0, 0.0, strong_total

    ratio_min = 2.0 ** (cfg.match_tolerance_cents / 1200.0) - 1.0
    matched: list[MatchedHarmonic] = []
    missing: list[int] = []
    used = set()
    for k in range(1, kmax + 1):
        exp = f0 * k
        tol_hz = max(exp * ratio_min, cfg.match_tolerance_min_hz)
        tol_cents = 1200.0 * math.log2(1.0 + tol_hz / exp)
        cents_diff = 1200.0 * np.log2(pf / exp)
        within = np.abs(cents_diff) <= tol_cents
        within &= np.isfinite(cents_diff)
        # do not let two harmonics claim the same peak
        cand_idx = [j for j in np.where(within)[0] if j not in used]
        if not cand_idx:
            missing.append(k)
            continue
        cand_idx = np.array(cand_idx)
        best = cand_idx[np.lexsort((np.abs(cents_diff[cand_idx]),
                                    -an[cand_idx]))[0]]
        matched.append(MatchedHarmonic(k, int(best), float(pf[best]),
                                       float(pa[best]), float(cents_diff[best])))
        used.add(int(best))

    explained = float(sum(an[m.peak_index] for m in matched))
    strong_mask = an >= cfg.strong_peak_ratio
    unexplained = float(sum(an[j] for j in range(len(peaks))
                            if j not in used and strong_mask[j]))
    # Subharmonic evidence: a strong, *unexplained* peak below f0 cannot belong
    # to this hypothesis at all (it is a subharmonic), so it is strong evidence
    # that the true fundamental is lower.  Sum those amplitudes separately.
    subharmonic = 0.0
    for j in range(len(peaks)):
        if j in used or not strong_mask[j]:
            continue
        if pf[j] < 0.75 * f0:
            subharmonic += float(an[j])
    strong_total = float(np.sum(an[strong_mask])) if np.any(strong_mask) else float(an.max())
    return matched, missing, explained, unexplained, subharmonic, strong_total


def _refine_f0(f0, matched, cfg) -> float:
    """Robust least-squares refinement over *strong* matched harmonics.

    Weight = (a_k)**2 / k**2 so that a handful of loud low harmonics dominate;
    quiet high harmonics (often noise peaks) cannot drag the estimate.  Only
    harmonics within the match tolerance are used, and the result may not move
    more than 60 cents from the hypothesis.
    """
    if not matched:
        return f0
    max_amp = max(m.peak_amp for m in matched)
    if max_amp <= 0:
        return f0
    num = 0.0
    den = 0.0
    for m in matched:
        if m.k <= 0:
            continue
        rel = m.peak_amp / max_amp
        if rel < 0.15:          # ignore weak peaks: likely noise / partials
            continue
        w = (m.peak_amp ** 2) / (m.k ** 2)
        est = m.peak_freq / m.k
        num += w * est
        den += w
    if den <= 0:
        return f0
    refined = num / den
    if abs(cents(refined, f0)) > 60.0:
        return f0
    return refined


def score_hypothesis(f0, source, peaks, yin, cfg) -> Candidate | None:
    if not peaks or f0 <= 0:
        return None
    matched, missing, explained, unexplained, subharmonic, strong_total = _match_peaks(
        f0, peaks, cfg)
    if not matched:
        return None
    f0 = _refine_f0(f0, matched, cfg)
    # re-match at the refined f0 for consistent reporting
    matched, missing, explained, unexplained, subharmonic, strong_total = _match_peaks(
        f0, peaks, cfg)
    if not matched:
        return None

    p = cfg.harmonic_weight_power
    explained_w = sum(m.peak_amp ** 2 / (m.k ** p) for m in matched)
    odd_e = sum(m.peak_amp for m in matched if m.k % 2 == 1)
    even_e = sum(m.peak_amp for m in matched if m.k % 2 == 0)
    fund_present = any(m.k == 1 for m in matched)
    fund_cents = None
    for m in matched:
        if m.k == 1:
            fund_cents = m.cents_err

    # Only penalise *interior* gaps: a missing harmonic that sits below the
    # highest matched harmonic.  Harmonics above the top matched one may
    # simply not be radiated by the instrument (or exceed the band), and a
    # missing fundamental is a legitimate timbre (missing-fundamental tones),
    # so it gets a reduced weight.
    top_k = max(m.k for m in matched)
    miss_w = 0.0
    for k in missing:
        if k < top_k:
            miss_w += (0.5 if k == 1 else 1.0) / (k ** p)

    yin_agree = False
    if yin.f0 > 0 and yin.aperiodicity < 0.55:
        for mult in (0.5, 1.0, 2.0):
            if abs(cents(f0, yin.f0 * mult)) <= 70.0:
                yin_agree = True
                break
    periodicity = max(0.0, 1.0 - yin.aperiodicity) if yin.f0 > 0 else 0.0

    strong_total = max(strong_total, 1e-9)
    coverage = float(min(1.0, explained / strong_total))
    unexpl_ratio = unexplained / strong_total
    sub_ratio = subharmonic / strong_total
    viable = coverage >= 0.90 and unexpl_ratio <= 0.12

    # Tuning residual: a correct f0 explains strong peaks with *small* cents
    # errors; a wrong hypothesis can only absorb them as badly detuned
    # harmonics.  Weighted by peak strength and harmonic order.
    num_res = 0.0
    den_res = 0.0
    top_amp = max((m.peak_amp for m in matched), default=0.0)
    for m in matched:
        w = (m.peak_amp / top_amp) / m.k if top_amp > 0 else 0.0
        num_res += w * abs(m.cents_err)
        den_res += w
    mean_res_cents = num_res / den_res if den_res > 0 else 0.0
    residual_penalty = min(1.0, mean_res_cents / cfg.match_tolerance_cents)

    # Scale-invariant score: fraction of strong harmonic energy explained, with
    # a low-harmonic emphasis (sum a_k^2/k**p / total a_k^2), minus penalties.
    # The subharmonic term is what distinguishes "real high note" from
    # "low note whose fundamental is weak": a strong unexplained peak *below*
    # f0 cannot belong to this hypothesis and argues for a lower fundamental.
    total_energy = float(np.sum(np.array([p_.amp for p_ in peaks]) ** 2))
    low_emph = float(explained_w / max(total_energy, 1e-9))
    score = (coverage
             + 0.9 * low_emph
             - 1.2 * unexpl_ratio
             - 1.5 * sub_ratio
             - 0.15 * miss_w
             - 0.9 * residual_penalty
             + (0.15 if yin_agree else 0.0))
    score = max(0.0, score)

    return Candidate(
        f0=float(f0), score=float(score),
        explained_amp=explained, explained_weighted=float(explained_w),
        unexplained_amp=unexplained, subharmonic_amp=subharmonic,
        coverage=coverage, strong_peak_amp=strong_total,
        matched=matched, missing=missing,
        odd_energy=float(odd_e), even_energy=float(even_e),
        fundamental_present=fund_present, fundamental_cents=fund_cents,
        yin_agree=yin_agree, viable=bool(viable), source=source,
        n_harmonics=len(matched), periodicity=periodicity,
    )


def generate_candidates(peaks, yin, cfg, extra_sources=None) -> list[Candidate]:
    if not peaks:
        return []
    hyps = _hypotheses(peaks, yin, cfg)
    if extra_sources:
        hyps.extend(extra_sources)
    clustered = _cluster(hyps, cfg)
    cands: list[Candidate] = []
    for f, src in clustered:
        c = score_hypothesis(f, src, peaks, yin, cfg)
        if c is not None and c.n_harmonics >= 1:
            cands.append(c)
    if not cands:
        return []
    cands.sort(key=lambda c: c.score, reverse=True)
    # de-duplicate refined hypotheses that collapsed onto the same f0
    dedup: list[Candidate] = []
    for c in cands:
        if any(abs(cents(c.f0, d.f0)) <= cfg.cluster_cents for d in dedup):
            # keep the higher-scoring one (cands are sorted, so skip)
            continue
        dedup.append(c)
    cands = dedup

    # A candidate that explains essentially every strong peak (``viable``) is a
    # better physical description than one that only explains a subset, even
    # when the latter has a loud, isolated low harmonic.  Promote the best
    # viable candidate over a non-viable leader only when its score is
    # comparable (within this margin), so a genuinely better non-viable reading
    # (e.g. one with far more explained low-harmonic energy) is not displaced.
    # This is what stops a decaying tone — whose weak fundamental has faded
    # while its dominant second harmonic still rings — from being read an
    # octave high.
    if cands and not cands[0].viable:
        lead = cands[0]
        viable_best = None
        for c in cands[1:]:
            if c.viable and c.score >= cfg.viable_promotion_margin * lead.score:
                if viable_best is None or c.score > viable_best.score:
                    viable_best = c
        if viable_best is not None:
            cands.remove(viable_best)
            cands.insert(0, viable_best)

    best = cands[0].score
    kept: list[Candidate] = []
    for i, c in enumerate(cands):
        c.rank = i
        if c.score >= cfg.candidate_score_ratio * best:
            kept.append(c)
        else:
            c.rejection = "below_retention_threshold"
        if len(kept) >= cfg.candidate_keep:
            break

    # Always retain a viable lower-octave partner of the best candidate: it is
    # the missing-fundamental hypothesis and must be available to the
    # cross-frame layer even when its frame score is lower.
    if kept:
        best_c = kept[0]
        for c in cands:
            if any(c is k for k in kept):
                continue
            if (abs(cents(best_c.f0, 2.0 * c.f0)) <= cfg.octave_pair_cents
                    and c.viable):
                c.rejection = None
                c.rank = len(kept)
                kept.append(c)
                break

    kept.sort(key=lambda c: c.score, reverse=True)
    # Honour the viability preference in the final order too: a viable candidate
    # within the promotion margin of the top score should lead, so downstream
    # frame/tracker logic sees the physically better description first.
    if kept and not kept[0].viable:
        top_score = kept[0].score
        chosen = None
        for c in kept[1:]:
            if c.viable and c.score >= cfg.viable_promotion_margin * top_score:
                chosen = c
                break
        if chosen is not None:
            kept.remove(chosen)
            kept.insert(0, chosen)
    for i, c in enumerate(kept):
        c.rank = i
    for c in cands:
        if not any(c is k for k in kept) and c.rejection is None:
            c.rejection = "not_in_top_keep"
    return kept


def octave_relationships(cands: list[Candidate], cfg: AnalysisConfig) -> dict:
    """Describe octave partners among the retained candidates.

    A harmonic series is spectrally indistinguishable from an even-harmonic
    series one octave lower (the sub-octave reading has its fundamental and odd
    harmonics missing, its even harmonics coincident with the upper reading).
    So a frame is octave-uncertain when *both* the best candidate and a viable
    sub-octave candidate explain the strong peaks with comparable scores.

    This function reports that purely spectral relation.  It does NOT decide
    the octave — that is done over time by the tracker / streaming gate using
    attack and sustain evidence, exactly because a single frame cannot.
    """
    empty = {"has_pair": False, "lower_f0": None, "lower_ratio": None,
             "upper_f0": None}
    if not cands:
        return empty
    best = cands[0]
    lower = None
    best_ratio = 0.0
    for c in cands[1:]:
        if c.f0 <= 0:
            continue
        if abs(cents(best.f0, 2.0 * c.f0)) <= cfg.octave_pair_cents:
            # A sub-octave reading is a genuine spectral alias when it is the
            # missing-fundamental kind: it explains the strong peaks using only
            # *even* harmonics of the proposed lower f0, with the lower
            # fundamental and odd multiples absent.  That is spectrally
            # identical to the upper reading, which is exactly why the octave
            # cannot be settled from one frame.
            if not c.viable or c.fundamental_present:
                continue
            ratio = c.score / best.score if best.score > 1e-9 else 0.0
            if ratio > best_ratio:
                best_ratio = ratio
                lower = c
    has_pair = lower is not None and best_ratio >= cfg.octave_ambiguity_margin
    return {
        "has_pair": bool(has_pair),
        "lower_f0": lower.f0 if lower is not None else None,
        "lower_ratio": best_ratio if lower is not None else None,
        "lower_missing_fundamental": (not lower.fundamental_present) if lower else None,
        "upper_f0": best.f0,
    }
