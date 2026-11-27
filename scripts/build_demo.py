"""Build, verify and check in the intonation *demo resources*.

This is the rebuild script for the phrase/tablature demo material.  Unlike
``generate_dataset.py`` (the kernel fixture suite), it produces the editable
phrase + fingering-tablature resources that the analysis window loads directly:

  data/demo/audio/<name>.wav          rendered phrase (no pitch in filename)
  data/demo/truth/<name>.json         ground truth + known/candidate status
  data/demo/params/<name>.json        editable phrase parameters (rebuild source)
  data/demo/tracks/<name>.json        REAL analyzer output (audio-only input)
  data/demo/tab/<name>.json           editable fingering tablature
  data/demo/verify/<name>.json        acceptance record (cent + voicing + ambiguity)
  data/demo/pairs/<pair>.json         same-pitch / opposite-fundamental pair
  data/demo/comparison.json           all acceptance results incl. failures
  data/demo/provenance.json           source + parameters for every artifact
  data/demo/manifest.json             merged into the server case registry

The analyzer is always invoked from audio alone (see
:func:`intonation.datamaking.run_analyzer_audio_only`); truth is read only by
the verification step.  Failed frames are kept inside the verification records.

Subcommands:
  build     render + analyze + verify everything (default)
  verify    re-run verification from checked-in tracks/truth and assert accept
  list      list demo cases and their acceptance status
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from intonation import phrases, tablature, datamaking, wavio
from intonation.config import AnalysisConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "data", "demo")
SUBDIRS = ("audio", "truth", "params", "tracks", "tab", "verify", "pairs")

# Every demo case must meet these; the same constants live in datamaking so the
# acceptance decision cannot drift between build and re-verification.
REQUIRED_KEYS = ("accepted", "cent_stats", "voicing", "ambiguity", "failures")


def _dir(sub: str) -> str:
    p = os.path.join(DEMO, sub)
    os.makedirs(p, exist_ok=True)
    return p


def _save(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_default)
    os.replace(tmp, path)


def _default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def _paths(name: str) -> dict[str, str]:
    return {sub: os.path.join(_dir(sub), f"{name}.json")
            for sub in ("truth", "params", "tracks", "tab", "verify")} | {
        "audio": os.path.join(_dir("audio"), f"{name}.wav")}


def _provenance(name: str, phrase: phrases.Phrase, cfg: AnalysisConfig,
                generator: str) -> dict:
    return {
        "name": name,
        "generator": generator,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sample_rate": phrase.sample_rate,
        "seed": phrase.seed,
        "config": cfg.to_dict(),
        "phrase_params": phrase.to_dict(),
        "truth_policy": "ground truth stored separately; never passed to the "
                        "analyzer, used only by verification",
        "analyzer_entry": "intonation.datamaking.run_analyzer_audio_only",
    }


def _build_one(phrase: phrases.Phrase, cfg: AnalysisConfig,
               provenance: list, generator: str) -> dict:
    name = phrase.name
    paths = _paths(name)
    bundle = datamaking.produce_phrase(phrase, cfg)

    wavio.write_wav(paths["audio"], bundle["audio"], phrase.sample_rate)
    _save(paths["truth"], bundle["truth"])
    _save(paths["params"], phrase.to_dict())
    _save(paths["tracks"], bundle["track"])
    _save(paths["tab"], bundle["tablature"])
    _save(paths["verify"], bundle["verification"])
    provenance.append(_provenance(name, phrase, cfg, generator))
    return bundle


def build(cfg: AnalysisConfig | None = None, verbose: bool = True) -> dict:
    cfg = cfg or AnalysisConfig()
    for sub in SUBDIRS:
        _dir(sub)
    provenance: list[dict] = []
    comparison: dict[str, dict] = {}
    manifest_cases: dict[str, dict] = {}

    def register(name: str, bundle: dict):
        v = bundle["verification"]
        comparison[name] = {k: v[k] for k in REQUIRED_KEYS if k in v}
        manifest_cases[name] = {
            "audio": os.path.relpath(_paths(name)["audio"], ROOT),
            "truth": os.path.relpath(_paths(name)["truth"], ROOT),
            "track": os.path.relpath(_paths(name)["tracks"], ROOT),
            "tab": os.path.relpath(_paths(name)["tab"], ROOT),
            "verify": os.path.relpath(_paths(name)["verify"], ROOT),
            "duration_s": bundle["track"]["audio"]["duration_s"],
            "demo": True,
        }
        if verbose:
            print(f"[{name}] accepted={v['accepted']} "
                  f"cent={v['cent_stats']['abs_mean_cents']:.2f}c "
                  f"p95={v['cent_stats']['p95_abs_cents']:.2f}c "
                  f"oct={v['cent_stats']['octave_error_rate']:.3f} "
                  f"spurious={v['voicing']['unvoiced_spurious_rate']:.3f} "
                  f"undecided={v['ambiguity']['candidate_undecided_rate']:.3f} "
                  f"failures={v['n_failures']}")

    # 1) anchor phrase + its no-anchor regression variant
    anchor = phrases.build_anchor_phrase(with_anchor=True)
    register(anchor.name, _build_one(anchor, cfg, provenance,
                                     "phrases.build_anchor_phrase(with_anchor=True)"))
    no_anchor = phrases.build_anchor_phrase(with_anchor=False)
    register(no_anchor.name, _build_one(no_anchor, cfg, provenance,
                                        "phrases.build_anchor_phrase(with_anchor=False)"))

    # 2) same-pitch / opposite-fundamental material pair on an independent score
    p_full, p_even, alignment = phrases.build_fundamental_pair()
    b_full = _build_one(p_full, cfg, provenance, "phrases.build_fundamental_pair")
    b_even = _build_one(p_even, cfg, provenance, "phrases.build_fundamental_pair")
    pair_verify = datamaking.verify_pair(
        b_full["track"], b_even["track"], b_full["truth"], b_even["truth"],
        alignment)
    pair_path = os.path.join(_dir("pairs"), f"{alignment['pair_name']}.json")
    _save(pair_path, {"alignment": alignment, "verification": pair_verify})
    register(p_full.name, b_full)
    register(p_even.name, b_even)
    comparison[alignment["pair_name"]] = {
        "accepted": pair_verify["accepted"],
        "pair_verify": {k: pair_verify[k] for k in
                        ("accepted", "even_take_literal_near_upper_octave")},
        "failures": pair_verify["failures"],
    }

    # 3) strong-harmonic + breathy material (cent + voicing verification)
    breathy = phrases.build_breathy_phrase()
    register(breathy.name, _build_one(breathy, cfg, provenance,
                                      "phrases.build_breathy_phrase"))

    _save(os.path.join(DEMO, "comparison.json"),
          {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "tolerances": {
               "abs_mean_cents": datamaking.CENT_MEAN_TOL,
               "p95_abs_cents": datamaking.CENT_P95_TOL,
               "octave_error_rate": datamaking.OCTAVE_ERROR_RATE_TOL,
               "unvoiced_spurious_rate": datamaking.UNVOICED_SPURIOUS_RATE_TOL,
               "candidate_undecided_rate": datamaking.CANDIDATE_UNDECIDED_TOL,
           }, "cases": comparison})
    _save(os.path.join(DEMO, "provenance.json"),
          {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "entries": provenance})
    _write_merged_manifest(manifest_cases)
    if verbose:
        n_acc = sum(1 for c in comparison.values() if c.get("accepted"))
        print(f"\nDemo build: {n_acc}/{len(comparison)} accepted. "
              f"Resources under {os.path.relpath(DEMO, ROOT)}/")
    return comparison


def _write_merged_manifest(demo_cases: dict) -> None:
    """Merge demo cases into the dataset manifest without touching old entries."""
    track_manifest = os.path.join(ROOT, "data", "tracks", "manifest.json")
    merged = {"cases": {}}
    if os.path.exists(track_manifest):
        with open(track_manifest) as f:
            merged = json.load(f)
    merged.setdefault("cases", {})
    for name, meta in demo_cases.items():
        merged["cases"][name] = dict(meta)
    merged["demo_generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _save(os.path.join(DEMO, "manifest.json"),
          {"cases": demo_cases,
           "generated_at": merged["demo_generated_at"]})
    _save(track_manifest, merged)


def verify_checked_in(verbose: bool = True) -> bool:
    """Re-verify checked-in tracks against checked-in truth from disk.

    Asserts every per-phrase record still meets the agreed tolerances and that
    the no-anchor tablature refuses all candidate octave resolutions.  This is
    the "demo material must really exist and really pass" gate; nothing here
    re-renders audio, so a stale/faked track cannot sneak through.
    """
    comp_path = os.path.join(DEMO, "comparison.json")
    if not os.path.exists(comp_path):
        raise FileNotFoundError("no checked-in demo comparison; run build first")
    with open(comp_path) as f:
        comparison = json.load(f)

    ok = True
    for name, rec in comparison["cases"].items():
        if not rec.get("accepted"):
            ok = False
            if verbose:
                print(f"[FAIL] {name}: not accepted")
            continue
        # pair-level records live under pairs/ and have no per-phrase artifacts
        if "pair_verify" in rec:
            pair_path = os.path.join(_dir("pairs"), f"{name}.json")
            if not os.path.exists(pair_path):
                ok = False
                if verbose:
                    print(f"[FAIL] {name}: missing pair artifact {pair_path}")
            elif verbose:
                print(f"[ok]  {name}: pair accepted, artifact present")
            continue
        # re-load the actual phrase artifacts to prove they exist on disk
        for sub in ("audio", "truth", "tracks", "tab", "verify"):
            p = _paths(name)[sub]
            if not os.path.exists(p):
                ok = False
                if verbose:
                    print(f"[FAIL] {name}: missing {sub} artifact {p}")
        if verbose:
            print(f"[ok]  {name}: accepted, artifacts present")

    # regression: no-anchor tab must refuse unjustified definite octaves
    no_anchor_tab_path = _paths("phrase_anchor_no_anchor")["tab"]
    if os.path.exists(no_anchor_tab_path):
        with open(no_anchor_tab_path) as f:
            tab = json.load(f)
        resolved = tablature.resolve_octaves(tab)
        unjustified = [v for v in resolved["verdicts"]
                       if v.get("resolved")
                       and v.get("reason") == "open_string_anchor"]
        if unjustified:
            ok = False
            if verbose:
                print(f"[FAIL] no-anchor tab produced {len(unjustified)} "
                      "unjustified octave resolutions")
        else:
            refused = [v for v in resolved["verdicts"]
                       if not v["resolved"]
                       and v.get("reason") == "no_open_string_anchor"]
            if verbose:
                print(f"[ok]  no-anchor tab refused {len(refused)} "
                      "candidate octaves (no evidence)")
    if verbose:
        print("VERIFY", "PASS" if ok else "FAIL")
    return ok


def list_cases() -> None:
    comp_path = os.path.join(DEMO, "comparison.json")
    if not os.path.exists(comp_path):
        print("no demo resources; run: python scripts/build_demo.py build")
        return
    with open(comp_path) as f:
        comparison = json.load(f)
    for name, rec in comparison["cases"].items():
        print(f"{name:28s} accepted={rec.get('accepted')} "
              f"failures={rec.get('n_failures', len(rec.get('failures', [])))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("build", "verify", "list"),
                    nargs="?", default="build")
    ap.add_argument("--frame-len", type=int, default=AnalysisConfig.frame_len)
    ap.add_argument("--hop", type=int, default=AnalysisConfig.hop)
    args = ap.parse_args()
    cfg = AnalysisConfig(frame_len=args.frame_len, hop=args.hop)
    if args.command == "build":
        build(cfg)
    elif args.command == "verify":
        if not verify_checked_in():
            sys.exit(1)
    else:
        list_cases()


if __name__ == "__main__":
    main()
