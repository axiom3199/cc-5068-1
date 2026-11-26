"""Generate the synthetic dataset, run the offline analyzer, export tracks and
evaluation reports.

Outputs (all under data/):
  audio/<name>.wav           audio only (no pitch in filename)
  truth/<name>.json          ground truth, separate from audio
  tracks/<name>.json         full inspectable analysis track
  results/report.json        evaluation metrics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from intonation import synth
from intonation import wavio
from intonation.analysis import analyze_signal
from intonation.evaluate import (evaluate_two_take, jump_delay, fixture_stats,
                                 evaluate_silence_noise)
from intonation.config import AnalysisConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def _paths(name: str):
    return (
        os.path.join(DATA, "audio", f"{name}.wav"),
        os.path.join(DATA, "truth", f"{name}.json"),
        os.path.join(DATA, "tracks", f"{name}.json"),
    )


def _save_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_default)
    os.replace(tmp, path)


def _default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def run_full(cfg: AnalysisConfig, verbose: bool = True) -> dict:
    t_start = time.time()
    all_data = synth.build_all()
    report = {"config": cfg.to_dict(), "cases": {}, "generated_at": None}
    tracks_manifest = {}

    # ---- two-take scenario -------------------------------------------
    sc = all_data["two_take"]
    for take in ("a", "b"):
        name = f"two_take_{take}"
        d = sc[take]
        audio, sr, truth = d["audio"], d["sr"], d["truth"]
        wav_path, truth_path, track_path = _paths(name)
        wavio.write_wav(wav_path, audio, sr)
        _save_json(truth_path, truth)
        res = analyze_signal(audio, sr, cfg)
        res["source"] = {"name": name, "audio": os.path.relpath(wav_path, ROOT),
                         "truth": os.path.relpath(truth_path, ROOT),
                         "analysis": "offline", "allow_revision": True}
        _save_json(track_path, res)
        ev = evaluate_two_take(res, truth)
        ev["jump"] = jump_delay(res, truth)
        report["cases"][name] = ev
        tracks_manifest[name] = {
            "audio": os.path.relpath(wav_path, ROOT),
            "truth": os.path.relpath(truth_path, ROOT),
            "track": os.path.relpath(track_path, ROOT),
            "duration_s": res["audio"]["duration_s"],
        }
        if verbose:
            print(f"[{name}] coverage={ev['coverage']:.2f} "
                  f"strict_err={ev['strict_cent_stats'].get('abs_mean_cents', -1):.2f}c "
                  f"octave_err_rate={ev['strict_cent_stats'].get('octave_error_rate', -1):.3f} "
                  f"spurious={ev['spurious_rate']:.3f} "
                  f"rev_ok={ev['revision_expectation_met']} "
                  f"jump_delay={ev['jump']['delay_ms'] if ev.get('jump') and ev['jump']['delay_ms'] is not None else '-'}")

    # ---- calibration / robustness fixtures ---------------------------
    fixtures = {
        "harmonic": all_data["harmonic"],
        "vibrato": all_data["vibrato"],
        "noisy": all_data["noisy"],
    }
    for group, items in fixtures.items():
        for item in items:
            name = item["name"]
            audio, sr, truth = item["audio"], item["sr"], item["truth"]
            wav_path, truth_path, track_path = _paths(name)
            wavio.write_wav(wav_path, audio, sr)
            _save_json(truth_path, truth)
            res = analyze_signal(audio, sr, cfg)
            res["source"] = {"name": name, "audio": os.path.relpath(wav_path, ROOT),
                             "truth": os.path.relpath(truth_path, ROOT),
                             "analysis": "offline", "allow_revision": True}
            _save_json(track_path, res)
            st = fixture_stats(res, truth)
            report["cases"][name] = st
            tracks_manifest[name] = {
                "audio": os.path.relpath(wav_path, ROOT),
                "truth": os.path.relpath(truth_path, ROOT),
                "track": os.path.relpath(track_path, ROOT),
                "duration_s": res["audio"]["duration_s"],
            }
            if verbose:
                if st.get("kind") == "vibrato":
                    print(f"[{name}] coverage={st.get('coverage', 0):.2f} "
                          f"track_abs_mean={st.get('tracking_abs_mean_cents', -1):.2f}c "
                          f"p95={st.get('tracking_p95_cents', -1):.2f}c "
                          f"vib_ref={st.get('reference_vibrato_hz')}Hz/"
                          f"{st.get('reference_vibrato_cents')}c "
                          f"vib_rec={st.get('recovered_vibrato_hz'):.2f}Hz/"
                          f"{(st.get('recovered_vibrato_cents_p5_p95') or 0):.1f}c")
                else:
                    print(f"[{name}] coverage={st.get('coverage', 0):.2f} "
                          f"abs_mean={st.get('abs_mean_cents', -1):.2f}c "
                          f"p95={st.get('p95_abs_cents', -1):.2f}c "
                          f"oct_err={st.get('octave_error_rate', -1):.3f}")

    # ---- silence / noise fixture ------------------------------------
    sn = all_data["silence_noise"]
    name = sn["name"]
    audio, sr, truth = sn["audio"], sn["sr"], sn["truth"]
    wav_path, truth_path, track_path = _paths(name)
    wavio.write_wav(wav_path, audio, sr)
    _save_json(truth_path, truth)
    res = analyze_signal(audio, sr, cfg)
    res["source"] = {"name": name, "audio": os.path.relpath(wav_path, ROOT),
                     "truth": os.path.relpath(truth_path, ROOT),
                     "analysis": "offline", "allow_revision": True}
    _save_json(track_path, res)
    ev = evaluate_silence_noise(res, truth)
    report["cases"][name] = ev
    tracks_manifest[name] = {
        "audio": os.path.relpath(wav_path, ROOT),
        "truth": os.path.relpath(truth_path, ROOT),
        "track": os.path.relpath(track_path, ROOT),
        "duration_s": res["audio"]["duration_s"],
    }
    if verbose:
        print(f"[{name}] silence_spurious={ev['silence_rate']:.3f} "
              f"noise_spurious={ev['noise_rate']:.3f}")

    elapsed = time.time() - t_start
    report["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report["elapsed_s"] = round(elapsed, 2)
    _save_json(os.path.join(DATA, "results", "report.json"), report)
    _save_json(os.path.join(DATA, "tracks", "manifest.json"),
               {"cases": tracks_manifest, "generated_at": report["generated_at"]})
    if verbose:
        print(f"\nDone in {elapsed:.1f}s. Report: data/results/report.json")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame-len", type=int, default=AnalysisConfig.frame_len)
    ap.add_argument("--hop", type=int, default=AnalysisConfig.hop)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    cfg = AnalysisConfig(frame_len=args.frame_len, hop=args.hop)
    run_full(cfg, verbose=not args.quiet)


if __name__ == "__main__":
    main()
