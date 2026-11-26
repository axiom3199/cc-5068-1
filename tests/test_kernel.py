"""Kernel correctness tests (stdlib unittest, no third-party deps).

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from intonation.config import AnalysisConfig, cents
from intonation import synth, wavio
from intonation.analysis import analyze_signal
from intonation.streaming import stream_analyze
from intonation.config import StreamingConfig


class TestHarmonicFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = AnalysisConfig()
        cls.fixtures = synth.build_harmonic_fixtures()

    def test_known_frequency_cent_error(self):
        for item in self.fixtures:
            res = analyze_signal(item["audio"], item["sr"], self.cfg)
            truth = item["truth"]
            t0, t1 = truth["tone_t0"], truth["tone_t1"]
            errs = [abs(cents(f["tracked_f0_hz"], truth["f0_hz"]))
                    for f in res["frames"]
                    if f["tracked_f0_hz"] and t0 + 0.05 <= f["time"]["t_center"] <= t1 - 0.05]
            self.assertGreater(len(errs), 5, item["name"])
            self.assertEqual(int(np.sum(np.array(errs) > 600)), 0, item["name"])
            # sub-cent on clean harmonic groups (one low-f0 fixture is ~1c)
            self.assertLess(float(np.mean(errs)), 1.5,
                            f"{item['name']} mean {np.mean(errs):.2f}c")

    def test_dominant_second_harmonic_not_octave_wrong(self):
        # fixture_dom2 has a strong 2nd harmonic; naive max-peak picking would
        # report an octave high.  Scored on the tone interval (as the published
        # evaluator does): the release tail past the tone end may decay into
        # noise and is not a reliable-pitch region.
        item = next(f for f in self.fixtures if f["name"] == "fixture_dom2")
        res = analyze_signal(item["audio"], item["sr"], self.cfg)
        truth = item["truth"]
        t0, t1 = truth["tone_t0"], truth["tone_t1"]
        errs = [abs(cents(f["tracked_f0_hz"], truth["f0_hz"]))
                for f in res["frames"]
                if f["tracked_f0_hz"] and t0 + 0.05 <= f["time"]["t_center"] <= t1 - 0.05]
        self.assertGreater(len(errs), 40)
        self.assertLess(max(errs), 600)


class TestVibrato(unittest.TestCase):
    def test_vibrato_rate_recovered(self):
        cfg = AnalysisConfig()
        for item in synth.build_vibrato_fixtures():
            res = analyze_signal(item["audio"], item["sr"], cfg)
            tr = item["truth"]
            t0, t1 = tr["tone_t0"], tr["tone_t1"]
            ts, vs = [], []
            for f in res["frames"]:
                tc = f["time"]["t_center"]
                v = f["tracked_f0_hz"]
                if v and t0 + 0.30 <= tc <= t1 - 0.06:
                    ts.append(tc); vs.append(cents(v, tr["f0_center_hz"]))
            self.assertGreater(len(ts), 30, item["name"])
            ts = np.array(ts); vs = np.array(vs)
            c = vs - vs.mean()
            zc = int(np.sum(np.diff(np.sign(c)) != 0))
            rec = (zc / 2.0) / (ts[-1] - ts[0])
            self.assertAlmostEqual(rec, tr["vibrato_hz"], delta=1.0, msg=item["name"])
            depth = (np.percentile(vs, 95) - np.percentile(vs, 5)) / 2.0
            self.assertAlmostEqual(depth, tr["vibrato_cents"], delta=0.30 * tr["vibrato_cents"],
                                   msg=item["name"])


class TestSilenceNoise(unittest.TestCase):
    def test_no_spurious_pitch_in_silence_or_noise(self):
        cfg = AnalysisConfig()
        fx = synth.build_silence_noise_fixture()
        res = analyze_signal(fx["audio"], fx["sr"], cfg)
        tr = fx["truth"]
        sil = tr["silence_regions"]; noi = tr["noise_regions"]
        bad = 0; total = 0
        for f in res["frames"]:
            fr = {"time": f["time"]}
            t = f["time"]
            contained = any(t["t_start"] >= a and t["t_end"] <= b for a, b in sil + noi)
            if contained:
                total += 1
                if f["tracked_f0_hz"] is not None:
                    bad += 1
        self.assertGreater(total, 10)
        self.assertEqual(bad, 0)


class TestTwoTake(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = AnalysisConfig()
        cls.scene = synth.build_two_take_scenario()

    def test_take_a_relabels_opening_to_low_fundamental(self):
        res = analyze_signal(self.scene["a"]["audio"], self.scene["a"]["sr"], self.cfg)
        revs = res["revisions"]
        self.assertTrue(any(r["kind"] == "octave_relabel"
                            and abs(r["from_f0_hz"] - 392.0) < 5
                            and abs(r["to_f0_hz"] - 196.0) < 5 for r in revs))
        # the shared opening frames must now read the low fundamental
        opening = [f for f in res["frames"]
                   if 0.30 <= f["time"]["t_center"] <= 1.30 and f["tracked_f0_hz"]]
        self.assertTrue(opening)
        vals = [f["tracked_f0_hz"] for f in opening]
        self.assertLess(abs(np.median(vals) - 196.0), 6)

    def test_take_b_keeps_opening_and_leaps(self):
        res = analyze_signal(self.scene["b"]["audio"], self.scene["b"]["sr"], self.cfg)
        # must NOT relabel the opening
        self.assertFalse(any(r["kind"] == "octave_relabel" for r in res["revisions"]))
        opening = [f["tracked_f0_hz"] for f in res["frames"]
                   if 0.30 <= f["time"]["t_center"] <= 1.30 and f["tracked_f0_hz"]]
        self.assertLess(abs(np.median(opening) - 392.0), 8)
        # genuine leap detected within 100 cents of 784
        jump_t = 1.4
        after = [f for f in res["frames"]
                 if f["time"]["t_center"] >= jump_t and f["tracked_f0_hz"]
                 and abs(cents(f["tracked_f0_hz"], 784.0)) <= 100]
        self.assertTrue(after)
        self.assertLess(after[0]["time"]["t_center"] - jump_t, 0.10)


class TestStreamingGate(unittest.TestCase):
    def test_strict_fails_closed_on_ambiguous_texture(self):
        sc = synth.build_two_take_scenario()
        res = stream_analyze(sc["a"]["audio"], scfg=StreamingConfig(admission_policy="strict"))
        # genuinely ambiguous even-harmonic opening => strict admits nothing
        self.assertEqual(res["admitted_frames"], 0)

    def test_best_effort_admits_at_deadline(self):
        sc = synth.build_two_take_scenario()
        res = stream_analyze(sc["a"]["audio"], scfg=StreamingConfig(admission_policy="best_effort"))
        self.assertGreater(res["admitted_frames"], 0)
        admitted = [f for f in res["frames"] if f["admitted"]]
        # every admission must cite a concrete reason, and deadlines are reported
        self.assertTrue(all(f["admission_reason"] for f in admitted))
        self.assertTrue(any(f["overdue"] for f in res["frames"]))


if __name__ == "__main__":
    unittest.main()
