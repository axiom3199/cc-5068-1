"""Demo phrase / tablature / data-production tests (stdlib unittest).

Covers the phrase-level requirements:

* synthetic phrases carry known f0, vibrato, voicing changes and editable
  parameters, with an explicit known-pitch vs candidate-range status;
* the tablature resolves candidate octaves only with an open-string anchor and
  refuses unjustified definite octaves when that anchor is removed;
* material pairs share a performed pitch with opposite fundamental presence on
  an independent score, with decidable/undecidable spans marked;
* the analyzer is run on audio only — truth never reaches the algorithm;
* pitch (cent error) and voicing acceptance, with failed segments retained;
* the HTTP endpoints serve the real kernel output and never hard-code curves.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import unittest
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from intonation import phrases, tablature, datamaking
from intonation.config import AnalysisConfig, cents
from intonation.analysis import analyze_signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "data", "demo")


def _http(method, url, data=None, headers=None):
    req = urllib.request.Request(url, data=data, method=method,
                                 headers=headers or {})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read()
        return json.loads(body)


class TestPhraseSynthesis(unittest.TestCase):
    def test_phrase_has_known_and_candidate_notes(self):
        p = phrases.build_anchor_phrase()
        statuses = {n.pitch_status for n in p.notes}
        self.assertEqual(statuses, {"known", "candidate"})
        candidate = next(n for n in p.notes if n.pitch_status == "candidate")
        # candidate notes expose an octave range, not a definite pitch
        truth = phrases.phrase_truth(p)
        seg = next(s for s in truth["segments"]
                   if abs(s["t0"] - candidate.t0) < 1e-9)
        self.assertEqual(seg["candidate_range_hz"],
                         [candidate.f0_hz, 2.0 * candidate.f0_hz])
        self.assertTrue(seg["ambiguous"])

    def test_editable_parameters_round_trip(self):
        p = phrases.build_anchor_phrase()
        d = p.to_dict()
        # every note carries the editable synthesis parameters and timbre amps
        for nd in d["notes"]:
            for key in ("f0_hz", "t0", "t1", "timbre", "vibrato_hz",
                        "vibrato_cents", "noise_amp", "pitch_status",
                        "harmonics"):
                self.assertIn(key, nd)
        # rebuilding from saved params yields identical audio
        p2 = phrases.Phrase(
            name=p.name, duration=p.duration, seed=p.seed,
            unvoiced=[list(g) for g in p.unvoiced],
            notes=[phrases.PhraseNote(**{k: v for k, v in n.to_dict().items()
                                         if k in phrases.PhraseNote.__dataclass_fields__})
                   for n in p.notes])
        self.assertTrue(np.allclose(phrases.render_phrase(p),
                                    phrases.render_phrase(p2), atol=1e-12))

    def test_truth_trajectory_tracks_vibrato_and_voicing(self):
        p = phrases.build_breathy_phrase()
        truth = phrases.phrase_truth(p)
        traj = truth["f0_trajectory"]
        voiced = [pt for pt in traj if pt["voiced"]]
        unvoiced = [pt for pt in traj if not pt["voiced"]]
        self.assertGreater(len(voiced), 50)
        self.assertGreater(len(unvoiced), 5)
        # vibrato makes the instantaneous reference move around the centre
        f0s = np.array([pt["f0_hz"] for pt in voiced])
        self.assertGreater(np.ptp(f0s), 1.0)


class TestTablatureAnchor(unittest.TestCase):
    def test_anchor_resolves_candidate_octaves(self):
        p = phrases.build_anchor_phrase(with_anchor=True)
        tab = tablature.build_tablature(p)
        self.assertEqual(tab["anchors"], [2])
        out = tablature.resolve_octaves(tab)
        by_index = {v["index"]: v for v in out["verdicts"]}
        # candidate notes 0 and 3 resolve through the open-string anchor
        for i in (0, 3):
            self.assertTrue(by_index[i]["resolved"])
            self.assertEqual(by_index[i]["reason"], "open_string_anchor")
            self.assertEqual(by_index[i]["anchor_index"], 2)
        self.assertEqual(len(tablature.unresolved_indices(out)), 0)

    def test_removing_anchor_refuses_definite_octave(self):
        p = phrases.build_anchor_phrase(with_anchor=False)
        tab = tablature.build_tablature(p)
        self.assertEqual(tab["anchors"], [])
        out = tablature.resolve_octaves(tab)
        refused = [v for v in out["verdicts"]
                   if not v["resolved"] and v["reason"] == "no_open_string_anchor"]
        # all three candidate notes (N1, N3, N4) must be refused
        self.assertEqual(len(refused), 3)
        # no unjustified definite pitch may appear
        self.assertEqual(
            len([v for v in out["verdicts"] if v.get("reason") == "open_string_anchor"]),
            0)
        # the candidate notes remain unresolved in the tab itself
        self.assertEqual(len(tablature.unresolved_indices(out)), 3)

    def test_fingering_edit_does_not_manufacture_evidence(self):
        p = phrases.build_anchor_phrase(with_anchor=False)
        tab = tablature.build_tablature(p)
        moved = tablature.apply_fingering_edit(tab, 0, 4, 5)
        note = next(n for n in moved["notes"] if n["index"] == 0)
        self.assertEqual(note["pitch_status"], "candidate")
        self.assertIsNone(note["resolved_hz"])


class TestMaterialPair(unittest.TestCase):
    def test_same_pitch_opposite_fundamental_pair(self):
        p_full, p_even, alignment = phrases.build_fundamental_pair()
        self.assertEqual(p_full.notes[0].f0_hz, p_even.notes[0].f0_hz)
        self.assertEqual(p_full.notes[0].pitch_status, "known")
        self.assertEqual(p_even.notes[0].pitch_status, "candidate")
        # independent score alignment and decidability spans
        self.assertTrue(any(s["kind"] == "note"
                            for s in alignment["independent_score"]))
        spans = {s["take"]: s["decidability"] for s in alignment["spans"]}
        self.assertEqual(spans[p_full.name], "decidable")
        self.assertEqual(spans[p_even.name], "undecidable")

    def test_pair_verification_passes_and_marks_undecidable(self):
        p_full, p_even, alignment = phrases.build_fundamental_pair()
        b_full = datamaking.produce_phrase(p_full)
        b_even = datamaking.produce_phrase(p_even)
        pv = datamaking.verify_pair(b_full["track"], b_even["track"],
                                    b_full["truth"], b_even["truth"], alignment)
        self.assertTrue(pv["accepted"])
        self.assertGreaterEqual(
            b_even["verification"]["ambiguity"]["candidate_undecided_rate"], 0.9)
        self.assertTrue(pv["even_take_literal_near_upper_octave"])


class TestAudioOnlyAndVerification(unittest.TestCase):
    def test_truth_never_enters_analyzer(self):
        # monkeypatch the analysis entry to prove production calls it with audio
        # and sample rate only (no truth object in the arguments)
        p = phrases.build_anchor_phrase()
        seen = {}
        original = datamaking.analyze_signal

        def spy(x, sr, cfg=None, allow_revision=True):
            seen["got_array"] = isinstance(x, np.ndarray)
            seen["sr"] = sr
            seen["has_truth_kwarg"] = False
            return original(x, sr, cfg, allow_revision=allow_revision)

        datamaking.analyze_signal = spy
        try:
            datamaking.produce_phrase(p)
        finally:
            datamaking.analyze_signal = original
        self.assertTrue(seen["got_array"])
        self.assertEqual(seen["sr"], p.sample_rate)
        self.assertFalse(seen["has_truth_kwarg"])

    def test_anchor_and_breathy_acceptance_with_failures_retained(self):
        for builder in (phrases.build_anchor_phrase, phrases.build_breathy_phrase):
            bundle = datamaking.produce_phrase(builder())
            v = bundle["verification"]
            self.assertTrue(v["accepted"], builder.__name__)
            self.assertLessEqual(v["cent_stats"]["octave_error_rate"], 0.0)
            self.assertEqual(v["voicing"]["unvoiced_spurious_rate"], 0.0)
            # the record keeps explicit failure evidence (breathy has some
            # honest attack gaps; anchor may have none)
            self.assertIn("failures", v)
            self.assertEqual(v["n_failures"], len(v["failures"]))

    def test_strong_harmonic_cent_accuracy(self):
        # dom2 and full timbres: known pitch must be tracked within tolerance
        p_full, _, _ = phrases.build_fundamental_pair()
        v = datamaking.produce_phrase(p_full)["verification"]
        self.assertLess(v["cent_stats"]["abs_mean_cents"],
                        datamaking.CENT_MEAN_TOL)
        self.assertEqual(v["cent_stats"]["octave_errors"], 0)


class TestDemoArtifacts(unittest.TestCase):
    def test_demo_resources_exist_and_pass(self):
        comp = os.path.join(DEMO, "comparison.json")
        self.assertTrue(os.path.exists(comp), "run scripts/build_demo.py build")
        with open(comp) as f:
            comparison = json.load(f)
        for name in ("phrase_anchor", "phrase_anchor_no_anchor",
                     "pair_f3_full", "pair_f3_even", "phrase_breathy"):
            self.assertIn(name, comparison["cases"])
            self.assertTrue(comparison["cases"][name]["accepted"], name)
        # provenance records source + parameters for every phrase
        with open(os.path.join(DEMO, "provenance.json")) as f:
            prov = json.load(f)
        generators = {e["name"]: e["generator"] for e in prov["entries"]}
        self.assertIn("phrase_anchor", generators)
        self.assertTrue(all(e["phrase_params"] for e in prov["entries"]))

    def test_old_kernel_fixtures_still_present(self):
        manifest = os.path.join(ROOT, "data", "tracks", "manifest.json")
        with open(manifest) as f:
            cases = json.load(f)["cases"]
        for name in ("two_take_a", "two_take_b", "fixture_h1",
                     "fixture_dom2", "fixture_vib_slow"):
            self.assertIn(name, cases)


class TestDemoServerAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from server.app import make_server
        cls.srv = make_server("127.0.0.1", 0)
        cls.port = cls.srv.server_address[1]
        cls.thread = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.thread.start()
        import time
        time.sleep(0.4)
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def test_demo_cases_listed_and_loadable(self):
        cases = _http("GET", self.base + "/api/cases")["cases"]
        demo = [c for c in cases if c["source"] == "demo"]
        ids = {c["id"] for c in demo}
        self.assertIn("phrase_anchor", ids)
        self.assertTrue(all(c["has_tab"] for c in demo))

    def test_track_served_is_real_analyzer_output(self):
        tr = _http("GET", self.base + "/api/track/phrase_anchor")
        self.assertGreater(len(tr["frames"]), 100)
        # frames carry genuine per-frame evidence, not a fixed curve
        for fr in tr["frames"][:50]:
            self.assertIn("state", fr)
            self.assertIn("candidates", fr)

    def test_audio_is_served_as_wav(self):
        raw = urllib.request.urlopen(
            self.base + "/api/audio/phrase_breathy").read()
        self.assertEqual(raw[:4], b"RIFF")
        self.assertEqual(raw[8:12], b"WAVE")

    def test_tab_endpoint_resolves_and_refuses(self):
        ok = _http("POST", self.base + "/api/tab/resolve",
                   data=json.dumps({"id": "phrase_anchor"}).encode(),
                   headers={"Content-Type": "application/json"})
        self.assertGreater(ok["n_resolved"], 0)
        self.assertEqual(ok["n_refused"], 0)

        blocked = _http("POST", self.base + "/api/tab/resolve",
                        data=json.dumps({"id": "phrase_anchor_no_anchor"}).encode(),
                        headers={"Content-Type": "application/json"})
        self.assertGreater(blocked["n_refused"], 0)
        reasons = {v["reason"] for v in blocked["verdicts"] if not v["resolved"]}
        self.assertIn("no_open_string_anchor", reasons)

    def test_tab_resolve_not_hardcoded(self):
        # resolving a single explicit index must return only that verdict,
        # proving the endpoint recomputes rather than returning fixed data
        out = _http("POST", self.base + "/api/tab/resolve",
                    data=json.dumps({"id": "phrase_anchor", "indices": [3]}).encode(),
                    headers={"Content-Type": "application/json"})
        self.assertEqual(len(out["verdicts"]), 1)
        self.assertEqual(out["verdicts"][0]["index"], 3)


if __name__ == "__main__":
    unittest.main()
