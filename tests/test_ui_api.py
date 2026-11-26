"""UI-facing module + HTTP API tests (stdlib unittest).

Starts the real server in a background thread on an ephemeral port and drives
the JSON API exactly as the browser does, so the tested path is the shipped
path.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import sys
import threading
import unittest
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from intonation import score as score_mod, accompaniment, feedback, synth
from intonation.analysis import analyze_signal
from intonation.config import AnalysisConfig


def _http(method, url, data=None, headers=None):
    req = urllib.request.Request(url, data=data, method=method,
                                 headers=headers or {})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read()
        ctype = r.headers.get("Content-Type", "")
        if "json" in ctype:
            return json.loads(body)
        return body


class TestScoreAndAccompaniment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sc = synth.build_two_take_scenario()
        cls.res = analyze_signal(sc["a"]["audio"], sc["a"]["sr"], AnalysisConfig())
        cls.res["mode"] = "offline"

    def test_score_relabels_are_scores(self):
        s = score_mod.build_score(self.res)
        note_idx = [i for i, n in enumerate(s["notes"]) if n["kind"] == "note"]
        self.assertTrue(note_idx)
        # the opening was relabelled; the score must carry that and show G3
        self.assertEqual(s["notes"][note_idx[0]]["pitch_name"], "G3")
        self.assertTrue(s["notes"][note_idx[0]]["octave_corrected"])

    def test_octave_edit_changes_rendered_accompaniment(self):
        s = score_mod.build_score(self.res)
        note_idx = next(i for i, n in enumerate(s["notes"]) if n["kind"] == "note")
        edited = score_mod.apply_edits(s, [{"action": "octave", "index": note_idx, "dir": 1}])
        rendered = accompaniment.render_accompaniment(
            self.res, self.res["audio"]["sample_rate"], follow_score=edited)
        self.assertEqual(rendered["n_spans"], 1)
        self.assertAlmostEqual(rendered["spans"][0]["f0"], 392.0, delta=6)

    def test_musicxml_and_midi_are_valid_containers(self):
        s = score_mod.build_score(self.res)
        xml = score_mod.to_musicxml(s)
        self.assertIn("<score-partwise", xml)
        self.assertIn("<pitch>", xml)
        mid = score_mod.to_midi_bytes(s)
        self.assertEqual(mid[:4], b"MThd")
        self.assertIn(b"MTrk", mid)

    def test_feedback_classifies_octave_override(self):
        s = score_mod.build_score(self.res)
        note_idx = next(i for i, n in enumerate(s["notes"]) if n["kind"] == "note")
        edited = score_mod.apply_edits(s, [{"action": "octave", "index": note_idx, "dir": 1}])
        records = feedback.classify_edits(s, edited, self.res)
        self.assertTrue(any(r["kind"] == "octave_failure" for r in records))


class TestServerAPI(unittest.TestCase):
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

    def test_cases_and_track(self):
        cases = _http("GET", self.base + "/api/cases")
        self.assertIn("two_take_a", [c["id"] for c in cases["cases"]])
        tr = _http("GET", self.base + "/api/track/two_take_a")
        self.assertEqual(len(tr["frames"]), len(tr["frames"]))
        self.assertTrue(len(tr["frames"]) > 100)
        self.assertIn("candidates", tr["frames"][40])

    def test_streaming_endpoint_policies(self):
        strict = _http("GET", self.base + "/api/stream/two_take_a?policy=strict")
        self.assertEqual(strict["admitted_frames"], 0)
        best = _http("GET", self.base + "/api/stream/two_take_a?policy=best_effort")
        self.assertGreater(best["admitted_frames"], 0)

    def test_audio_bytes_are_wav(self):
        raw = _http("GET", self.base + "/api/audio/two_take_a")
        self.assertEqual(raw[:4], b"RIFF")
        self.assertEqual(raw[8:12], b"WAVE")

    def test_index_served(self):
        raw = _http("GET", self.base + "/")
        self.assertIn(b"<canvas", raw)
        self.assertIn(b'app.js', raw)

    def test_score_edit_round_trip_returns_audio_and_exports(self):
        sc = _http("POST", self.base + "/api/score",
                   data=json.dumps({"id": "two_take_a", "tempo_bpm": 100}).encode(),
                   headers={"Content-Type": "application/json"})
        note_idx = next(i for i, n in enumerate(sc["score"]["notes"]) if n["kind"] == "note")
        res = _http("POST", self.base + "/api/score/edit",
                    data=json.dumps({"id": "two_take_a",
                                     "edits": [{"action": "octave",
                                                "index": note_idx, "dir": 1}]}).encode(),
                    headers={"Content-Type": "application/json"})
        self.assertIn("score", res)
        self.assertTrue(res["audio_wav_b64"])
        self.assertTrue(any(r["kind"] == "octave_failure" for r in res["feedback"]))
        self.assertIn("exports", res)

    def test_upload_analyze(self):
        sc = synth.build_two_take_scenario()
        import io, wave
        buf = io.BytesIO()
        ints = np.round(np.clip(sc["a"]["audio"], -1, 1) * 32767).astype("<i2")
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sc["a"]["sr"])
            w.writeframes(ints.tobytes())
        raw = buf.getvalue()
        out = _http("POST", self.base + "/api/analyze", data=raw,
                    headers={"Content-Type": "application/octet-stream",
                             "X-Filename": "melody.wav"})
        self.assertIn("id", out)
        self.assertTrue(out["id"].startswith("up_"))
        self.assertGreater(len(out["track"]["frames"]), 10)


if __name__ == "__main__":
    unittest.main()
