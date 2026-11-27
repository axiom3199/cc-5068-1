"""HTTP server that exposes the intonation kernel to the browser UI.

No third-party dependencies: the standard library serves static files and a
small JSON API.  Every analysis decision shown in the UI comes from the
kernel, and every override the user makes on the score is sent back here as
failure feedback (see :mod:`intonation.feedback`).

Run with::

    python -m server.app --port 8000
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import sys
import threading
import wave
import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from intonation import wavio, accompaniment, score as score_mod, feedback  # noqa: E402
from intonation.analysis import analyze_signal  # noqa: E402
from intonation.streaming import stream_analyze  # noqa: E402
from intonation.config import AnalysisConfig, StreamingConfig  # noqa: E402
from intonation import tablature as tab_mod  # noqa: E402

DATA = os.path.join(ROOT, "data")
AUDIO_DIR = os.path.join(DATA, "audio")
TRACK_DIR = os.path.join(DATA, "tracks")
TRUTH_DIR = os.path.join(DATA, "truth")
UPLOAD_DIR = os.path.join(DATA, "uploads")
EXPORT_DIR = os.path.join(DATA, "exports")
FEEDBACK_DIR = os.path.join(DATA, "feedback")
DEMO_DIR = os.path.join(DATA, "demo")
WEB_DIR = os.path.join(ROOT, "web")

_CACHE_LOCK = threading.Lock()
_TRACK_CACHE: dict[str, dict] = {}
_STREAM_CACHE: dict[str, dict] = {}
_UPLOADS: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# dataset registry
# ---------------------------------------------------------------------------

def _manifest() -> dict:
    path = os.path.join(TRACK_DIR, "manifest.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {"cases": {}}


def _truth_kind(name: str) -> str:
    path = os.path.join(TRUTH_DIR, f"{name}.json")
    if not os.path.exists(path):
        return "unknown"
    try:
        with open(path) as f:
            return json.load(f).get("kind", "unknown")
    except Exception:
        return "unknown"


def list_cases() -> list[dict]:
    man = _manifest()
    cases = []
    for name, meta in man.get("cases", {}).items():
        is_demo = bool(meta.get("demo"))
        cases.append({
            "id": name, "name": name,
            "duration_s": meta.get("duration_s"),
            "kind": ("demo_" + _demo_subkind(name)) if is_demo
                    else _truth_kind(name),
            "audio_url": f"/api/audio/{name}",
            "has_truth": os.path.exists(os.path.join(TRUTH_DIR, f"{name}.json"))
                         or _demo_path(name, "truth") is not None,
            "has_tab": _demo_path(name, "tab") is not None,
            "source": "demo" if is_demo else "dataset",
        })
    with _CACHE_LOCK:
        for uid, up in _UPLOADS.items():
            cases.append({
                "id": uid, "name": up["name"], "duration_s": up["duration_s"],
                "kind": "upload", "audio_url": f"/api/audio/{uid}",
                "has_truth": False, "has_tab": False, "source": "upload",
            })
    return sorted(cases, key=lambda c: (c["source"] == "upload",
                                        c["source"] != "demo", c["id"]))


def _demo_subkind(name: str) -> str:
    if name.startswith("pair_"):
        return "pair"
    if "breathy" in name:
        return "breathy"
    if "no_anchor" in name:
        return "anchor_no_anchor"
    return "anchor"


def resolve_audio(case_id: str) -> tuple[np.ndarray, int] | None:
    with _CACHE_LOCK:
        if case_id in _UPLOADS:
            path = _UPLOADS[case_id]["path"]
            return wavio.read_wav(path)
    demo = resolve_demo_audio(case_id)
    if demo is not None:
        return demo
    path = os.path.join(AUDIO_DIR, f"{case_id}.wav")
    if os.path.exists(path):
        return wavio.read_wav(path)
    return None


def _audio_path(case_id: str) -> str | None:
    with _CACHE_LOCK:
        if case_id in _UPLOADS:
            return _UPLOADS[case_id]["path"]
    p = os.path.join(DEMO_DIR, "audio", f"{case_id}.wav")
    if os.path.exists(p):
        return p
    p = os.path.join(AUDIO_DIR, f"{case_id}.wav")
    return p if os.path.exists(p) else None


def get_truth(case_id: str) -> dict | None:
    path = os.path.join(TRUTH_DIR, f"{case_id}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _demo_path(case_id: str, sub: str) -> str | None:
    """Return a checked-in demo artifact path for a demo case, else None."""
    p = os.path.join(DEMO_DIR, sub, f"{case_id}.json")
    return p if os.path.exists(p) else None


def get_demo_tab(case_id: str) -> dict | None:
    """Load the editable tablature for a demo phrase (checked-in artifact)."""
    p = _demo_path(case_id, "tab")
    if p is None:
        return None
    with open(p) as f:
        return json.load(f)


def get_demo_verify(case_id: str) -> dict | None:
    p = _demo_path(case_id, "verify")
    if p is None:
        return None
    with open(p) as f:
        return json.load(f)


def resolve_demo_audio(case_id: str) -> tuple[np.ndarray, int] | None:
    """Demo audio lives under data/demo/audio (separate from kernel fixtures)."""
    p = os.path.join(DEMO_DIR, "audio", f"{case_id}.wav")
    if os.path.exists(p):
        return wavio.read_wav(p)
    return None


def get_track(case_id: str, fresh: bool = False, allow_revision: bool = True) -> dict | None:
    """Return an offline track: cached if fresh, else from disk / on-demand."""
    cache_key = f"{case_id}:rev={allow_revision}"
    with _CACHE_LOCK:
        if not fresh and cache_key in _TRACK_CACHE:
            return _TRACK_CACHE[cache_key]
    if not fresh and allow_revision:
        demo_track = _demo_path(case_id, "tracks")
        disk = demo_track or os.path.join(TRACK_DIR, f"{case_id}.json")
        if os.path.exists(disk):
            with open(disk) as f:
                tr = json.load(f)
            tr["mode"] = "offline"
            tr["demo"] = True
            with _CACHE_LOCK:
                _TRACK_CACHE[cache_key] = tr
            return tr
    au = resolve_audio(case_id)
    if au is None:
        return None
    x, sr = au
    tr = analyze_signal(x, sr, AnalysisConfig(), allow_revision=allow_revision)
    tr["mode"] = "offline"
    tr["allow_revision"] = allow_revision
    with _CACHE_LOCK:
        _TRACK_CACHE[cache_key] = tr
    return tr


def get_stream(case_id: str, policy: str = "strict") -> dict | None:
    key = f"{case_id}:{policy}"
    with _CACHE_LOCK:
        if key in _STREAM_CACHE:
            return _STREAM_CACHE[key]
    au = resolve_audio(case_id)
    if au is None:
        return None
    x, sr = au
    scfg = StreamingConfig(admission_policy=policy, tempo_bpm=100.0)
    res = stream_analyze(x, scfg=scfg, block_size=256)
    res["mode"] = "streaming"
    res["policy"] = policy
    with _CACHE_LOCK:
        _STREAM_CACHE[key] = res
    return res


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def wav_bytes(x: np.ndarray, sr: int) -> bytes:
    x = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    ints = np.round(x * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(ints.tobytes())
    return buf.getvalue()


def _trim_track(track: dict) -> dict:
    """Drop the heaviest per-frame fields the curve view does not need.

    Candidates and peaks are retained (they are the inspectable evidence) but
    the onset flux array and duplicate keys are dropped.
    """
    out = dict(track)
    out.pop("onset", None)
    frames = []
    for fr in out.get("frames", []):
        if "time" in fr:
            t = fr["time"]
            slim = {
                "index": fr["index"],
                "t_center": t["t_center"],
                "t_start": t["t_start"],
                "t_end": t["t_end"],
                "sample_start": t["sample_start"],
                "state": fr["state"],
                "f0_hz": fr["f0_hz"],
                "tracked_f0_hz": fr.get("tracked_f0_hz"),
                "confidence": fr["confidence"],
                "yin_f0_hz": fr.get("yin", {}).get("f0_hz"),
                "yin_aperiodicity": fr.get("yin", {}).get("aperiodicity"),
                "ambiguous": fr.get("octave", {}).get("ambiguous", False),
                "lower_f0_hz": fr.get("octave", {}).get("lower_f0_hz"),
                "upper_f0_hz": fr.get("octave", {}).get("upper_f0_hz"),
                "lower_ratio": fr.get("octave", {}).get("lower_ratio"),
                "attack": fr.get("attack", {}).get("onset", False),
                "decision": fr.get("decision"),
                "octave_corrected": fr.get("octave_corrected", False),
                "candidates": fr.get("candidates", []),
                "peaks": fr.get("peaks", []),
            }
        else:
            slim = fr  # already-streaming-shaped
        frames.append(slim)
    out["frames"] = frames
    return out


# ---------------------------------------------------------------------------
# request handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter default logging
        if os.environ.get("INTONATION_VERBOSE") == "1":
            super().log_message(fmt, *args)

    # -- low-level responders ------------------------------------------------
    def _send(self, payload: bytes, status: int = 200, ctype: str = "application/json",
              extra: dict | None = None, compress: bool = True):
        headers = {"Content-Type": ctype,
                   "Access-Control-Allow-Origin": "*",
                   "Cache-Control": "no-cache"}
        if extra:
            headers.update(extra)
        body = payload
        if compress and "gzip" in self.headers.get("Accept-Encoding", "") and len(payload) > 4096:
            body = gzip.compress(payload, compresslevel=6)
            headers["Content-Encoding"] = "gzip"
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, status: int = 200):
        self._send(json.dumps(obj, default=_json_default).encode("utf-8"),
                   status=status, ctype="application/json")

    def _err(self, msg: str, status: int = 400):
        self._json({"error": msg}, status=status)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0") or 0)
        return self.rfile.read(length) if length else b""

    def _read_json(self) -> dict:
        raw = self._read_body()
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    # -- routing -------------------------------------------------------------
    def do_GET(self):
        url = urlparse(self.path)
        path = unquote(url.path)
        try:
            if path in ("/", "/index.html"):
                return self._serve_file(os.path.join(WEB_DIR, "index.html"))
            if path.startswith("/static/"):
                return self._serve_file(os.path.join(WEB_DIR, path[len("/static/"):]))
            if path == "/api/cases":
                return self._json({"cases": list_cases()})
            if path.startswith("/api/track/"):
                return self._get_track(path[len("/api/track/"):], url.query)
            if path.startswith("/api/stream/"):
                return self._get_stream(path[len("/api/stream/"):], url.query)
            if path.startswith("/api/audio/"):
                return self._get_audio(path[len("/api/audio/"):])
            if path == "/api/report":
                return self._get_report()
            if path.startswith("/api/truth/"):
                return self._get_truth(path[len("/api/truth/"):])
            if path.startswith("/api/feedback/"):
                return self._get_feedback(path[len("/api/feedback/"):])
            if path.startswith("/api/tab/"):
                return self._get_tab(path[len("/api/tab/"):])
            if path.startswith("/api/verify/"):
                return self._get_verify(path[len("/api/verify/"):])
            if path.startswith("/api/export/"):
                return self._get_export(path)
            return self._err("not found", 404)
        except Exception as e:  # pragma: no cover - defensive
            import traceback
            traceback.print_exc()
            return self._err(f"{type(e).__name__}: {e}", 500)

    def do_POST(self):
        url = urlparse(self.path)
        path = unquote(url.path)
        try:
            if path == "/api/analyze":
                return self._post_analyze()
            if path == "/api/stream":
                return self._post_stream()
            if path == "/api/accompaniment":
                return self._post_accompaniment()
            if path == "/api/score":
                return self._post_score()
            if path == "/api/score/edit":
                return self._post_score_edit()
            if path == "/api/tab/resolve":
                return self._post_tab_resolve()
            if path == "/api/tab/fingering":
                return self._post_tab_fingering()
            return self._err("not found", 404)
        except Exception as e:  # pragma: no cover - defensive
            import traceback
            traceback.print_exc()
            return self._err(f"{type(e).__name__}: {e}", 500)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Filename")
        self.send_header("Content-Length", "0")
        self.end_headers()

    # -- endpoints -----------------------------------------------------------
    def _serve_file(self, full: str):
        if not os.path.isfile(full):
            return self._err("not found", 404)
        ctype = "application/octet-stream"
        if full.endswith(".html"):
            ctype = "text/html; charset=utf-8"
        elif full.endswith(".js"):
            ctype = "text/javascript; charset=utf-8"
        elif full.endswith(".css"):
            ctype = "text/css; charset=utf-8"
        elif full.endswith(".svg"):
            ctype = "image/svg+xml"
        with open(full, "rb") as f:
            data = f.read()
        self._send(data, ctype=ctype, compress=full.endswith((".js", ".css", ".svg", ".html")))

    def _get_track(self, case_id: str, query: str):
        case_id = case_id.strip("/")
        q = parse_qs(query)
        fresh = q.get("fresh", ["0"])[0] in ("1", "true")
        rev = q.get("revision", ["1"])[0] not in ("0", "false")
        tr = get_track(case_id, fresh=fresh, allow_revision=rev)
        if tr is None:
            return self._err("unknown case", 404)
        return self._json(_trim_track(tr))

    def _get_stream(self, case_id: str, query: str):
        case_id = case_id.strip("/")
        q = parse_qs(query)
        policy = q.get("policy", ["strict"])[0]
        if policy not in ("strict", "best_effort"):
            policy = "strict"
        res = get_stream(case_id, policy)
        if res is None:
            return self._err("unknown case", 404)
        return self._json(res)

    def _get_audio(self, name: str):
        name = name.strip("/")
        if name.endswith(".wav"):
            name = name[:-4]
        path = _audio_path(name)
        if path is None:
            return self._err("unknown audio", 404)
        self._serve_bytes_with_ranges(path, "audio/wav")

    def _serve_bytes_with_ranges(self, path: str, ctype: str):
        """Serve a file with HTTP Range support so the browser can seek.

        Without Accept-Ranges + 206 responses a media element cannot seek, so
        click-to-audition (which seeks to a frame's time) would silently fail.
        """
        file_size = os.path.getsize(path)
        range_header = self.headers.get("Range")
        with open(path, "rb") as f:
            if range_header and range_header.startswith("bytes="):
                try:
                    start_s, _, end_s = range_header[len("bytes="):].partition("-")
                    if start_s == "":
                        # suffix range: last N bytes
                        n = int(end_s)
                        start = max(0, file_size - n)
                        end = file_size - 1
                    else:
                        start = int(start_s)
                        end = int(end_s) if end_s else file_size - 1
                    end = min(end, file_size - 1)
                    if start > end:
                        raise ValueError
                except Exception:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{file_size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                f.seek(start)
                chunk = f.read(end - start + 1)
                self.send_response(206)
                self.send_header("Content-Type", ctype)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(len(chunk)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _get_report(self):
        p = os.path.join(DATA, "results", "report.json")
        if not os.path.exists(p):
            return self._err("no report", 404)
        with open(p) as f:
            return self._json(json.load(f))

    def _get_truth(self, case_id: str):
        case_id = case_id.strip("/")
        tr = get_truth(case_id)
        if tr is None:
            return self._err("no truth", 404)
        return self._json(tr)

    def _get_feedback(self, case_id: str):
        case_id = case_id.strip("/")
        p = os.path.join(FEEDBACK_DIR, f"{case_id}.json")
        if not os.path.exists(p):
            return self._json({"case": case_id, "entries": [], "last": None})
        with open(p) as f:
            return self._json(json.load(f))

    def _get_tab(self, case_id: str):
        case_id = case_id.strip("/")
        tab = get_demo_tab(case_id)
        if tab is None:
            return self._err("no tablature for case (not a demo phrase)", 404)
        verify = get_demo_verify(case_id)
        return self._json({"tab": tab,
                           "verify_summary": None if verify is None else
                           {"accepted": verify["accepted"],
                            "n_failures": verify["n_failures"],
                            "cent_stats": verify["cent_stats"],
                            "voicing": verify["voicing"],
                            "ambiguity": verify["ambiguity"]}})

    def _get_verify(self, case_id: str):
        case_id = case_id.strip("/")
        verify = get_demo_verify(case_id)
        if verify is None:
            return self._err("no verification record for case", 404)
        return self._json(verify)

    def _post_tab_resolve(self):
        """Resolve candidate octaves via the real tablature implementation.

        No fixed pitch data is returned: the result is recomputed by
        :func:`intonation.tablature.resolve_octaves` from the checked-in tab
        each call, and unjustified resolutions come back refused.
        """
        body = self._read_json()
        case_id = body.get("id")
        tab = get_demo_tab(case_id)
        if tab is None:
            return self._err("no tablature for case (not a demo phrase)", 404)
        indices = body.get("indices")
        resolved = tab_mod.resolve_octaves(tab, indices)
        refused = [v for v in resolved["verdicts"] if not v["resolved"]]
        return self._json({
            "tab": resolved,
            "verdicts": resolved["verdicts"],
            "n_resolved": sum(1 for v in resolved["verdicts"] if v["resolved"]),
            "n_refused": len(refused),
            "definite_pitches": tab_mod.definite_pitches(resolved),
        })

    def _post_tab_fingering(self):
        body = self._read_json()
        case_id = body.get("id")
        tab = get_demo_tab(case_id)
        if tab is None:
            return self._err("no tablature for case (not a demo phrase)", 404)
        try:
            updated = tab_mod.apply_fingering_edit(
                tab, int(body["index"]), int(body["string"]), int(body["fret"]))
        except (KeyError, IndexError, ValueError) as e:
            return self._err(f"bad fingering edit: {e}", 400)
        return self._json({"tab": updated,
                           "definite_pitches": tab_mod.definite_pitches(updated)})

    def _get_export(self, path: str):
        # /api/export/<case>/<fmt>
        parts = path[len("/api/export/"):].split("/")
        if len(parts) != 2:
            return self._err("bad export path", 400)
        case_id, fmt = parts
        ext = {"musicxml": "musicxml", "mid": "mid", "midi": "mid"}.get(fmt)
        if not ext:
            return self._err("bad format", 400)
        p = os.path.join(EXPORT_DIR, f"{case_id}.{ext}")
        if not os.path.exists(p):
            return self._err("export not ready", 404)
        with open(p, "rb") as f:
            data = f.read()
        ctype = "application/vnd.recordare.musicxml+xml" if ext == "musicxml" else "audio/midi"
        return self._send(data, ctype=ctype,
                          extra={"Content-Disposition":
                                 f'attachment; filename="{case_id}.{ext}"'},
                          compress=False)

    def _post_analyze(self):
        raw = self._read_body()
        if not raw:
            return self._err("empty upload")
        fname = self.headers.get("X-Filename", "upload.wav")
        uid = "up_" + hashlib.sha1(raw).hexdigest()[:12]
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        path = os.path.join(UPLOAD_DIR, f"{uid}.wav")
        with open(path, "wb") as f:
            f.write(raw)
        try:
            x, sr = wavio.read_wav(path)
        except Exception as e:
            return self._err(f"cannot read WAV: {e}", 400)
        tr = analyze_signal(x, sr, AnalysisConfig(), allow_revision=True)
        tr["mode"] = "offline"
        with _CACHE_LOCK:
            _UPLOADS[uid] = {"name": fname, "path": path,
                             "duration_s": tr["audio"]["duration_s"]}
            _TRACK_CACHE[f"{uid}:rev=True"] = tr
        with open(os.path.join(UPLOAD_DIR, f"{uid}.json"), "w") as f:
            json.dump(tr, f, default=_json_default)
        return self._json({"id": uid, "name": fname,
                           "track": _trim_track(tr)})

    def _post_stream(self):
        body = self._read_json()
        case_id = body.get("id")
        policy = body.get("policy", "strict")
        if policy not in ("strict", "best_effort"):
            policy = "strict"
        res = get_stream(case_id, policy)
        if res is None:
            return self._err("unknown case", 404)
        return self._json(res)

    def _post_accompaniment(self):
        body = self._read_json()
        case_id = body.get("id")
        mode = body.get("mode", "offline")
        policy = body.get("policy", "strict")
        upload_id = body.get("upload_id")
        edits = body.get("edits") or []
        follow = body.get("follow", "score")

        if mode == "streaming":
            res = get_stream(case_id, policy)
        else:
            res = get_track(case_id)
        if res is None:
            return self._err("unknown case", 404)

        sr = int(res.get("audio", {}).get("sample_rate") or 22050)
        duration = float(res.get("audio", {}).get("duration_s") or
                         (res["frames"][-1]["t_end"] if res.get("frames") else 0.0))

        follow_score = None
        if mode != "streaming" and follow == "score":
            base = score_mod.build_score(res, tempo_bpm=100.0)
            if edits:
                base = score_mod.apply_edits(base, edits)
            follow_score = base

        rendered = accompaniment.render_accompaniment(
            res, sr, duration=duration, follow_score=follow_score)
        return self._json({
            "audio_wav_b64": _b64(rendered["audio"], sr),
            "sample_rate": sr,
            "duration_s": rendered["duration_s"],
            "spans": rendered["spans"],
            "gated": rendered["gated"],
            "n_spans": rendered["n_spans"],
            "mode": mode,
        })

    def _post_score(self):
        body = self._read_json()
        case_id = body.get("id")
        tr = get_track(case_id)
        if tr is None:
            return self._err("unknown case", 404)
        sc = score_mod.build_score(tr, tempo_bpm=body.get("tempo_bpm", 100.0))
        return self._json({"score": sc})

    def _post_score_edit(self):
        body = self._read_json()
        case_id = body.get("id")
        edits = body.get("edits") or []
        tr = get_track(case_id)
        if tr is None:
            return self._err("unknown case", 404)
        pristine = score_mod.build_score(tr, tempo_bpm=body.get("tempo_bpm", 100.0))
        edited = score_mod.apply_edits(pristine, edits)
        records = feedback.classify_edits(pristine, edited, tr)
        fb_path = os.path.join(FEEDBACK_DIR, f"{case_id}.json")
        feedback.append_feedback(fb_path, case_id, records, edited)

        os.makedirs(EXPORT_DIR, exist_ok=True)
        score_mod.write_exports(edited, EXPORT_DIR, name=case_id)

        # re-render accompaniment from the edited score so the user hears the fix
        sr = int(tr.get("audio", {}).get("sample_rate") or 22050)
        duration = float(tr.get("audio", {}).get("duration_s") or
                         (tr["frames"][-1]["t_end"] if tr.get("frames") else 0.0))
        rendered = accompaniment.render_accompaniment(
            tr, sr, duration=duration, follow_score=edited)

        return self._json({
            "score": edited,
            "feedback": records,
            "exports": {"musicxml": f"/api/export/{case_id}/musicxml",
                        "midi": f"/api/export/{case_id}/mid"},
            "audio_wav_b64": _b64(rendered["audio"], sr),
            "sample_rate": sr,
            "spans": rendered["spans"],
        })


def _b64(audio: np.ndarray, sr: int) -> str:
    import base64
    return base64.b64encode(wav_bytes(audio, sr)).decode("ascii")


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def make_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    return srv


def main():
    ap = argparse.ArgumentParser(description="Intonation analysis UI server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    srv = make_server(args.host, args.port)
    print(f"Intonation UI: http://{args.host}:{args.port}/")
    print("Press Ctrl+C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
