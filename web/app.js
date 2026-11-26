/* Intonation analysis UI.
 *
 * Everything shown is derived from the kernel's exported track: per-frame
 * literal f0, committed (tracked) f0, octave-ambiguity pairs, candidate
 * evidence with rejection reasons, and the streaming admission gate.  Edits on
 * the staff are posted back to the server, which re-renders the accompaniment
 * and records failure feedback for the kernel.
 */

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const midiOf = (f) => (f && f > 0) ? 69 + 12 * Math.log2(f / 440) : null;
  const STEP = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"];
  const noteName = (m) => {
    if (m == null) return "—";
    const r = Math.round(m);
    return STEP[((r % 12) + 12) % 12] + (Math.floor(r / 12) - 1);
  };

  const state = {
    cases: [],
    caseId: null,
    mode: "offline",
    policy: "strict",
    track: null,          // raw response (offline trimmed or streaming)
    frames: [],           // normalized
    audio: null,          // decoded Float32Array
    audioRate: 22050,
    duration: 0,
    notes: [],
    deferrals: [],
    revisions: [],
    score: null,
    selectedNote: -1,
    selectedFrame: -1,
    edits: [],
    hovering: false,
    audioCache: {},
  };

  // ---------------------------------------------------------------------
  // frame normalization
  // ---------------------------------------------------------------------
  function normalize(data, mode) {
    const frames = (data.frames || []).map((f, i) => {
      if (mode === "streaming") {
        return {
          index: f.index ?? i,
          t_start: f.t_start, t_center: f.t_center, t_end: f.t_end,
          literal: f.f0_hz ?? null,
          tracked: f.admitted ? (f.f0_hz ?? null) : null,
          state: f.state,
          confidence: f.confidence ?? 0,
          ambiguous: !!f.ambiguous,
          lower: f.lower_f0_hz ?? null,
          upper: f.upper_f0_hz ?? null,
          attack: !!f.is_attack,
          admitted: !!f.admitted,
          gate: !f.admitted,
          gateReason: f.admission_reason,
          deadline: f.deadline_t ?? null,
          overdue: !!f.overdue,
          resolution: f.deadline_resolution,
          candidates: [], peaks: [],
          corrected: false,
        };
      }
      return {
        index: f.index ?? i,
        t_start: f.t_start, t_center: f.t_center, t_end: f.t_end,
        literal: f.f0_hz ?? null,
        tracked: f.tracked_f0_hz ?? null,
        state: f.state,
        confidence: f.confidence ?? 0,
        ambiguous: !!f.ambiguous,
        lower: f.lower_f0_hz ?? null,
        upper: f.upper_f0_hz ?? null,
        attack: !!f.attack,
        admitted: f.tracked_f0_hz != null,
        gate: f.tracked_f0_hz == null && (f.state === "voiced" || f.state === "ambiguous"),
        gateReason: f.decision || f.state,
        deadline: null, overdue: false, resolution: null,
        candidates: f.candidates || [], peaks: f.peaks || [],
        corrected: !!f.octave_corrected,
      };
    });
    return frames;
  }

  // ---------------------------------------------------------------------
  // API
  // ---------------------------------------------------------------------
  async function api(path, opts) {
    const res = await fetch(path, opts);
    if (!res.ok) {
      let msg = res.status + " " + res.statusText;
      try { const j = await res.json(); msg = j.error || msg; } catch (e) {}
      throw new Error(msg);
    }
    return res.json();
  }

  function setStatus(msg, busy) {
    const el = $("status");
    el.textContent = msg;
    el.classList.toggle("busy", !!busy);
  }

  // ---------------------------------------------------------------------
  // load cases & track
  // ---------------------------------------------------------------------
  async function loadCases() {
    const data = await api("/api/cases");
    state.cases = data.cases || [];
    const sel = $("case");
    sel.innerHTML = "";
    for (const c of state.cases) {
      const opt = document.createElement("option");
      opt.value = c.id;
      opt.dataset.source = c.source;
      opt.textContent = `${c.name}  ·  ${(c.kind)}  ·  ${(c.duration_s || 0).toFixed(2)}s`;
      sel.appendChild(opt);
    }
    if (state.cases.length) {
      const first = state.cases.find(c => c.id === "two_take_a")
        || state.cases.find(c => c.source === "dataset") || state.cases[0];
      sel.value = first.id;
      await loadCase(first.id);
    }
  }

  async function loadCase(id) {
    state.caseId = id;
    state.edits = [];
    state.score = null;
    state.selectedNote = -1;
    state.selectedFrame = -1;
    setStatus("分析中…", true);
    try {
      await loadAudio(id);
      if (state.mode === "offline") {
        state.track = await api(`/api/track/${encodeURIComponent(id)}`);
      } else {
        state.track = await api(`/api/stream/${encodeURIComponent(id)}?policy=${state.policy}`);
      }
      applyTrack(state.track, state.mode);
      await loadScore(id);
      await loadBadgesAndReport(id);
      setStatus("就绪");
    } catch (e) {
      setStatus("错误: " + e.message, false);
      console.error(e);
    }
  }

  async function reloadMode() {
    if (!state.caseId) return;
    state.edits = [];
    state.selectedNote = -1;
    setStatus(state.mode === "streaming" ? "实时分析…" : "离线分析…", true);
    try {
      if (state.mode === "offline") {
        state.track = await api(`/api/track/${encodeURIComponent(state.caseId)}`);
      } else {
        state.track = await api(`/api/stream/${encodeURIComponent(state.caseId)}?policy=${state.policy}`);
      }
      applyTrack(state.track, state.mode);
      await loadScore(state.caseId);
      setStatus("就绪");
    } catch (e) {
      setStatus("错误: " + e.message, false);
    }
  }

  function applyTrack(track, mode) {
    state.frames = normalize(track, mode);
    state.notes = (track.notes || []);
    state.deferrals = track.deferrals || [];
    state.revisions = track.revisions || [];
    state.duration = (track.audio && track.audio.duration_s) ||
      (state.frames.length ? state.frames[state.frames.length - 1].t_end : 0) ||
      state.duration;
    $("tdur").textContent = state.duration.toFixed(2);
    $("view-title").textContent = mode === "streaming"
      ? "基频曲线与原音对照 · 实时（因果，一拍截止）"
      : "基频曲线与原音对照 · 离线（可用后续证据回改）";
    $("policy-wrap").classList.toggle("hidden", mode !== "streaming");
    drawWave();
    drawTrack();
    renderBadges();
    renderReportForCase();
    if (state.frames.length) selectFrame(nearestFrame(0.4));
  }

  async function loadAudio(id) {
    if (state.audioCache[id]) {
      const a = state.audioCache[id];
      state.audio = a.data; state.audioRate = a.rate; state.duration = a.duration;
      try { $("player").src = a.url; } catch (e) {}
      return;
    }
    const url = `/api/audio/${encodeURIComponent(id)}`;
    const player = $("player");
    player.src = url;
    try {
      const buf = await (await fetch(url)).arrayBuffer();
      const ctx = new (AudioContext || webkitAudioContext)();
      const decoded = await ctx.decodeAudioData(buf.slice(0));
      const data = decoded.getChannelData(0);
      state.audioCache[id] = { data, rate: decoded.sampleRate, duration: decoded.duration, url };
      state.audio = data; state.audioRate = decoded.sampleRate; state.duration = decoded.duration;
      ctx.close && ctx.close();
    } catch (e) {
      console.warn("decode failed, waveform disabled", e);
      state.audio = null;
    }
  }

  async function loadScore(id) {
    try {
      const data = await api("/api/score", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id, tempo_bpm: 100 }),
      });
      state.score = data.score;
      state.selectedNote = firstNoteIndex();
      drawStaff();
      renderScoreInfo();
    } catch (e) {
      console.warn(e);
    }
  }

  async function loadBadgesAndReport(id) {
    renderReportForCase();
  }

  // ---------------------------------------------------------------------
  // canvas sizing
  // ---------------------------------------------------------------------
  function fitCanvas(cv) {
    const dpr = window.devicePixelRatio || 1;
    const rect = cv.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: rect.width, h: rect.height };
  }

  // ---------------------------------------------------------------------
  // waveform
  // ---------------------------------------------------------------------
  function drawWave() {
    const cv = $("wave");
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    if (!state.audio) {
      ctx.fillStyle = "#0b0e13"; ctx.fillRect(0, 0, w, h);
      return;
    }
    const data = state.audio;
    const n = data.length;
    const cols = Math.floor(w);
    const mid = h / 2;
    ctx.fillStyle = "#1f2a3a";
    for (let x = 0; x < cols; x++) {
      const i0 = Math.floor(x / cols * n);
      const i1 = Math.floor((x + 1) / cols * n);
      let mn = 0, mx = 0;
      const stride = Math.max(1, Math.floor((i1 - i0) / 40));
      for (let i = i0; i < i1; i += stride) {
        const v = data[i];
        if (v < mn) mn = v; if (v > mx) mx = v;
      }
      const y0 = mid - mx * (h * 0.46);
      const y1 = mid - mn * (h * 0.46);
      ctx.fillRect(x, y0, 1, Math.max(1, y1 - y0));
    }
    // factor marking: ambiguous regions tinted
    ctx.fillStyle = "rgba(192,125,255,0.10)";
    for (const f of state.frames) {
      if (f.ambiguous) {
        const x0 = timeToX(f.t_start, w), x1 = timeToX(f.t_end, w);
        ctx.fillRect(x0, 0, Math.max(1, x1 - x0), h);
      }
    }
  }

  function timeToX(t, w) { return state.duration > 0 ? (t / state.duration) * w : 0; }
  function xToTime(x, w) { return state.duration > 0 ? (x / w) * state.duration : 0; }

  // ---------------------------------------------------------------------
  // pitch track
  // ---------------------------------------------------------------------
  function pitchRange() {
    let lo = Infinity, hi = -Infinity;
    for (const f of state.frames) {
      for (const v of [f.literal, f.tracked, f.lower, f.upper]) {
        if (v && v > 0) { const m = midiOf(v); if (m < lo) lo = m; if (m > hi) hi = m; }
      }
    }
    if (!isFinite(lo)) { lo = 40; hi = 90; }
    // include notes
    for (const n of state.notes) {
      if (n.median_f0_hz) { const m = midiOf(n.median_f0_hz); lo = Math.min(lo, m); hi = Math.max(hi, m); }
    }
    const pad = 3;
    return { lo: Math.floor(lo - pad), hi: Math.ceil(hi + pad) };
  }

  function pitchToY(f, range, h, topPad, botPad) {
    const m = midiOf(f);
    if (m == null) return h;
    const plotH = h - topPad - botPad;
    return topPad + plotH - (m - range.lo) / (range.hi - range.lo) * plotH;
  }

  function drawTrack() {
    const cv = $("track");
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const range = pitchRange();
    const topPad = 10, botPad = 18;

    // grid: 0.5 s verticals + octave horizontals
    ctx.strokeStyle = "rgba(255,255,255,0.05)";
    ctx.lineWidth = 1;
    for (let t = 0; t <= state.duration + 1e-3; t += 0.5) {
      const x = timeToX(t, w);
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillStyle = "rgba(138,149,168,0.55)";
    for (let m = range.lo; m <= range.hi; m++) {
      if (m % 12 !== 0) continue; // label each C
      const y = pitchToY(440 * Math.pow(2, (m - 69) / 12), range, h, topPad, botPad);
      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      ctx.fillText(noteName(m), 2, y - 2);
    }

    // gate regions (streaming) red band at bottom
    if (state.mode === "streaming") {
      ctx.fillStyle = "rgba(255,107,107,0.18)";
      for (const f of state.frames) {
        if (f.gate) {
          const x0 = timeToX(f.t_start, w), x1 = timeToX(f.t_end, w);
          ctx.fillRect(x0, h - 6, Math.max(1, x1 - x0), 6);
        }
      }
    }

    // literal f0 (raw estimate) — thin blue, drawn as small segments
    ctx.strokeStyle = "#5aa9ff";
    ctx.lineWidth = 1.3;
    ctx.globalAlpha = 0.9;
    for (const f of state.frames) {
      if (!f.literal) continue;
      const y = pitchToY(f.literal, range, h, topPad, botPad);
      const x = timeToX(f.t_center, w);
      ctx.beginPath(); ctx.arc(x, y, 1.6, 0, Math.PI * 2); ctx.fillStyle = "#5aa9ff"; ctx.fill();
    }
    ctx.globalAlpha = 1;

    // committed track — thick green line
    ctx.strokeStyle = "#38d39f";
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    let pen = false;
    for (const f of state.frames) {
      if (f.tracked) {
        const y = pitchToY(f.tracked, range, h, topPad, botPad);
        const x = timeToX(f.t_center, w);
        if (!pen) { ctx.moveTo(x, y); pen = true; } else ctx.lineTo(x, y);
      } else { pen = false; }
    }
    ctx.stroke();

    // ambiguous markers: purple vertical connector lower<->upper at literal
    for (const f of state.frames) {
      if (!f.ambiguous || !f.lower || !f.upper) continue;
      const x = timeToX(f.t_center, w);
      const yU = pitchToY(f.upper, range, h, topPad, botPad);
      const yL = pitchToY(f.lower, range, h, topPad, botPad);
      ctx.strokeStyle = "rgba(192,125,255,0.55)";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, yU); ctx.lineTo(x, yL); ctx.stroke();
      ctx.fillStyle = "#c07dff";
      ctx.beginPath(); ctx.arc(x, yL, 1.8, 0, Math.PI * 2); ctx.fill();
    }

    // selected frame line
    if (state.selectedFrame >= 0 && state.selectedFrame < state.frames.length) {
      const f = state.frames[state.selectedFrame];
      const x = timeToX(f.t_center, w);
      ctx.strokeStyle = "rgba(255,209,102,0.9)";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }

    // bottom axis labels
    ctx.fillStyle = "rgba(138,149,168,0.7)";
    ctx.font = "10px ui-monospace, monospace";
    for (let t = 0; t <= state.duration + 1e-3; t += 0.5) {
      const x = timeToX(t, w);
      ctx.fillText(t.toFixed(1) + "s", clamp(x + 2, 2, w - 30), h - 4);
    }
  }

  // ---------------------------------------------------------------------
  // frame selection / readout / inspectors
  // ---------------------------------------------------------------------
  function nearestFrame(t) {
    if (!state.frames.length) return -1;
    let best = 0, bd = Infinity;
    state.frames.forEach((f, i) => {
      const d = Math.abs(f.t_center - t);
      if (d < bd) { bd = d; best = i; }
    });
    return best;
  }

  function selectFrame(i) {
    if (i < 0 || i >= state.frames.length) return;
    state.selectedFrame = i;
    const f = state.frames[i];
    drawTrack();
    drawSpectrum();
    renderCandidates();
    renderReadout(f);
  }

  function renderReadout(f) {
    const conf = (f.confidence * 100).toFixed(0);
    const lit = f.literal ? f.literal.toFixed(1) + " Hz (" + noteName(midiOf(f.literal)) + ")" : "无可靠结果";
    const trk = f.tracked ? f.tracked.toFixed(1) + " Hz (" + noteName(midiOf(f.tracked)) + ")" : "—";
    const amb = f.ambiguous ? `未决 下=${f.lower?f.lower.toFixed(1):"—"} 上=${f.upper?f.upper.toFixed(1):"—"}` : "否";
    const gate = f.gate ? ` · 未准入(${f.gateReason})` : "";
    $("frame-readout").innerHTML =
      `<b>帧 ${f.index}</b> · t=${f.t_center.toFixed(3)}s ` +
      `· 状态=<span style="color:#e8edf4">${f.state}${f.attack ? " ⚡起音" : ""}</span> ` +
      `· 置信=${conf}% · 原始=${lit} · 确认=${trk} · 八度${amb}${gate}` +
      ` <span class="muted">— 点击轨迹试听该片段</span>`;
  }

  function renderCandidates() {
    const i = state.selectedFrame;
    const box = $("cand-inspector");
    const f = state.frames[i];
    if (!f) { box.innerHTML = '<p class="muted">选择一帧查看。</p>'; return; }
    if (state.mode === "streaming") {
      box.innerHTML = `<p class="muted">实时模式仅输出准入结论；候选证据见离线模式。</p>` +
        gateStory(f);
      return;
    }
    const cands = f.candidates || [];
    if (!cands.length) {
      box.innerHTML = `<p class="muted">该帧无可保留候选（状态 ${f.state}）。</p>`;
      return;
    }
    const best = cands[0];
    const html = cands.map((c, k) => {
      const rejected = !!c.rejection;
      const cls = `crow${k === 0 ? " best" : ""}${rejected ? " rejected" : ""}`;
      const tags = [];
      if (c.viable) tags.push('<span class="tag viable">证据充分</span>');
      if (!c.fundamental_present) tags.push('<span class="tag amb">缺基音</span>');
      if (c.yin_agree) tags.push('<span class="tag">YIN 一致</span>');
      if (rejected) tags.push(`<span class="tag bad">${c.rejection}</span>`);
      const harm = (c.matched || []).map(m => `${m.k}:${m.peak_freq.toFixed(0)}`).join(" ");
      const miss = (c.missing || []).length ? c.missing.join(",") : "无";
      const cover = clamp(c.coverage, 0, 1);
      return `<div class="${cls}">
        <div class="chead">
          <span class="f0">${c.f0_hz.toFixed(1)} Hz</span>
          <span class="muted">score ${c.score.toFixed(3)} · ${noteName(midiOf(c.f0_hz))}</span>
        </div>
        <div class="meter"><i style="width:${(cover*100).toFixed(0)}%"></i></div>
        <div class="harmlist">覆盖 ${(cover*100).toFixed(0)}% · 谐波 ${c.n_harmonics} · 缺失[${miss}] · 奇/偶 ${c.odd_even_ratio==null?"∞":c.odd_even_ratio.toFixed(2)}</div>
        <div class="harmlist">匹配 k:f = ${harm || "—"}</div>
        <div style="margin-top:5px;display:flex;gap:5px;flex-wrap:wrap">${tags.join("")}</div>
      </div>`;
    }).join("");
    const yin = f.yin_f0_hz ? `YIN ${f.yin_f0_hz.toFixed(1)}Hz(ap=${f.yin_aperiodicity.toFixed(2)})` : "YIN 无";
    box.innerHTML = `<div class="harmlist" style="margin-bottom:6px">${yin} · 恰取最大峰不是本方法：候选按谐波梳解释度生成</div>` + html;
  }

  function gateStory(f) {
    if (!f.gate) return `<p class="muted">该帧已准入伴奏。</p>`;
    let s = `<p class="muted">未准入：${f.gateReason || "—"}`;
    if (f.overdue) s += ` · 已过一拍截止`;
    if (f.deadline) s += ` · 截止 t=${f.deadline.toFixed(2)}s`;
    if (f.resolution) s += ` · 结论 ${f.resolution}`;
    s += `</p>`;
    return s;
  }

  // ---------------------------------------------------------------------
  // spectrum from decoded audio (own FFT) + kernel peaks overlay
  // ---------------------------------------------------------------------
  function fftRadix2(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = -2 * Math.PI / len;
      const wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < len / 2; k++) {
          const ur = re[i + k], ui = im[i + k];
          const vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
          const vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
          re[i + k] = ur + vr; im[i + k] = ui + vi;
          re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
          const ncr = cr * wr - ci * wi;
          ci = cr * wi + ci * wr; cr = ncr;
        }
      }
    }
  }

  function computeSpectrum(frameIdx) {
    if (!state.audio) return null;
    const f = state.frames[frameIdx];
    const sr = state.audioRate;
    const FL = 2048, NFFT = 8192;
    const start = Math.max(0, Math.round(f.t_start * sr));
    const re = new Float64Array(NFFT), im = new Float64Array(NFFT);
    let winSum = 0;
    for (let i = 0; i < FL; i++) {
      const wv = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / FL);
      winSum += wv;
      const x = (state.audio[start + i] || 0) * wv;
      re[i] = x;
    }
    fftRadix2(re, im);
    const bins = NFFT / 2 + 1;
    const mag = new Float32Array(bins);
    for (let k = 0; k < bins; k++) {
      mag[k] = 2 * Math.hypot(re[k], im[k]) / (winSum || 1);
    }
    return { mag, bins, sr, nfft: NFFT };
  }

  function drawSpectrum() {
    const cv = $("spectrum");
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#0b0e13"; ctx.fillRect(0, 0, w, h);
    const f = state.frames[state.selectedFrame];
    const spec = computeSpectrum(state.selectedFrame);
    const fmax = 2200;
    if (spec) {
      let gmax = 1e-9;
      const kHi = Math.round(fmax / spec.sr * spec.nfft);
      for (let k = 0; k < kHi; k++) gmax = Math.max(gmax, spec.mag[k]);
      ctx.strokeStyle = "rgba(90,169,255,0.08)";
      for (let hz = 100; hz <= fmax; hz += 100) {
        const x = hz / fmax * w;
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
        ctx.fillStyle = "rgba(138,149,168,0.45)";
        ctx.font = "9px ui-monospace, monospace";
        if (hz % 200 === 0) ctx.fillText(hz, x + 1, h - 2);
      }
      ctx.beginPath();
      for (let x = 0; x < w; x++) {
        const k = Math.round(x / w * kHi);
        const db = 20 * Math.log10(Math.max(spec.mag[k], gmax * 1e-5) / gmax);
        const y = h - (db + 80) / 80 * (h - 14) - 8;
        if (x === 0) ctx.moveTo(x, clamp(y, 0, h)); else ctx.lineTo(x, clamp(y, 0, h));
      }
      ctx.strokeStyle = "#5aa9ff"; ctx.lineWidth = 1; ctx.stroke();
      ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
      ctx.fillStyle = "rgba(90,169,255,0.10)"; ctx.fill();
    }
    // kernel peaks as dots
    if (f && f.peaks) {
      const pa = f.peaks.map(p => p.amp);
      const gmax = Math.max(1e-9, ...pa);
      for (const p of f.peaks) {
        if (p.freq_hz > fmax) continue;
        const x = p.freq_hz / fmax * w;
        const db = 20 * Math.log10(Math.max(p.amp, 1e-9) / gmax);
        const y = h - (db + 80) / 80 * (h - 14) - 8;
        ctx.fillStyle = "#ffb454";
        ctx.beginPath(); ctx.arc(x, clamp(y, 0, h), 2.4, 0, Math.PI * 2); ctx.fill();
      }
    }
    // harmonic marks of the selected/committed pitch
    const pitch = f ? (f.tracked || f.literal) : null;
    if (pitch) {
      ctx.strokeStyle = "rgba(56,211,159,0.5)"; ctx.lineWidth = 1;
      for (let k = 1; k * pitch < fmax; k++) {
        const x = k * pitch / fmax * w;
        ctx.beginPath(); ctx.moveTo(x, 8); ctx.lineTo(x, h - 8); ctx.stroke();
        ctx.fillStyle = "rgba(56,211,159,0.7)";
        ctx.font = "9px ui-monospace, monospace";
        ctx.fillText(k, x + 1, 10);
      }
    }
    ctx.fillStyle = "rgba(138,149,168,0.8)";
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillText("Hz (0–" + fmax + ")", 6, 12);
  }

  // ---------------------------------------------------------------------
  // badges
  // ---------------------------------------------------------------------
  function renderBadges() {
    const box = $("badges");
    const fr = state.frames;
    const voiced = fr.filter(f => f.tracked).length;
    const amb = fr.filter(f => f.ambiguous).length;
    const gated = fr.filter(f => f.gate).length;
    const revised = fr.filter(f => f.corrected).length;
    const total = fr.length || 1;
    const badges = [];
    badges.push(chip(`${voiced}/${total} 帧确认`, voiced / total > 0.6 ? "good" : "warn"));
    badges.push(chip(`${(voiced / total * 100).toFixed(0)}% 有效覆盖`));
    if (amb) badges.push(chip(`${amb} 帧八度未决`, "warn"));
    if (state.mode === "offline" && revised) badges.push(chip(`${revised} 帧经回改`, "good"));
    if (state.mode === "streaming") badges.push(chip(`${gated} 帧未准入`, gated ? "bad" : "good"));
    badges.push(chip(`策略 ${state.mode === "streaming" ? state.policy : "离线"}`, "warn"));
    box.innerHTML = badges.join("");
  }

  function chip(txt, cls) { return `<span class="badge ${cls || ""}">${txt}</span>`; }

  // ---------------------------------------------------------------------
  // staff (editable score)
  // ---------------------------------------------------------------------
  function firstNoteIndex() {
    if (!state.score) return -1;
    return state.score.notes.findIndex(n => n.kind === "note");
  }

  function renderScoreInfo() {
    if (!state.score) { $("score-info").textContent = ""; return; }
    const n = state.score.notes.filter(x => x.kind === "note").length;
    $("score-info").textContent =
      `${n} 个音符 · ${state.score.tempo_bpm} BPM · 量化 ${state.score.grid_beats} 拍 · 修改 ${state.edits.length}`;
  }

  function drawStaff() {
    const cv = $("staff");
    if (!state.score) return;
    const notes = state.score.notes.filter(n => n.kind !== "deleted");
    const totalBeats = Math.max(
      state.score.beats_per_measure,
      Math.ceil(Math.max(...notes.map(n => n.beat_start + n.beats), 1) /
                state.score.beats_per_measure) * state.score.beats_per_measure);
    const pxPerBeat = 62;
    const width = Math.max(900, totalBeats * pxPerBeat + 90);
    const height = 210;
    const dpr = window.devicePixelRatio || 1;
    cv.style.width = width + "px";
    cv.style.height = height + "px";
    cv.width = Math.round(width * dpr);
    cv.height = Math.round(height * dpr);
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#f4f1ea";
    ctx.fillRect(0, 0, width, height);

    // pitch mapping
    let loM = 55, hiM = 83;
    const voiced = notes.filter(n => n.kind === "note");
    if (voiced.length) {
      const ms = voiced.map(n => n.midi);
      loM = Math.floor(Math.min(...ms) - 3);
      hiM = Math.ceil(Math.max(...ms) + 3);
    }
    const top = 28, bot = height - 28;
    const plotH = bot - top;
    const yOf = (midi) => top + plotH - (midi - loM) / Math.max(1, hiM - loM) * plotH;

    const staffLines = 5;
    const baseMidi = loM;
    ctx.strokeStyle = "#b9b2a2"; ctx.lineWidth = 1;
    for (let s = 0; s < staffLines; s++) {
      const y = yOf(baseMidi + s * ((hiM - loM) / 4));
      ctx.beginPath(); ctx.moveTo(46, y); ctx.lineTo(width - 20, y); ctx.stroke();
    }
    // left margin + clef glyph
    ctx.fillStyle = "#6b6455";
    ctx.font = "30px serif";
    ctx.fillText("𝄞", 52, yOf(baseMidi) + 2);

    // bar lines and measure numbers
    const bpm = state.score.beats_per_measure;
    ctx.strokeStyle = "#c8c1b0";
    for (let b = 0; b <= totalBeats; b += bpm) {
      const x = 74 + b * pxPerBeat;
      ctx.beginPath(); ctx.moveTo(x, top - 8); ctx.lineTo(x, bot + 8); ctx.stroke();
      ctx.fillStyle = "#9a9280"; ctx.font = "10px monospace";
      ctx.fillText(Math.round(b / bpm) + 1, x + 3, top - 12);
    }

    // notes
    state.score.notes.forEach((n, idx) => {
      if (n.kind === "deleted") return;
      const x = 74 + n.beat_start * pxPerBeat;
      const wpx = Math.max(10, n.beats * pxPerBeat - 4);
      if (n.kind === "rest") {
        ctx.fillStyle = "rgba(120,112,98,0.5)";
        ctx.font = "14px serif";
        ctx.fillText("𝄽", x + wpx / 2 - 5, yOf((loM + hiM) / 2));
        return;
      }
      const y = yOf(n.midi);
      // ledger lines
      ctx.strokeStyle = "#b9b2a2";
      const lineStep = (hiM - loM) / 4;
      // above top line (midi > baseMidi + 4 lines)
      for (let m = baseMidi + 4 * lineStep + 2 * lineStep; m <= n.midi + 1e-6; m += lineStep) {
        ctx.beginPath(); ctx.moveTo(x - 4, yOf(m)); ctx.lineTo(x + wpx + 4, yOf(m)); ctx.stroke();
      }
      for (let m = baseMidi - lineStep; m >= n.midi - 1e-6; m -= lineStep) {
        ctx.beginPath(); ctx.moveTo(x - 4, yOf(m)); ctx.lineTo(x + wpx + 4, yOf(m)); ctx.stroke();
      }
      const sel = idx === state.selectedNote;
      // notehead
      ctx.fillStyle = n.edited ? "#c0562f" : "#232323";
      ctx.beginPath();
      ctx.ellipse(x + 6, y, 7.5, 5.5, -0.35, 0, Math.PI * 2);
      ctx.fill();
      if (sel) { ctx.strokeStyle = "#2e7fe0"; ctx.lineWidth = 2; ctx.stroke(); ctx.lineWidth = 1; }
      // stem
      ctx.strokeStyle = n.edited ? "#c0562f" : "#232323";
      ctx.beginPath(); ctx.moveTo(x + 13, y); ctx.lineTo(x + 13, y - 26); ctx.stroke();
      // label
      ctx.fillStyle = n.octave_corrected ? "#1f7a52" : "#3a3a3a";
      ctx.font = "11px ui-monospace, monospace";
      ctx.fillText(noteName(n.midi), x - 4, y + 22);
      if (n.octave_corrected) {
        ctx.fillStyle = "#1f7a52";
        ctx.fillText("↺", x + wpx + 2, y + 2);
      }
      // duration underline
      ctx.strokeStyle = "rgba(35,35,35,0.25)";
      ctx.beginPath(); ctx.moveTo(x, y + 30); ctx.lineTo(x + wpx, y + 30); ctx.stroke();
    });

    cv._staffGeom = { pxPerBeat, left: 74, yOf, loM, hiM, top, bot };
  }

  function scoreHitTest(px, py) {
    const cv = $("staff");
    const g = cv._staffGeom;
    if (!g || !state.score) return -1;
    let idx = -1, best = 22;
    state.score.notes.forEach((n, i) => {
      if (n.kind !== "note") return;
      const x = g.left + n.beat_start * g.pxPerBeat + 6;
      const y = g.yOf(n.midi);
      const d = Math.hypot(px - x, py - y);
      if (d < best) { best = d; idx = i; }
    });
    return idx;
  }

  // ---------------------------------------------------------------------
  // edits -> server
  // ---------------------------------------------------------------------
  async function pushEdit(action, delta) {
    if (!state.caseId || state.mode !== "offline") {
      setStatus("编辑仅在离线（可回改）模式可用", false);
      return;
    }
    const idx = state.selectedNote;
    if (idx < 0 || !state.score || state.score.notes[idx].kind !== "note") return;
    let ed;
    if (action === "octave") ed = { action: "octave", index: idx, dir: delta };
    else if (action === "nudge") ed = { action: "set_midi", index: idx, midi: Math.round(state.score.notes[idx].midi) + delta };
    else if (action === "delete") ed = { action: "delete", index: idx };
    else return;
    state.edits.push(ed);
    setStatus("应用修改…", true);
    try {
      const res = await api("/api/score/edit", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: state.caseId, edits: state.edits }),
      });
      state.score = res.score;
      renderScoreInfo();
      drawStaff();
      renderFeedback(res.feedback || []);
      if (res.audio_wav_b64) {
        const url = "data:audio/wav;base64," + res.audio_wav_b64;
        $("accomp").src = url;
      }
      // re-select same note if still a note
      if (state.score.notes[idx] && state.score.notes[idx].kind === "note") state.selectedNote = idx;
      setStatus(`已应用修改，跟奏已重渲染（${(res.spans || []).length} 段）`);
    } catch (e) {
      setStatus("修改失败: " + e.message, false);
    }
  }

  function renderFeedback(records) {
    const box = $("feedback-panel");
    if (!records || !records.length) {
      box.innerHTML = `<p class="muted">本次修改没有触发内核失败记录（或修改与内核一致）。</p>`;
      box.classList.remove("hidden");
      return;
    }
    const rows = records.map(r => {
      const bad = r.kind !== "pitch_error";
      const detail = r.user_midi != null
        ? `内核 ${noteName(r.kernel_midi)} → 你改为 ${noteName(r.user_midi)}`
        : "";
      return `<div class="frow ${bad ? "bad" : ""}">
        <b>${r.kind}</b> · ${detail} · t=${r.t_range.join("–")}s<br>
        <span class="muted">${r.suggestion}</span>
      </div>`;
    }).join("");
    box.innerHTML = `<div class="viewhead"><h2 style="font-size:13px">回传内核的失败反馈</h2></div>` + rows;
    box.classList.remove("hidden");
  }

  // ---------------------------------------------------------------------
  // report
  // ---------------------------------------------------------------------
  async function renderReportForCase() {
    const box = $("report");
    if (!state.caseId) return;
    let rep;
    try { rep = await api("/api/report"); } catch (e) { box.innerHTML = '<p class="muted">无报告。</p>'; return; }
    const c = (rep.cases || {})[state.caseId];
    if (!c) { box.innerHTML = '<p class="muted">该案例为上传音频，无预生成评估。</p>'; return; }
    const note = state.mode === "streaming"
      ? '<p class="muted" style="grid-column:1/-1">以下为同一录音的<b>离线</b>参考评估（内核逐帧原始估计）；实时模式的可听准入结论见上方轨迹与门控条带。</p>'
      : "";
    const cards = [];
    if (c.kind === "vibrato") {
      cards.push(rcard("颤音追踪", [
        ["覆盖率", pct(c.coverage)],
        ["跟踪误差(均)", fmt(c.tracking_abs_mean_cents) + " c"],
        ["p95", fmt(c.tracking_p95_cents) + " c"],
        ["八度错", c.octave_errors],
        ["参考速率", c.reference_vibrato_hz + " Hz"],
        ["恢复速率", fmt(c.recovered_vibrato_hz) + " Hz"],
        ["参考深度", c.reference_vibrato_cents + " c"],
        ["恢复深度", fmt(c.recovered_vibrato_cents_p5_p95) + " c"],
      ]));
    } else if (c.abs_mean_cents != undefined) {
      cards.push(rcard("音分误差", [
        ["覆盖率", pct(c.coverage)],
        ["平均|误差|", fmt(c.abs_mean_cents) + " c"],
        ["中位", fmt(c.median_abs_cents) + " c"],
        ["p95", fmt(c.p95_abs_cents) + " c"],
        ["八度错", c.octave_errors],
        ["帧数", c.n],
      ]));
    } else if (c.silence_total != undefined) {
      cards.push(rcard("静音/噪声防误报", [
        ["静音误报帧", c.silence],
        ["静音总帧", c.silence_total],
        ["噪声误报帧", c.noise],
        ["噪声总帧", c.noise_total],
        ["静音误报率", pct(c.silence_rate)],
        ["噪声误报率", pct(c.noise_rate)],
      ]));
    } else {
      cards.push(rcard("两段式场景", [
        ["覆盖率", pct(c.coverage)],
        ["未决八度标注", pct(c.ambiguity_flag_rate)],
        ["回改期望满足", c.revision_expectation_met ? "是" : "否"],
        ["回改次数", c.n_revisions],
        ["延迟记录", c.n_deferrals],
        ["严格段误差", fmt(c.strict_cent_stats && c.strict_cent_stats.abs_mean_cents) + " c"],
        ["严格段八度错率", pct(c.strict_cent_stats && c.strict_cent_stats.octave_error_rate)],
        ["误报率", pct(c.spurious_rate)],
      ]));
      if (c.jump) {
        cards.push(rcard("跳音延迟", [
          ["真实跳音", c.jump.true_jump_s + " s"],
          ["声明", c.jump.declared_s != null ? c.jump.declared_s + " s" : "—"],
          ["延迟", c.jump.delay_ms != null ? c.jump.delay_ms.toFixed(1) + " ms" : "—"],
          ["目标", c.jump.to_f0_hz + " Hz"],
        ]));
      }
    }
    box.innerHTML = note + cards.join("");
  }
  const pct = (x) => x == null ? "—" : (x * 100).toFixed(1) + "%";
  const fmt = (x) => x == null ? "—" : Number(x).toFixed(2);
  function rcard(title, rows) {
    return `<div class="rcard"><h3>${title}</h3>` +
      rows.map(([k, v]) => `<div class="kv"><span>${k}</span><b>${v}</b></div>`).join("") + "</div>";
  }

  // ---------------------------------------------------------------------
  // playback & audition
  // ---------------------------------------------------------------------
  let auditionTimer = null;
  function playPause() {
    const p = $("player");
    if (p.paused) {
      if (!p.src) { setStatus("无可播放音频", false); return; }
      p.play().catch(e => setStatus("播放失败: " + e.message, false));
      $("play").textContent = "❚❚ 暂停";
    } else { p.pause(); $("play").textContent = "▶ 播放"; }
  }

  function auditionAt(t) {
    const p = $("player");
    if (!p.src) return;
    const start = clamp(t - 0.12, 0, Math.max(0, state.duration - 0.05));
    state.auditioning = true;
    const go = () => {
      p.currentTime = start;
      p.play().then(() => {
        if (auditionTimer) clearTimeout(auditionTimer);
        auditionTimer = setTimeout(() => {
          p.pause();
          state.auditioning = false;
          $("play").textContent = "▶ 播放";
        }, 420);
      }).catch(() => { state.auditioning = false; });
    };
    if (p.readyState >= 1) go();
    else p.addEventListener("loadedmetadata", go, { once: true });
  }

  function onPlayerTime() {
    const p = $("player");
    const t = p.currentTime;
    $("tcur").textContent = t.toFixed(2);
    const roll = $("roll").getBoundingClientRect();
    const ph = $("playhead");
    const x = state.duration > 0 ? (t / state.duration) * roll.width : 0;
    ph.style.left = clamp(x, 0, roll.width) + "px";
    // auto-follow selected frame while playing (but not during a short audition,
    // whose seek is intentional and must not be overridden)
    const fi = nearestFrame(t);
    if (!p.paused && !state.auditioning && fi !== state.selectedFrame && fi >= 0) {
      if (Math.abs(state.frames[fi].t_center - t) < 0.1) selectFrame(fi);
    }
  }

  async function playAccompaniment() {
    if (!state.caseId) return;
    const btn = $("play-accomp");
    if (btn.dataset.ready === "1" && !$("accomp").paused) { $("accomp").pause(); btn.textContent = "♪ 跟奏"; return; }
    if (btn.dataset.ready === "1") { $("accomp").play(); btn.textContent = "❚❚ 跟奏"; return; }
    btn.textContent = "♪ 生成中…"; btn.disabled = true;
    try {
      const res = await api("/api/accompaniment", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id: state.caseId, mode: state.mode, policy: state.policy,
          follow: (state.mode === "streaming") ? "track" : (state.score ? "score" : "track"),
          edits: state.edits,
        }),
      });
      $("accomp").src = "data:audio/wav;base64," + res.audio_wav_b64;
      btn.dataset.ready = "1";
      btn.disabled = false;
      btn.textContent = "❚❚ 跟奏";
      $("accomp").play();
      const g = (res.gated || []).map(x => `${x.reason}@${x.t0.toFixed(1)}–${x.t1.toFixed(1)}s`).join(", ");
      $("frame-readout").innerHTML =
        `跟奏 ${res.n_spans} 段（受准入门控）` + (g ? ` · 未准入区: ${g}` : "");
    } catch (e) {
      btn.disabled = false; btn.textContent = "♪ 跟奏";
      setStatus("跟奏失败: " + e.message, false);
    }
  }

  // ---------------------------------------------------------------------
  // wiring
  // ---------------------------------------------------------------------
  function wire() {
    $("file").addEventListener("change", async (ev) => {
      const file = ev.target.files[0];
      if (!file) return;
      setStatus("上传并分析 " + file.name + " …", true);
      try {
        const buf = await file.arrayBuffer();
        const data = await api("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/octet-stream", "X-Filename": file.name },
          body: buf,
        });
        await loadCases();
        $("case").value = data.id;
        await loadCase(data.id);
        setStatus("已载入 " + file.name);
      } catch (e) {
        setStatus("上传失败: " + e.message, false);
      }
    });

    $("case").addEventListener("change", (e) => loadCase(e.target.value));

    $("play").addEventListener("click", playPause);
    $("play-accomp").addEventListener("click", playAccompaniment);

    document.querySelectorAll(".segbtn").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".segbtn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        state.mode = btn.dataset.mode;
        $("play-accomp").dataset.ready = "0";
        $("play-accomp").textContent = "♪ 跟奏";
        reloadMode();
      });
    });

    $("policy").addEventListener("change", (e) => {
      state.policy = e.target.value;
      $("play-accomp").dataset.ready = "0";
      reloadMode();
    });

    const player = $("player");
    player.addEventListener("timeupdate", onPlayerTime);
    player.addEventListener("ended", () => { $("play").textContent = "▶ 播放"; });

    // click roll -> audition frame's original snippet
    const roll = $("roll");
    roll.addEventListener("click", (ev) => {
      const rect = roll.getBoundingClientRect();
      const x = ev.clientX - rect.left;
      const t = xToTime(x, rect.width);
      const fi = nearestFrame(t);
      if (fi >= 0) {
        selectFrame(fi);
        auditionAt(state.frames[fi].t_center);
      }
    });

    // staff selection + keyboard edits
    const staff = $("staff");
    staff.addEventListener("click", (ev) => {
      const rect = staff.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const idx = scoreHitTest(ev.clientX - rect.left, ev.clientY - rect.top);
      if (idx >= 0) { state.selectedNote = idx; drawStaff(); }
    });

    window.addEventListener("keydown", (e) => {
      if (!state.score) return;
      const tag = (e.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "select" || tag === "textarea") return;
      if (state.selectedNote < 0) return;
      if (e.key === "ArrowUp") { e.preventDefault(); pushEdit("octave", 1); }
      else if (e.key === "ArrowDown") { e.preventDefault(); pushEdit("octave", -1); }
      else if (e.key === "ArrowRight") { e.preventDefault(); pushEdit("nudge", 1); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); pushEdit("nudge", -1); }
      else if (e.key === "Delete" || e.key === "Backspace") { e.preventDefault(); pushEdit("delete"); }
    });

    $("btn-export-mxl").addEventListener("click", () => {
      if (!state.caseId) return;
      window.open(`/api/export/${state.caseId}/musicxml`, "_blank");
    });
    $("btn-export-mid").addEventListener("click", () => {
      if (!state.caseId) return;
      window.open(`/api/export/${state.caseId}/mid`, "_blank");
    });
    $("btn-feedback").addEventListener("click", async () => {
      if (!state.caseId) return;
      const box = $("feedback-panel");
      try {
        const fb = await api(`/api/feedback/${state.caseId}`);
        const recs = (fb.last && fb.last.records) || [];
        renderFeedback(recs);
      } catch (e) { box.textContent = "无反馈记录"; box.classList.remove("hidden"); }
    });

    window.addEventListener("resize", () => {
      drawWave(); drawTrack(); drawSpectrum(); drawStaff();
    });
  }

  wire();
  loadCases();
})();
