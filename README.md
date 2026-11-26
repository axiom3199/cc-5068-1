# 乐器音准分析器 · Monophonic Instrument Intonation Analyzer

短时基频 / 清浊状态 / 可信度估计 → 与样本对齐的音高轨迹 → 点击试听原始片段
→ 跟奏合成器 → 可回改的五线谱（含离线八度回改）。

这是一个**内核 + 交互界面**的完整交付：分析不是脚本输出即止，界面由音准内核
驱动，界面上的修改会回传内核作为失败反馈。

---

## 一、界面怎样启动

界面是一个零第三方依赖的本地 Web 应用（Python 标准库 HTTP 服务 + 原生
Canvas/Web Audio 前端）。在仓库根目录执行：

```bash
# 1) 如尚未安装依赖（numpy / scipy）：
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt   # 已有 .venv 可跳过

# 2) 启动界面服务器（默认 http://127.0.0.1:8000）
.venv/bin/python -m server.app

# 或指定端口：
.venv/bin/python -m server.app --port 8000
```

然后在浏览器打开终端打印的地址（默认 <http://127.0.0.1:8000/>）。

启动后即可：

- 顶部下拉框选择内置录音（含两段式场景 `two_take_a` / `two_take_b`、谐波组、
  颤音、短噪声片段、静音/噪声等）。
- 「载入旋律」按钮上传任意单声道 WAV（≤ 几 MB 即可，服务端即时分析）。
- 「播放」播放原音，「♪ 跟奏」播放由音准轨驱动的跟奏合成器。
- 「离线 / 实时」切换两种分析：离线可用后续证据回改；实时为因果路径，带一拍
  截止的伴奏准入门控（`strict` 不决则不放行 / `best_effort` 截止即放行）。
- 点击音高轨迹任意处 → 试听该帧周围的原始片段，并查看该帧的**候选证据**
  （每个假设的匹配谐波、缺失谐波、奇/偶能量、覆盖度、YIN 一致性、拒绝原因）。
- 在五线谱上点选音符后：`↑/↓` 八度回改、`←/→` 微调、`Delete` 删除。
  每次修改都会重新渲染跟奏、导出 MusicXML/MIDI，并把修改分类为**回传内核的
  失败记录**（例如 `octave_failure` / `spurious_note` / `pitch_error`）。

内置数据可先重新生成（音频无音名文件名，真值单独成文件）：

```bash
.venv/bin/python scripts/generate_dataset.py
```

---

## 二、交付内容

```
intonation/            音准内核（DSP + 候选 + 跟踪/回改 + 流式 + 评估）
  config.py            帧长/步长/时间标签、阈值、截止拍等显式配置
  spectrum.py          帧化、幅度谱、峰值（含抛物线插值）、谱通量/起音
  yin.py               YIN 周期证据（独立于谐波梳）
  candidates.py        谐波梳假设生成 + 打分 + 保留/拒绝（非“取最大峰”）
  analyzer.py          逐帧证据与清浊判定（允许无可靠结果）
  tracker.py           帧间跟踪、八度延迟、音符分段、离线回改
  streaming.py         因果路径 + 一拍截止的伴奏准入门控
  accompaniment.py     由音准轨驱动的跟奏合成器（受准入门控）
  score.py             可回改五线谱：量化、编辑、MusicXML/MIDI 导出
  feedback.py          将界面修改回传为内核失败记录
  analysis.py          离线总入口
  evaluate.py          音分误差/八度错率/有效覆盖率/跳音延迟/颤音恢复/误报
  synth.py             合成测试数据（含缺失基音两段式场景）
  wavio.py             WAV 读写与重采样
server/app.py          HTTP 服务：静态页面 + JSON API（分析/流式/跟奏/乐谱/反馈）
web/                  前端（index.html / style.css / app.js，原生 Canvas）
scripts/generate_dataset.py   生成音频、真值、轨迹与评估报告
tests/                unittest（内核 + HTTP API 全链路）
data/                 生成产物（audio/ truth/ tracks/ results/ exports/ feedback/）
```

---

## 三、方法要点（对应题目要求）

- **不只取最大峰**：候选由“某个峰可能是更低基频的 k 次谐波”这一谐波梳假设
  生成，按解释了多少强峰能量、低次谐波强调、未解释强峰、缺失（仅计内部缺口）、
  调音残差与 YIN 一致性打分；每个候选保留 `matched/missing/odd_even/coverage/
  rejection`，可逐帧审查。因此缺失基音、强二次谐波不会被误判为基频。
- **八度是时间问题，不是单帧问题**：任意谐波列与低八度的“偶次谐波列（缺基音）”
  在频谱上不可区分——这是物理事实。故单帧只能报告“未决候选对”，八度由
  **起音 + 延音的时间证据**在线（截止内）或离线（后续证据回改）判定：
  - 共享开头只有偶次谐波 → 每帧读作高八度，标记未决；
  - A 段低基音/奇次在连奏中淡入（无新起音）→ 离线回改为低基频；
  - B 段真跳高八度（有新起音）→ 开头不回改，跳音被识别（实测延迟约 5 ms）。
- **允许无结果**：静音、噪声、起音瞬态不强行填音符；报告给出 0% 静音/噪声误报。
- **显式时间标签**：每帧含 `sample_start/center/end` 与 `t_start/center/end`，
  `frame_len=2048`(≈92.88 ms)、`hop=256`(≈11.61 ms)@22050 Hz。
- **原始候选与拒绝原因保留**：轨迹文件逐帧保存候选与拒绝原因；评估用原始逐帧
  估计，不以平滑曲线“看起来稳定”当准确。
- **伴奏准入一拍截止**：实时路径在“候选保留时间”内等待，超过一拍截止按策略
  放行或失败关闭，并报告 `deadline_t / overdue / deadline_resolution`。

---

## 四、量化结果（`data/results/report.json`，可重现）

- 已知频率谐波组：音分误差均值 < 0.1 c（一处低基频 55 Hz 约 0.74 c），
  **八度错率 0%**；含 `fixture_dom2`（强二次谐波）与 `fixture_missing_fund`。
- 颤音：跟随瞬时基频，跟踪误差 1–4 c，恢复的颤音速率/深度与参考接近。
- 短噪声片段：覆盖率 100%，误差 < 1 c。
- 静音/噪声：**误报率 0%**（按整窗落在区域内的诚实准则）。
- 两段式：A 段回改正确（392→196），B 段不回改（保持 392）且跳音到 784，
  跳音延迟约 4.8 ms；严格段误差约 3.4 c（含 A 段淡入过渡），八度错率 0%。

## 五、运行测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```
