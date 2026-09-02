"""Self-contained HTML reporting for RLAttack experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_dashboard(data: dict[str, Any], *, api_enabled: bool = False) -> str:
    """Render an offline-capable, dependency-free experiment dashboard."""

    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    api_flag = "true" if api_enabled else "false"
    return (
        _HTML_START
        + f"<script>window.__RLATTACK_DATA__={payload};"
        + f"window.__RLATTACK_API__={api_flag};</script>"
        + _HTML_END
    )


def write_dashboard_report(data: dict[str, Any], output_path: Path) -> Path:
    """Write a portable dashboard report and return its resolved path."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_dashboard(data), encoding="utf-8")
    return output_path.resolve()


def render_transfer_report(data: dict[str, Any]) -> str:
    """Render a compact, dependency-free transfer table."""

    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    return (
        _TRANSFER_START
        + f"<script>window.__RLATTACK_TRANSFER__={payload};</script>"
        + _TRANSFER_END
    )


def write_transfer_report(data: dict[str, Any], output_path: Path) -> Path:
    """Write a portable transfer report and return its resolved path."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_transfer_report(data), encoding="utf-8")
    return output_path.resolve()


_HTML_START = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>RLAttack · Simulation Observatory</title>
  <style>
    :root {
      --bg:#07100e; --panel:#0c1815; --panel-2:#10201c; --line:#20352f;
      --ink:#edf7f2; --muted:#91a69f; --green:#56f39a; --green-2:#20bd72;
      --amber:#ffcb66; --red:#ff6e6e; --blue:#71a7ff; --shadow:0 24px 70px #0007;
    }
    * { box-sizing:border-box }
    body {
      margin:0; min-width:320px; background:
        radial-gradient(circle at 12% -5%,#144631 0,transparent 26rem),
        radial-gradient(circle at 92% 10%,#112a23 0,transparent 24rem),var(--bg);
      color:var(--ink); font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;
      font-feature-settings:"tnum"; letter-spacing:-.01em;
    }
    body:before {
      content:""; position:fixed; inset:0; pointer-events:none; opacity:.2;
      background-image:linear-gradient(#9fffc708 1px,transparent 1px),
        linear-gradient(90deg,#9fffc708 1px,transparent 1px);
      background-size:46px 46px; mask-image:linear-gradient(to bottom,#000,transparent 70%);
    }
    button,select,input { font:inherit }
    .shell { width:min(1420px,calc(100% - 48px)); margin:auto; padding:28px 0 60px }
    nav { display:flex; align-items:center; justify-content:space-between; margin-bottom:58px }
    .brand { display:flex; gap:12px; align-items:center; font-weight:760; letter-spacing:-.04em }
    .mark {
      width:34px; height:34px; border:1px solid #61ffad66; border-radius:10px;
      display:grid; place-items:center; background:#12241e; box-shadow:inset 0 0 18px #4aff9a17;
    }
    .mark svg { width:20px }
    .safe {
      display:flex; align-items:center; gap:8px; color:#b8d3c9; font-size:12px;
      border:1px solid var(--line); border-radius:999px; padding:8px 12px; background:#091310aa;
    }
    .safe i { width:7px; height:7px; background:var(--green); border-radius:50%; box-shadow:0 0 12px var(--green) }
    .hero { display:grid; grid-template-columns:1.4fr .6fr; align-items:end; gap:30px; margin-bottom:30px }
    .eyebrow { color:var(--green); font-size:11px; font-weight:800; letter-spacing:.2em; text-transform:uppercase }
    h1 { max-width:850px; margin:13px 0 16px; font-size:clamp(42px,5vw,76px); line-height:.98; letter-spacing:-.07em; font-weight:720 }
    h1 em { color:var(--green); font-style:normal }
    .lede { max-width:720px; color:#a9bbb5; font-size:16px; line-height:1.7 }
    .hero-meta { display:grid; grid-template-columns:1fr 1fr; gap:10px }
    .hero-meta div { border-top:1px solid var(--line); padding-top:12px }
    .hero-meta span { color:var(--muted); display:block; font-size:10px; letter-spacing:.12em; text-transform:uppercase }
    .hero-meta strong { display:block; margin-top:5px; font-size:14px; font-weight:600 }
    .toolbar {
      position:relative; display:grid; grid-template-columns:repeat(6,minmax(110px,1fr)) auto;
      gap:10px; padding:14px; border:1px solid var(--line); border-radius:18px;
      background:#0a1512dd; box-shadow:var(--shadow); margin-bottom:14px;
    }
    .wide-only { display:inline }
    @media (max-width:720px) { .wide-only { display:none } }
    .conditions { display:flex; flex-wrap:wrap; gap:8px; margin:-8px 0 26px }
    .cond {
      display:inline-flex; gap:8px; align-items:center; padding:7px 12px; border-radius:999px;
      background:var(--panel-2); border:1px solid var(--line); font-size:11px; letter-spacing:.06em;
      text-transform:uppercase; font-weight:700; color:var(--muted);
    }
    .cond b { color:var(--ink); font-weight:750; letter-spacing:0; text-transform:none }
    .cond.treatment { border-color:#ffcb6655; color:var(--amber) }
    .field { position:relative }
    .field label { position:absolute; top:8px; left:12px; color:#718a81; font-size:9px; font-weight:750; letter-spacing:.12em; text-transform:uppercase; z-index:1 }
    .field select,.field input {
      width:100%; height:55px; padding:22px 11px 5px; color:var(--ink); background:#101f1b;
      border:1px solid transparent; border-radius:11px; outline:none;
    }
    .field select:focus,.field input:focus { border-color:#4ee89488 }
    .run {
      border:0; border-radius:11px; padding:0 20px; color:#07100e; background:var(--green);
      font-weight:800; cursor:pointer; transition:.2s; min-height:55px;
    }
    .run:hover { transform:translateY(-1px); background:#79ffb4 }
    .run:disabled { cursor:wait; opacity:.6 }
    .notice { min-height:22px; color:var(--muted); font-size:11px; padding:0 4px }
    .stats { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:12px 0 14px }
    .stat,.panel { border:1px solid var(--line); background:linear-gradient(145deg,#0d1916ee,#091310ee); border-radius:18px }
    .stat { padding:20px 22px; min-height:120px; position:relative; overflow:hidden }
    .stat:after { content:""; position:absolute; width:80px; height:80px; border-radius:50%; right:-35px; top:-35px; background:var(--accent,#56f39a); filter:blur(38px); opacity:.16 }
    .stat .k { color:var(--muted); font-size:10px; letter-spacing:.13em; text-transform:uppercase }
    .stat .v { display:block; margin:10px 0 4px; font-size:30px; font-weight:690; letter-spacing:-.05em }
    .stat small { color:#6f857d; font-size:11px }
    .main-grid { display:grid; grid-template-columns:minmax(0,1.55fr) minmax(300px,.65fr); gap:14px; margin-bottom:14px }
    .panel { padding:21px; min-width:0 }
    .panel-head { display:flex; justify-content:space-between; align-items:flex-start; gap:12px; margin-bottom:15px; flex-wrap:wrap }
    .panel h2 { margin:0; font-size:15px; letter-spacing:-.02em }
    .sub { margin-top:5px; color:var(--muted); font-size:11px }
    .tag { border:1px solid var(--line); border-radius:999px; padding:6px 9px; color:#9ab0a8; font-size:9px; letter-spacing:.1em; text-transform:uppercase; white-space:nowrap  max-width:100%; white-space:normal }
    #graph { width:100%; height:430px; display:block; border-radius:12px; background:#08120f; border:1px solid #1a2c27 }
    .edge { stroke:#294039; stroke-width:1.6 }
    .edge.route { stroke:#50d991; stroke-width:2.3; stroke-dasharray:6 5 }
    .edge-label { fill:#607970; font-size:10px }
    .node-card { fill:#10211c; stroke:#2b453c; stroke-width:1.5 }
    .node.visited .node-card { stroke:#56f39a; fill:#11281f }
    .node.objective .node-card { stroke:#ffcb66 }
    .node-title { fill:#ecf7f2; font-size:12px; font-weight:750 }
    .node-meta { fill:#81978f; font-size:9px }
    .node-icon { fill:#091310; stroke:#42665a }
    .node.visited .node-icon { fill:#56f39a; stroke:#56f39a }
    .node.objective .node-icon { fill:#ffcb66; stroke:#ffcb66 }
    .node.monitored .node-card { stroke:#ff6e6e; stroke-dasharray:5 3 }
    .watch circle { fill:#ff6e6e22; stroke:#ff6e6e; stroke-width:1.2 }
    .watch-mark { fill:#ff6e6e; font-size:9px; text-anchor:middle }
    .episode { display:flex; flex-direction:column }
    .outcome {
      padding:20px; border-radius:14px; margin:2px 0 16px;
      background:linear-gradient(135deg,#123323,#0c1d18); border:1px solid #2d6046;
    }
    .outcome.fail { background:linear-gradient(135deg,#382119,#1d1411); border-color:#6e4030 }
    .outcome .status { font-size:10px; color:var(--green); letter-spacing:.14em; text-transform:uppercase }
    .outcome.fail .status { color:var(--amber) }
    .outcome strong { display:block; font-size:26px; margin:8px 0 5px; letter-spacing:-.05em }
    .outcome p { margin:0; color:#88a096; font-size:11px; line-height:1.6 }
    .route { margin:0; padding:0; list-style:none }
    .route li { display:grid; grid-template-columns:26px 1fr auto; gap:10px; align-items:center; padding:11px 0; border-bottom:1px solid #192b25 }
    .route b { width:24px; height:24px; border-radius:7px; display:grid; place-items:center; background:#152922; color:var(--green); font-size:9px }
    .route span { font-size:11px; color:#bed0c9 }
    .route small { color:#657b73; font-size:9px }
    .bottom-grid { display:grid; grid-template-columns:.8fr 1.2fr; gap:14px }
    .bench-row { display:grid; grid-template-columns:88px 1fr 48px; gap:10px; align-items:center; margin:17px 0 }
    .bench-row span { color:#b8cbc4; font-size:11px }
    .bar { height:7px; border-radius:99px; background:#172923; overflow:hidden }
    .bar i { display:block; height:100%; border-radius:inherit; background:linear-gradient(90deg,var(--green-2),var(--green)); box-shadow:0 0 16px #56f39a55 }
    .bench-row strong { text-align:right; font-size:11px }
    @media (max-width:720px) {
      .bench-row { grid-template-columns:1fr auto; gap:6px 10px }
      .bench-row .bar { grid-column:1 / -1; order:3 }
      .bench-row strong { text-align:right }
      .bench-row strong br { display:none }
      .bench-row strong small { margin-left:8px }
    }
    .legend { display:flex; gap:15px; color:#70867e; font-size:9px; margin-top:17px; padding-top:13px; border-top:1px solid #192b25 }
    .table-wrap { overflow:auto; max-height:365px }
    table { width:100%; border-collapse:collapse; min-width:680px }
    th { position:sticky; top:0; background:#0d1916; color:#667e75; font-size:9px; letter-spacing:.1em; text-transform:uppercase; text-align:left; padding:10px 8px; z-index:1 }
    td { border-top:1px solid #192b25; padding:10px 8px; font-size:10px; color:#afc3bb }
    td.action { color:#ecf7f2; font-family:ui-monospace,SFMono-Regular,monospace }
    .pill { display:inline-flex; padding:4px 7px; border-radius:99px; color:#7dffb6; background:#163525; font-size:8px; text-transform:uppercase }
    .pill.invalid { color:#ffb18d; background:#352019 }
    .panel-actions { display:flex; gap:8px }
    .ghost { border:1px solid var(--line); border-radius:8px; padding:7px 10px; background:#0c1915; color:#a7bcb4; font-size:9px; cursor:pointer }
    footer { display:flex; justify-content:space-between; gap:20px; color:#667c74; font-size:10px; margin-top:22px; padding:0 4px }
    footer strong { color:#9db1aa }
    @media(max-width:980px) {
      .hero { grid-template-columns:1fr }.hero-meta { display:none }
      .toolbar { grid-template-columns:repeat(3,1fr) }.run { grid-column:span 3 }
      .main-grid,.bottom-grid { grid-template-columns:1fr }
    }
    @media(max-width:620px) {
      .shell { width:min(100% - 24px,1420px); padding-top:18px } nav { margin-bottom:38px }
      .safe { display:none } h1 { font-size:44px }
      .toolbar { grid-template-columns:1fr 1fr }.run { grid-column:span 2 }
      .stats { grid-template-columns:1fr 1fr }.stat { padding:17px; min-height:105px }
      #graph { height:350px } footer { flex-direction:column }
    }
  </style>
</head>
<body>
  <main class="shell">
    <nav>
      <div class="brand">
        <span class="mark" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none"><path d="M4 7.5 12 3l8 4.5v9L12 21l-8-4.5v-9Z" stroke="#56f39a" stroke-width="1.6"/><path d="m8 10 4-2 4 2v4l-4 2-4-2v-4Z" fill="#56f39a"/></svg></span>
        <span>RLAttack</span>
      </div>
      <div class="safe"><i></i> SIMULATION-ONLY · NO EXTERNAL TARGETS</div>
    </nav>

    <section class="hero">
      <div>
        <div class="eyebrow">Reinforcement-learning security observatory</div>
        <h1>See every decision.<br><em>Trust every run.</em></h1>
        <div class="lede">Explore deterministic attack-path policies inside a synthetic graph.
          Compare baselines, inspect rewards, and follow the exact trajectory—without touching a live system.</div>
      </div>
      <div class="hero-meta">
        <div><span>Runtime</span><strong>Gymnasium MDP</strong></div>
        <div><span>Execution</span><strong>In-process only</strong></div>
        <div><span>Graph model</span><strong>Pydantic + NetworkX</strong></div>
        <div><span>Reproducibility</span><strong>Seed locked</strong></div>
      </div>
    </section>

    <form class="toolbar" id="controls">
      <div class="field"><label for="size">Scenario size</label><select id="size"><option>small</option><option>medium</option><option>large</option></select></div>
      <div class="field"><label for="difficulty">Difficulty</label><select id="difficulty"><option>easy</option><option>medium</option><option>hard</option></select></div>
      <div class="field"><label for="agent">Policy</label><select id="agent"><option value="greedy">Greedy</option><option value="rule-based">Rule-based</option><option value="shortest-path">Graph oracle</option><option value="random">Random</option></select></div>
      <div class="field"><label for="reward">Reward</label><select id="reward"><option value="shaped">Shaped</option><option value="risk-aware">Risk-aware</option><option value="cost-aware">Cost-aware</option><option value="sparse">Sparse</option></select></div>
      <div class="field"><label for="seed">Seed</label><input id="seed" type="number" min="0" step="1"></div>
      <div class="field"><label for="budget">Step budget</label><input id="budget" type="number" min="1" max="500"></div>
      <div class="field"><label for="defender">Defender</label><select id="defender"><option value="passive">Passive (control)</option><option value="adaptive">Adaptive (uniform)</option><option value="targeted">Targeted attention</option></select></div>
      <div class="field"><label for="discovery">Discovery</label><select id="discovery"><option value="exact">Exact adjacency</option><option value="noisy">Noisy scan</option></select></div>
      <div class="field"><label for="threshold">Detection threshold</label><input id="threshold" type="number" min="0.05" max="1" step="0.05" title="Accumulated risk that ends an episode. At 0.9 detection rarely fires, so risk is not the binding constraint."></div>
      <button class="run" id="run" type="submit">Run experiment ↗</button>
    </form>
    <div class="notice" id="notice"></div>

    <section class="stats">
      <article class="stat"><span class="k">Episode status</span><strong class="v" id="stat-status">—</strong><small id="stat-status-note"></small></article>
      <article class="stat" style="--accent:#71a7ff"><span class="k">Cumulative reward</span><strong class="v" id="stat-reward">—</strong><small id="stat-reward-note"></small></article>
      <article class="stat" style="--accent:#ffcb66"><span class="k">Detection risk</span><strong class="v" id="stat-risk">—</strong><small>normalized terminal risk</small></article>
      <article class="stat" style="--accent:#c78bff"><span class="k">Graph path cost</span><strong class="v" id="stat-cost">—</strong><small>sum of traversed edge weights</small></article>
      <article class="stat" style="--accent:#ff6e6e"><span class="k">Defender responses</span><strong class="v" id="stat-defender">—</strong><small id="stat-defender-note"></small></article>
    </section>

    <section class="conditions" id="conditions"></section>

    <section class="main-grid">
      <article class="panel">
        <div class="panel-head">
          <div><h2>Scenario topology</h2><div class="sub" id="scenario-sub"></div></div>
          <span class="tag" id="graph-tag">Oracle route · dashed</span>
        </div>
        <svg id="graph" role="img" aria-label="Simulated scenario graph"></svg>
      </article>
      <article class="panel episode">
        <div class="panel-head"><div><h2>Episode briefing</h2><div class="sub">Observable, reproducible, explainable</div></div><span class="tag" id="agent-tag"></span></div>
        <div class="outcome" id="outcome"><span class="status" id="outcome-status"></span><strong id="outcome-title"></strong><p id="outcome-copy"></p></div>
        <h2>Oracle route</h2>
        <div class="sub">Shortest host-only route in the static graph</div>
        <ol class="route" id="route"></ol>
      </article>
    </section>

    <section class="bottom-grid">
      <article class="panel">
        <div class="panel-head"><div><h2>Baseline comparison</h2><div class="sub" id="benchmark-sub"></div></div><span class="tag">Generalization · per-seed graphs</span></div>
        <div id="benchmarks"></div>
        <div class="legend"><span>BAR · SUCCESS RATE</span><span>LABEL · MEAN STEPS ± SD · DETECTION RATE</span></div>
      </article>
      <article class="panel">
        <div class="panel-head">
          <div><h2>Decision trace</h2><div class="sub">Every action and immediate contribution</div></div>
          <div class="panel-actions"><button class="ghost" id="export" type="button">Export JSON</button></div>
        </div>
        <div class="table-wrap"><table><thead><tr><th>Step</th><th>Action</th><th>Target</th><th>State</th><th>Risk</th><th>Reward</th><th>Outcome</th></tr></thead><tbody id="trace"></tbody></table></div>
      </article>
    </section>

    <footer><span><strong>RLAttack</strong> · deterministic attack-path research</span><span>Local simulator · no scanners · no shells · no external network access</span></footer>
  </main>
"""


_HTML_END = """
  <script>
    const $ = (id) => document.getElementById(id);
    let model = window.__RLATTACK_DATA__;
    const esc = (value) => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const pct = (value) => `${(Number(value) * 100).toFixed(1)}%`;
    const num = (value, digits=2) => Number(value).toFixed(digits);

    function syncControls(data) {
      const c=data.config;
      $('size').value=c.size; $('difficulty').value=c.difficulty; $('agent').value=c.agent;
      $('reward').value=c.reward_strategy; $('seed').value=c.seed; $('budget').value=c.step_budget;
      $('defender').value=c.defender; $('discovery').value=c.discovery;
      $('threshold').value=c.detection_threshold;
    }

    function renderGraph(data) {
      const svg=$('graph'), nodes=data.scenario.nodes, edges=data.scenario.network_edges;
      const cols=Math.min(3,nodes.length), rows=Math.ceil(nodes.length/cols);
      const width=cols*260+50, height=Math.max(300,rows*145+55), positions={};
      nodes.forEach((node,index) => {
        const col=index%cols, row=Math.floor(index/cols);
        positions[node.id]={x:55+col*260,y:45+row*145};
      });
      const edgeSvg=edges.map((edge,index) => {
        const a=positions[edge.source], b=positions[edge.target];
        if(!a||!b) return '';
        const y1=a.y+39,y2=b.y+39,x1=a.x+176,x2=b.x;
        const sameRow=Math.abs(y1-y2)<2;
        const path=sameRow ? `M${x1} ${y1} L${x2} ${y2}` :
          `M${a.x+88} ${a.y+78} C${a.x+88} ${a.y+115},${b.x+88} ${b.y-35},${b.x+88} ${b.y}`;
        const lx=sameRow?(x1+x2)/2:(a.x+b.x)/2+88, ly=sameRow?y1-8:(a.y+b.y)/2+42;
        return `<g><path class="edge ${edge.route?'route':''}" d="${path}" fill="none"/><text class="edge-label" x="${lx}" y="${ly}">${esc(edge.cost)}</text></g>`;
      }).join('');
      const nodeSvg=nodes.map(node => {
        const p=positions[node.id], classes=`node ${node.visited?'visited':''} ${node.objective?'objective':''} ${node.monitored?'monitored':''}`;
        const role=node.entry?'ENTRY':node.objective?'OBJECTIVE':node.visited?'OBSERVED':'UNKNOWN';
        // A targeted defender is only legible if you can see where it is looking, so
        // the watched hosts are marked on the graph rather than only counted.
        const watch=node.monitored
          ? `<g class="watch" transform="translate(158 16)"><circle r="7"/><text class="watch-mark" x="0" y="4">◉</text></g>`
          : '';
        return `<g class="${classes}" transform="translate(${p.x} ${p.y})">
          <rect class="node-card" width="176" height="78" rx="12"/>
          <circle class="node-icon" cx="24" cy="25" r="7"/>
          <text class="node-title" x="40" y="29">${esc(node.label)}</text>
          <text class="node-meta" x="16" y="54">${esc(node.os.toUpperCase())} · ${esc(node.services)} SERVICES</text>
          <text class="node-meta" x="16" y="68">${role} · RISK ${esc(node.detection)}</text>
          ${watch}
        </g>`;
      }).join('');
      svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
      svg.innerHTML=edgeSvg+nodeSvg;
      const watched=nodes.filter(n=>n.monitored).length;
      $('graph-tag').textContent=watched
        ? `Oracle route · dashed  ·  ◉ ${watched} watched`
        : 'Oracle route · dashed';
    }

    function render(data) {
      model=data; syncControls(data);
      const e=data.episode, c=data.config, s=data.scenario;
      $('stat-status').textContent=e.success?'SUCCESS':e.detected?'DETECTED':e.truncated?'BUDGET':'STOPPED';
      $('stat-status-note').textContent=`${e.steps} / ${c.step_budget} steps · ${e.agent_label}`;
      $('stat-reward').textContent=(e.cumulative_reward>=0?'+':'')+num(e.cumulative_reward);
      $('stat-reward-note').textContent=`${c.reward_strategy} reward strategy`;
      $('stat-risk').textContent=pct(e.detection_risk);
      $('stat-cost').textContent=num(e.path_cost,1);
      $('stat-defender').textContent=c.defender==='passive'?'—':String(e.defender_actions);
      const watched=data.scenario.nodes.filter(n=>n.monitored).map(n=>n.label);
      $('stat-defender-note').textContent=c.defender==='passive'
        ? 'passive control condition'
        : watched.length
          ? `watching ${watched.join(', ')} · ${e.revoked_credentials} credential${e.revoked_credentials===1?'':'s'} revoked`
          : `${e.revoked_credentials} credential${e.revoked_credentials===1?'':'s'} revoked`;
      const conditions=[
        ['Dynamics', c.stochastic?'Stochastic':'Deterministic', c.stochastic],
        ['Defender', {passive:'Passive',adaptive:'Adaptive (uniform)',targeted:'Targeted attention'}[c.defender], c.defender!=='passive'],
        ['Discovery', c.discovery==='noisy'?'Noisy scan':'Exact adjacency', c.discovery==='noisy'],
        ['Detection', `threshold ${c.detection_threshold}`, c.detection_threshold<0.9],
        ['Observation', c.observation==='curriculum'?'Fixed capacity':'Scenario sized', c.observation==='curriculum'],
        ['Benchmark', data.benchmark_protocol.mode, false],
      ];
      $('conditions').innerHTML=conditions.map(([key,value,treatment]) =>
        `<span class="cond ${treatment?'treatment':''}">${esc(key)} <b>${esc(value)}</b></span>`
      ).join('');
      $('scenario-sub').textContent=`${s.id} · ${s.hosts} hosts · ${s.services} services · ${s.edges} links`;
      $('agent-tag').textContent=e.agent_label;
      const outcome=$('outcome');
      outcome.classList.toggle('fail',!e.success);
      $('outcome-status').textContent=e.success?'Objective collected':'Episode incomplete';
      $('outcome-title').textContent=e.success?'Policy reached the goal.':e.detected?'Policy tripped the detection threshold.':e.truncated?'Step budget exhausted.':'Policy stopped early.';
      $('outcome-copy').textContent=`Seed ${c.seed} produced ${e.steps} reproducible decisions with ${pct(e.detection_risk)} terminal detection risk under ${c.stochastic?'stochastic':'deterministic'} dynamics.`;
      $('benchmark-sub').textContent=`${c.benchmark_episodes} independently seeded scenarios per policy · budget ${c.step_budget}`;
      $('route').innerHTML=s.oracle_route.map((node,index) =>
        `<li><b>${String(index+1).padStart(2,'0')}</b><span>${esc(node)}</span><small>${index===0?'ENTRY':index===s.oracle_route.length-1?'GOAL':'PIVOT'}</small></li>`
      ).join('');
      $('benchmarks').innerHTML=data.benchmarks.map(metric =>
        `<div class="bench-row"><span>${esc(metric.label)}</span><div class="bar"><i style="width:${Math.max(1,metric.success_rate*100)}%"></i></div><strong>${pct(metric.success_rate)}<br><small>${num(metric.mean_steps,1)}±${num(metric.std_steps,1)} st · det ${pct(metric.detection_rate)}<span class="wide-only"> · R ${num(metric.mean_reward,1)} [${num(metric.reward_ci_low,1)}, ${num(metric.reward_ci_high,1)}]</span></small></strong></div>`
      ).join('');
      $('trace').innerHTML=e.trace.map(row =>
        `<tr><td>${row.step}</td><td class="action">${esc(row.action)}</td><td class="action">${esc(row.target_id??'—')}</td><td>${row.state.discovered_hosts}H · ${row.state.known_services}S · ${row.state.acquired_privileges}P</td><td>${pct(row.detection_risk)}</td><td>${row.reward>=0?'+':''}${num(row.reward)}</td><td><span class="pill ${row.valid&&row.outcome!=='failed'?'':'invalid'}">${esc(row.outcome)}</span></td></tr>`
      ).join('');
      renderGraph(data);
    }

    $('controls').addEventListener('submit', async (event) => {
      event.preventDefault();
      if(!window.__RLATTACK_API__) {
        $('notice').textContent='Offline report: launch `rlattack dashboard` to run new experiments here.';
        return;
      }
      const button=$('run'); button.disabled=true; button.textContent='Running…';
      const query=new URLSearchParams({
        size:$('size').value,difficulty:$('difficulty').value,agent:$('agent').value,
        reward_strategy:$('reward').value,seed:$('seed').value,step_budget:$('budget').value,
        defender:$('defender').value,discovery:$('discovery').value,
        detection_threshold:$('threshold').value,
        benchmark_episodes:model.config.benchmark_episodes
      });
      try {
        const response=await fetch(`/api/experiment?${query}`);
        if(!response.ok) throw new Error(`Experiment failed (${response.status})`);
        render(await response.json());
        $('notice').textContent='Experiment completed locally. The same inputs will reproduce this trajectory.';
      } catch(error) {
        $('notice').textContent=error.message;
      } finally {
        button.disabled=false; button.textContent='Run experiment ↗';
      }
    });

    $('export').addEventListener('click', () => {
      const blob=new Blob([JSON.stringify(model,null,2)],{type:'application/json'});
      const link=document.createElement('a'); link.href=URL.createObjectURL(blob);
      link.download=`rlattack-${model.config.seed}.json`; link.click(); URL.revokeObjectURL(link.href);
    });
    render(model);
  </script>
</body>
</html>
"""


_TRANSFER_START = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>RLAttack · Transfer</title>
  <style>
    :root {
      --bg:#07100e; --panel:#0c1815; --line:#20352f; --ink:#edf7f2; --muted:#91a69f;
      --green:#56f39a; --amber:#ffcb66; --red:#ff6e6e;
    }
    * { box-sizing:border-box }
    body {
      margin:0; padding:32px 20px 60px; background:var(--bg); color:var(--ink);
      font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;
      font-feature-settings:"tnum"; letter-spacing:-.01em;
    }
    main { width:min(1080px,100%); margin:auto }
    h1 { font-size:22px; margin:0 0 6px; letter-spacing:-.02em }
    .sub { color:var(--muted); font-size:13px; margin-bottom:26px }
    .chips { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:26px }
    .chip {
      padding:6px 12px; border-radius:999px; background:var(--panel); border:1px solid var(--line);
      font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); font-weight:700;
    }
    .chip b { color:var(--ink); text-transform:none; letter-spacing:0 }
    .wrap { overflow-x:auto; border:1px solid var(--line); border-radius:16px; background:var(--panel) }
    table { border-collapse:collapse; width:100%; min-width:760px; font-size:13px }
    th,td { padding:12px 14px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap }
    th:first-child,td:first-child { text-align:left; font-family:ui-monospace,SFMono-Regular,monospace }
    thead th {
      font-size:10px; letter-spacing:.11em; text-transform:uppercase; color:var(--muted); font-weight:750;
    }
    tbody tr:last-child td { border-bottom:0 }
    .bar { position:relative; display:block; height:6px; border-radius:99px; background:#ffffff12; margin-top:6px }
    .bar i { position:absolute; inset:0 auto 0 0; border-radius:99px; background:var(--green) }
    .sig { color:var(--amber); font-weight:750 }
    .ref { color:var(--muted) }
    footer { margin-top:22px; color:var(--muted); font-size:12px; line-height:1.6 }
  </style>
</head>
<body>
  <main>
    <h1>Transfer across scenario classes</h1>
    <div class="sub" id="sub"></div>
    <div class="chips" id="chips"></div>
    <div class="wrap">
      <table>
        <thead><tr>
          <th>Scenario class</th><th>Success</th><th>Detected</th><th>Steps</th>
          <th>Reward</th><th>95% CI</th><th>vs reference</th><th>p</th>
        </tr></thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
    <footer id="note"></footer>
  </main>
"""


_TRANSFER_END = """
  <script>
    const data = window.__RLATTACK_TRANSFER__;
    const $ = (id) => document.getElementById(id);
    const esc = (v) => String(v).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const pct = (v) => `${(Number(v) * 100).toFixed(1)}%`;
    const num = (v, d=2) => Number(v).toFixed(d);

    $('sub').textContent = `${esc(data.policy)} · ${data.seeds.length} seeds shared across every class`;
    $('chips').innerHTML = data.conditions
      .map(([k, v]) => `<span class="chip">${esc(k)} <b>${esc(v)}</b></span>`).join('');

    const tests = Object.fromEntries((data.comparisons || []).map(c => [c.candidate, c]));
    $('rows').innerHTML = data.stages.map(stage => {
      const test = tests[stage.agent_name];
      const isRef = stage.agent_name === data.reference;
      const diff = isRef ? '<span class="ref">reference</span>'
        : test ? `<span class="${test.significant ? 'sig' : ''}">${test.mean_difference >= 0 ? '+' : ''}${num(test.mean_difference)}</span>`
        : '—';
      const p = isRef ? '<span class="ref">—</span>' : test ? num(test.p_value, 4) : '—';
      return `<tr>
        <td>${esc(stage.agent_name)}</td>
        <td>${pct(stage.success_rate)}<span class="bar"><i style="width:${Math.max(1, stage.success_rate * 100)}%"></i></span></td>
        <td>${pct(stage.detection_rate)}</td>
        <td>${num(stage.mean_steps, 1)} ± ${num(stage.std_steps, 1)}</td>
        <td>${num(stage.mean_reward)}</td>
        <td>[${num(stage.reward_ci_low, 1)}, ${num(stage.reward_ci_high, 1)}]</td>
        <td>${diff}</td>
        <td>${p}</td>
      </tr>`;
    }).join('');

    $('note').textContent = data.note;
  </script>
</body>
</html>
"""
