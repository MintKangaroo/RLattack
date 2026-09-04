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
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>RLAttack · 시뮬레이션 관측소</title>
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
    .eyebrow { color:var(--green); font-size:11px; font-weight:800; letter-spacing:.08em }
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
    .howto { color:#8fb4a8; font-size:12px; line-height:1.6; padding:2px 4px 0; margin-bottom:2px }
    .howto b { color:#cfe7dd }
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
    .node.target .node-card { stroke:#4ee894; stroke-width:2.5 }
    .node.target .node-icon { fill:#4ee894; stroke:#4ee894 }
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
      <div class="safe"><i></i> 시뮬레이션 전용 · 외부 대상 없음</div>
    </nav>

    <section class="hero">
      <div>
        <div class="eyebrow">강화학습 보안 관측소</div>
        <h1>모든 결정을 본다.<br><em>모든 실행을 믿는다.</em></h1>
        <div class="lede">합성 그래프 안에서 공격 경로 정책을 실제 시스템을 건드리지 않고 탐구합니다.
          기준 정책과 비교하고, 보상을 살펴보고, 정확한 경로를 따라가 보세요.</div>
      </div>
      <div class="hero-meta">
        <div><span>런타임</span><strong>Gymnasium MDP</strong></div>
        <div><span>실행 범위</span><strong>프로세스 내부만</strong></div>
        <div><span>그래프 모델</span><strong>Pydantic + NetworkX</strong></div>
        <div><span>재현성</span><strong>Seed 고정</strong></div>
      </div>
    </section>

    <form class="toolbar" id="controls">
      <div class="field"><label for="size">시나리오 크기</label><select id="size"><option value="small">소형</option><option value="medium">중형</option><option value="large">대형</option></select></div>
      <div class="field"><label for="difficulty">난이도</label><select id="difficulty"><option value="easy">쉬움</option><option value="medium">보통</option><option value="hard">어려움</option></select></div>
      <div class="field"><label for="agent">정책</label><select id="agent"><option value="greedy">Greedy(탐욕)</option><option value="rule-based">규칙 기반</option><option value="shortest-path">그래프 오라클</option><option value="random">무작위</option></select></div>
      <div class="field"><label for="reward">보상</label><select id="reward"><option value="shaped">단계 보상</option><option value="risk-aware">위험 인지</option><option value="cost-aware">비용 인지</option><option value="sparse">희소</option></select></div>
      <div class="field"><label for="seed">Seed</label><input id="seed" type="number" min="0" step="1"></div>
      <div class="field"><label for="budget">스텝 예산</label><input id="budget" type="number" min="1" max="500"></div>
      <div class="field"><label for="defender">방어자</label><select id="defender"><option value="passive">비활성(대조군)</option><option value="adaptive">적응형(균일 감시)</option><option value="targeted">표적 주의</option></select></div>
      <div class="field"><label for="discovery">탐색</label><select id="discovery"><option value="exact">정확한 인접</option><option value="noisy">노이즈 스캔</option></select></div>
      <div class="field"><label for="threshold">탐지 임계값</label><input id="threshold" type="number" min="0.05" max="1" step="0.05" title="에피소드를 끝내는 누적 위험. 0.9에서는 탐지가 거의 발동하지 않아 위험이 구속조건이 되지 않습니다."></div>
      <div class="field"><label for="target">공격 대상</label><select id="target" title="에피소드가 도달해야 할 그래프 안의 목표. 시뮬레이션 전용 — 외부 주소는 없습니다."><option value="">전체 목표</option></select></div>
      <button class="run" id="run" type="submit">실험 실행 ↗</button>
    </form>
    <div class="howto">조건을 고른 뒤 <b>실험 실행</b>을 누르면 같은 seed로 즉시 다시 계산됩니다. 아래에 에피소드 결과·그래프·기준 비교·결정 로그가 나타납니다.</div>
    <div class="notice" id="notice"></div>

    <section class="stats">
      <article class="stat"><span class="k">에피소드 상태</span><strong class="v" id="stat-status">—</strong><small id="stat-status-note"></small></article>
      <article class="stat" style="--accent:#71a7ff"><span class="k">누적 보상</span><strong class="v" id="stat-reward">—</strong><small id="stat-reward-note"></small></article>
      <article class="stat" style="--accent:#ffcb66"><span class="k">탐지 위험</span><strong class="v" id="stat-risk">—</strong><small>정규화된 종료 시점 위험</small></article>
      <article class="stat" style="--accent:#c78bff"><span class="k">그래프 경로 비용</span><strong class="v" id="stat-cost">—</strong><small>지나간 엣지 가중치 합</small></article>
      <article class="stat" style="--accent:#ff6e6e"><span class="k">방어자 대응</span><strong class="v" id="stat-defender">—</strong><small id="stat-defender-note"></small></article>
    </section>

    <section class="conditions" id="conditions"></section>

    <section class="main-grid">
      <article class="panel">
        <div class="panel-head">
          <div><h2>시나리오 토폴로지</h2><div class="sub" id="scenario-sub"></div></div>
          <span class="tag" id="graph-tag">오라클 경로 · 점선</span>
        </div>
        <svg id="graph" role="img" aria-label="시뮬레이션 시나리오 그래프"></svg>
      </article>
      <article class="panel episode">
        <div class="panel-head"><div><h2>에피소드 브리핑</h2><div class="sub">관측 가능 · 재현 가능 · 설명 가능</div></div><span class="tag" id="agent-tag"></span></div>
        <div class="outcome" id="outcome"><span class="status" id="outcome-status"></span><strong id="outcome-title"></strong><p id="outcome-copy"></p></div>
        <h2>오라클 경로</h2>
        <div class="sub">정적 그래프에서의 호스트 최단 경로</div>
        <ol class="route" id="route"></ol>
      </article>
    </section>

    <section class="bottom-grid">
      <article class="panel">
        <div class="panel-head"><div><h2>기준 정책 비교</h2><div class="sub" id="benchmark-sub"></div></div><span class="tag">일반화 · seed별 그래프</span></div>
        <div id="benchmarks"></div>
        <div class="legend"><span>막대 · 성공률</span><span>라벨 · 평균 스텝 ± 표준편차 · 탐지율</span></div>
      </article>
      <article class="panel">
        <div class="panel-head">
          <div><h2>결정 로그</h2><div class="sub">모든 action과 즉시 기여</div></div>
          <div class="panel-actions"><button class="ghost" id="export" type="button">JSON 내보내기</button></div>
        </div>
        <div class="table-wrap"><table><thead><tr><th>스텝</th><th>Action</th><th>대상</th><th>상태</th><th>위험</th><th>보상</th><th>결과</th></tr></thead><tbody id="trace"></tbody></table></div>
      </article>
    </section>

    <footer><span><strong>RLAttack</strong> · 결정론적 공격 경로 연구</span><span>로컬 시뮬레이터 · 스캐너·셸·외부 네트워크 없음</span></footer>
  </main>
"""


_HTML_END = """
  <script>
    const $ = (id) => document.getElementById(id);
    let model = window.__RLATTACK_DATA__;
    const REWARD_KO={shaped:'단계',"risk-aware":'위험 인지',"cost-aware":'비용 인지',sparse:'희소'};
    const esc = (value) => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const pct = (value) => `${(Number(value) * 100).toFixed(1)}%`;
    const num = (value, digits=2) => Number(value).toFixed(digits);

    function syncControls(data) {
      const c=data.config;
      $('size').value=c.size; $('difficulty').value=c.difficulty; $('agent').value=c.agent;
      $('reward').value=c.reward_strategy; $('seed').value=c.seed; $('budget').value=c.step_budget;
      $('defender').value=c.defender; $('discovery').value=c.discovery;
      $('threshold').value=c.detection_threshold;
      syncTargets(data.target);
    }

    function syncTargets(target) {
      const select=$('target');
      const options=['<option value="">All objectives</option>'].concat(
        (target.available||[]).map(t =>
          `<option value="${esc(t.id)}">${esc(t.label)} · ${esc(t.privilege)}</option>`)
      );
      select.innerHTML=options.join('');
      select.value=target.selected||'';
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
        const p=positions[node.id], classes=`node ${node.visited?'visited':''} ${node.objective?'objective':''} ${node.monitored?'monitored':''} ${node.target?'target':''}`;
        const role=node.entry?'진입':node.target?'대상':node.objective?'목표':node.visited?'관측됨':'미상';
        // A targeted defender is only legible if you can see where it is looking, so
        // the watched hosts are marked on the graph rather than only counted.
        const watch=node.monitored
          ? `<g class="watch" transform="translate(158 16)"><circle r="7"/><text class="watch-mark" x="0" y="4">◉</text></g>`
          : '';
        return `<g class="${classes}" transform="translate(${p.x} ${p.y})">
          <rect class="node-card" width="176" height="78" rx="12"/>
          <circle class="node-icon" cx="24" cy="25" r="7"/>
          <text class="node-title" x="40" y="29">${esc(node.label)}</text>
          <text class="node-meta" x="16" y="54">${esc(node.os.toUpperCase())} · 서비스 ${esc(node.services)}</text>
          <text class="node-meta" x="16" y="68">${role} · 위험 ${esc(node.detection)}</text>
          ${watch}
        </g>`;
      }).join('');
      svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
      svg.innerHTML=edgeSvg+nodeSvg;
      const watched=nodes.filter(n=>n.monitored).length;
      const targetNode=nodes.find(n=>n.target);
      let tag='오라클 경로 · 점선';
      if(targetNode) tag+=`  ·  ⌖ 대상 ${targetNode.label}`;
      if(watched) tag+=`  ·  ◉ ${watched}곳 감시`;
      $('graph-tag').textContent=tag;
    }

    function render(data) {
      model=data; syncControls(data);
      const e=data.episode, c=data.config, s=data.scenario;
      $('stat-status').textContent=e.success?'성공':e.detected?'탐지됨':e.truncated?'예산 소진':'중단';
      $('stat-status-note').textContent=`${e.steps} / ${c.step_budget} 스텝 · ${e.agent_label}`;
      $('stat-reward').textContent=(e.cumulative_reward>=0?'+':'')+num(e.cumulative_reward);
      $('stat-reward-note').textContent=`${REWARD_KO[c.reward_strategy]||c.reward_strategy} 보상 전략`;
      $('stat-risk').textContent=pct(e.detection_risk);
      $('stat-cost').textContent=num(e.path_cost,1);
      $('stat-defender').textContent=c.defender==='passive'?'—':String(e.defender_actions);
      const watched=data.scenario.nodes.filter(n=>n.monitored).map(n=>n.label);
      $('stat-defender-note').textContent=c.defender==='passive'
        ? '비활성 대조군'
        : watched.length
          ? `${watched.join(', ')} 감시 · credential ${e.revoked_credentials}개 회수`
          : `credential ${e.revoked_credentials}개 회수`;
      const conditions=[
        ['동역학', c.stochastic?'확률적':'결정론적', c.stochastic],
        ['방어자', {passive:'비활성',adaptive:'적응형(균일)',targeted:'표적 주의'}[c.defender], c.defender!=='passive'],
        ['탐색', c.discovery==='noisy'?'노이즈 스캔':'정확한 인접', c.discovery==='noisy'],
        ['탐지', `임계값 ${c.detection_threshold}`, c.detection_threshold<0.9],
        ['관측', c.observation==='curriculum'?'고정 용량':'시나리오 크기', c.observation==='curriculum'],
        ['벤치마크', data.benchmark_protocol.mode, false],
      ];
      $('conditions').innerHTML=conditions.map(([key,value,treatment]) =>
        `<span class="cond ${treatment?'treatment':''}">${esc(key)} <b>${esc(value)}</b></span>`
      ).join('');
      $('scenario-sub').textContent=`${s.id} · 호스트 ${s.hosts} · 서비스 ${s.services} · 링크 ${s.edges}`;
      $('agent-tag').textContent=e.agent_label;
      const outcome=$('outcome');
      outcome.classList.toggle('fail',!e.success);
      $('outcome-status').textContent=e.success?'목표 수집됨':'에피소드 미완료';
      $('outcome-title').textContent=e.success?'정책이 목표에 도달했습니다.':e.detected?'정책이 탐지 임계값을 넘었습니다.':e.truncated?'스텝 예산을 소진했습니다.':'정책이 일찍 멈췄습니다.';
      $('outcome-copy').textContent=`Seed ${c.seed}는 ${c.stochastic?'확률적':'결정론적'} 동역학에서 ${e.steps}개의 재현 가능한 결정과 종료 시점 탐지 위험 ${pct(e.detection_risk)}를 만들었습니다.`;
      $('benchmark-sub').textContent=`정책당 독립 seed 시나리오 ${c.benchmark_episodes}개 · 예산 ${c.step_budget}`;
      $('route').innerHTML=s.oracle_route.map((node,index) =>
        `<li><b>${String(index+1).padStart(2,'0')}</b><span>${esc(node)}</span><small>${index===0?'진입':index===s.oracle_route.length-1?'목표':'경유'}</small></li>`
      ).join('');
      $('benchmarks').innerHTML=data.benchmarks.map(metric =>
        `<div class="bench-row"><span>${esc(metric.label)}</span><div class="bar"><i style="width:${Math.max(1,metric.success_rate*100)}%"></i></div><strong>${pct(metric.success_rate)}<br><small>${num(metric.mean_steps,1)}±${num(metric.std_steps,1)} 스텝 · 탐지 ${pct(metric.detection_rate)}<span class="wide-only"> · R ${num(metric.mean_reward,1)} [${num(metric.reward_ci_low,1)}, ${num(metric.reward_ci_high,1)}]</span></small></strong></div>`
      ).join('');
      $('trace').innerHTML=e.trace.map(row =>
        `<tr><td>${row.step}</td><td class="action">${esc(row.action)}</td><td class="action">${esc(row.target_id??'—')}</td><td>${row.state.discovered_hosts}H · ${row.state.known_services}S · ${row.state.acquired_privileges}P</td><td>${pct(row.detection_risk)}</td><td>${row.reward>=0?'+':''}${num(row.reward)}</td><td><span class="pill ${row.valid&&row.outcome!=='failed'?'':'invalid'}">${esc(row.outcome)}</span></td></tr>`
      ).join('');
      renderGraph(data);
    }

    $('controls').addEventListener('submit', async (event) => {
      event.preventDefault();
      if(!window.__RLATTACK_API__) {
        $('notice').textContent='오프라인 리포트입니다. 새 실험을 돌리려면 `rlattack dashboard`로 실행하세요.';
        return;
      }
      const button=$('run'); button.disabled=true; button.textContent='실행 중…';
      const query=new URLSearchParams({
        size:$('size').value,difficulty:$('difficulty').value,agent:$('agent').value,
        reward_strategy:$('reward').value,seed:$('seed').value,step_budget:$('budget').value,
        defender:$('defender').value,discovery:$('discovery').value,
        detection_threshold:$('threshold').value,target:$('target').value,
        benchmark_episodes:model.config.benchmark_episodes
      });
      try {
        const response=await fetch(`/api/experiment?${query}`);
        if(!response.ok) throw new Error(`실험 실패 (${response.status})`);
        render(await response.json());
        $('notice').textContent='로컬에서 실험을 완료했습니다. 같은 입력은 이 경로를 그대로 재현합니다.';
      } catch(error) {
        $('notice').textContent=error.message;
      } finally {
        button.disabled=false; button.textContent='실험 실행 ↗';
      }
    });

    $('export').addEventListener('click', () => {
      const blob=new Blob([JSON.stringify(model,null,2)],{type:'application/json'});
      const link=document.createElement('a'); link.href=URL.createObjectURL(blob);
      link.download=`rlattack-${model.config.seed}.json`; link.click(); URL.revokeObjectURL(link.href);
    });
    // The available objectives depend on the scenario, so a target chosen for one size
    // may not exist in another. Reset it rather than submit a target the server rejects.
    ['size','difficulty'].forEach(id => $(id).addEventListener('change', () => { $('target').value=''; }));
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
