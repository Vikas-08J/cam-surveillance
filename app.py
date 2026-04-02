import cv2
import threading
import time
import os
from flask import Flask, Response, jsonify, render_template_string
import main as pipeline

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))

# ─── HTML Dashboard ───────────────────────────────────────────────────────────
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Surveillance Live Monitor</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg:      #060e1a;
    --panel:   #0a1422;
    --border:  #1a2a1a;
    --green:   #00ff88;
    --green2:  #00cc66;
    --red:     #ff3355;
    --amber:   #ffaa00;
    --blue:    #00aaff;
    --dim:     #2a3a2a;
    --text:    #c8e6c8;
    --mono:    'Share Tech Mono', monospace;
    --sans:    'Rajdhani', sans-serif;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    overflow-x: hidden;
  }

  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
      repeating-linear-gradient(
        0deg, transparent, transparent 2px,
        rgba(0,255,136,0.012) 2px, rgba(0,255,136,0.012) 4px
      ),
      repeating-linear-gradient(
        90deg,
        rgba(255,255,255,0.018) 0px, rgba(255,255,255,0.018) 1px,
        transparent 1px, transparent 40px
      ),
      repeating-linear-gradient(
        0deg,
        rgba(255,255,255,0.018) 0px, rgba(255,255,255,0.018) 1px,
        transparent 1px, transparent 40px
      );
    pointer-events: none;
    z-index: 9999;
  }

  /* ── Header ── */
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 28px;
    border-bottom: 1px solid var(--border);
    background: var(--panel);
    position: sticky;
    top: 0;
    z-index: 100;
  }

  .logo { display: flex; align-items: center; gap: 12px; }

  .logo-icon {
    width: 36px; height: 36px;
    border: 2px solid var(--green);
    border-radius: 50%;
    display: grid; place-items: center;
    animation: pulse-ring 2.5s infinite;
  }
  @keyframes pulse-ring {
    0%, 100% { box-shadow: 0 0 0 0 rgba(0,255,136,0.4); }
    50%       { box-shadow: 0 0 0 8px rgba(0,255,136,0); }
  }

  .logo-icon svg { width: 18px; height: 18px; fill: var(--green); }

  h1 {
    font-family: var(--sans);
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    color: var(--green);
    text-transform: uppercase;
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: 16px;
    font-family: var(--mono);
    font-size: 0.75rem;
  }

  #clock { color: var(--green2); letter-spacing: 0.1em; }

  .badge {
    padding: 3px 10px;
    border-radius: 2px;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    font-weight: 600;
  }
  .badge-idle  { background: #222; color: #666; border: 1px solid #333; }
  .badge-live  { background: var(--red);   color: #fff; animation: blink 1.2s step-end infinite; }
  .badge-done  { background: var(--green); color: #000; }

  @keyframes blink { 50% { opacity: 0.3; } }

  /* ── START BUTTON ── */
  #start-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 20px;
    background: transparent;
    border: 2px solid var(--green);
    border-radius: 3px;
    color: var(--green);
    font-family: var(--mono);
    font-size: 0.78rem;
    letter-spacing: 0.18em;
    cursor: pointer;
    transition: background 0.2s, color 0.2s, box-shadow 0.2s;
    text-transform: uppercase;
    position: relative;
    overflow: hidden;
  }
  #start-btn:hover:not(:disabled) {
    background: var(--green);
    color: #000;
    box-shadow: 0 0 16px rgba(0,255,136,0.4);
  }
  #start-btn:disabled {
    border-color: #333;
    color: #444;
    cursor: not-allowed;
  }
  #start-btn .btn-icon { font-size: 1rem; }

  /* Spinning state */
  #start-btn.running .btn-icon {
    display: inline-block;
    animation: spin 1s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Layout ── */
  .main-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    padding: 16px 20px;
    max-width: 1600px;
    margin: 0 auto;
  }

  /* ── Idle overlay — full viewport centred card ── */
  #idle-overlay {
    grid-column: 1 / -1;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: calc(100vh - 80px);   /* fill below header */
    background: transparent;
    padding: 40px 20px;
  }

  .idle-card {
    background: linear-gradient(145deg, #0d1b2e 0%, #0a1628 60%, #0c1f35 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 56px 64px 48px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 32px;
    max-width: 680px;
    width: 100%;
    box-shadow:
      0 32px 80px rgba(0,0,0,0.6),
      0 0 0 1px rgba(255,255,255,0.04) inset;
    position: relative;
    overflow: hidden;
  }

  /* Subtle grid pattern inside card */
  .idle-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background-image:
      linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
    background-size: 32px 32px;
    border-radius: 24px;
    pointer-events: none;
  }

  .idle-card-title {
    display: flex;
    align-items: center;
    gap: 16px;
    font-family: var(--sans);
    font-size: 2rem;
    font-weight: 700;
    color: #e8f0ff;
    letter-spacing: 0.02em;
    text-align: center;
  }
  .idle-card-title .emoji { font-size: 1.8rem; }

  /* Status pill */
  .idle-status-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 18px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 999px;
    font-family: var(--mono);
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    color: #a0b4cc;
  }
  .idle-status-pill .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #00cc66;
    box-shadow: 0 0 6px #00cc66;
    animation: pulse-dot 2s ease-in-out infinite;
  }
  @keyframes pulse-dot {
    0%, 100% { opacity: 1;   transform: scale(1);    }
    50%       { opacity: 0.5; transform: scale(0.85); }
  }

  /* Big start button */
  #idle-start-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 18px;
    width: 100%;
    max-width: 440px;
    padding: 22px 40px;
    background: linear-gradient(135deg, #1a3a5c 0%, #1e4570 50%, #1a3a5c 100%);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 999px;
    cursor: pointer;
    transition: all 0.25s ease;
    box-shadow:
      0 8px 32px rgba(0,0,0,0.4),
      0 1px 0 rgba(255,255,255,0.06) inset;
    position: relative;
    overflow: hidden;
  }
  #idle-start-btn::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg,
      rgba(255,255,255,0.06) 0%,
      transparent 60%);
    border-radius: 999px;
    pointer-events: none;
  }
  #idle-start-btn:hover {
    transform: translateY(-2px);
    background: linear-gradient(135deg, #1f4570 0%, #245280 50%, #1f4570 100%);
    box-shadow:
      0 16px 48px rgba(0,80,160,0.4),
      0 1px 0 rgba(255,255,255,0.1) inset;
  }
  #idle-start-btn:active {
    transform: translateY(0px);
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
  }

  .idle-btn-circle {
    width: 42px; height: 42px;
    border-radius: 50%;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    display: grid; place-items: center;
    flex-shrink: 0;
    font-size: 1rem;
    color: #c0d8f0;
    transition: background 0.2s;
  }
  #idle-start-btn:hover .idle-btn-circle {
    background: rgba(255,255,255,0.14);
  }

  .idle-btn-text {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 2px;
  }
  .idle-btn-text .btn-main {
    font-family: var(--sans);
    font-size: 1.25rem;
    font-weight: 700;
    color: #e8f4ff;
    letter-spacing: 0.02em;
    line-height: 1.1;
  }
  .idle-btn-text .btn-sub {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: #6a8aaa;
    letter-spacing: 0.1em;
  }

  /* Feature badges row */
  .idle-badges {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    justify-content: center;
  }

  .idle-badge {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 7px 16px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 999px;
    font-family: var(--mono);
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    color: #7a9ab8;
    transition: border-color 0.2s;
  }
  .idle-badge:hover { border-color: rgba(255,255,255,0.2); }
  .idle-badge .badge-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
  }
  .idle-badge .badge-dot.green  { background: #00cc66; box-shadow: 0 0 4px #00cc66; }
  .idle-badge .badge-dot.blue   { background: #4488ff; box-shadow: 0 0 4px #4488ff; }
  .idle-badge .badge-dot.amber  { background: #ffaa00; box-shadow: 0 0 4px #ffaa00; }

  .idle-footer-text {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: #3a5060;
    letter-spacing: 0.08em;
    text-align: center;
  }

  /* ── Camera grid (hidden until pipeline starts) ── */
  #cam-grid {
    display: contents;   /* participates in main-grid */
  }
  #cam-grid.hidden { display: none; }

  /* ── Feed panels ── */
  .feed-panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
  }

  .feed-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    background: rgba(0,0,0,0.4);
  }

  .feed-title {
    font-family: var(--mono);
    font-size: 0.8rem;
    letter-spacing: 0.15em;
    color: var(--green);
  }

  .feed-meta {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--dim);
  }

  .video-wrap {
    position: relative;
    background: #000;
    aspect-ratio: 16/9;
    display: flex; align-items: center; justify-content: center;
    overflow: hidden;
  }
  .video-wrap img { width: 100%; height: 100%; object-fit: contain; display: block; }

  .video-wrap::before, .video-wrap::after {
    content: '';
    position: absolute;
    width: 24px; height: 24px;
    border-color: var(--green);
    border-style: solid;
    opacity: 0.6;
    z-index: 2;
  }
  .video-wrap::before { top: 8px;    left: 8px;   border-width: 2px 0 0 2px; }
  .video-wrap::after  { bottom: 8px; right: 8px;  border-width: 0 2px 2px 0; }

  .cam-label {
    position: absolute;
    bottom: 10px; left: 10px;
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--green);
    letter-spacing: 0.1em;
    z-index: 3;
    text-shadow: 0 0 6px var(--green);
  }

  .feed-footer {
    display: flex;
    gap: 16px;
    padding: 10px 16px;
    border-top: 1px solid var(--border);
  }

  .stat-chip { display: flex; flex-direction: column; gap: 2px; }

  .stat-chip .label {
    font-size: 0.62rem;
    color: #4a6a4a;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-family: var(--mono);
  }

  .stat-chip .value {
    font-family: var(--mono);
    font-size: 1.1rem;
    color: var(--green);
    line-height: 1;
  }

  /* ── Stats bar ── */
  .stats-bar {
    grid-column: 1 / -1;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
  }

  .stat-block {
    background: var(--panel);
    padding: 18px 24px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    position: relative;
    overflow: hidden;
  }
  .stat-block::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--green);
    opacity: 0.5;
  }
  .stat-block.red::after   { background: var(--red);   }
  .stat-block.amber::after { background: var(--amber); }

  .stat-block .label {
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #4a6a4a;
    font-family: var(--mono);
  }

  .stat-block .big {
    font-size: 3rem;
    font-weight: 700;
    font-family: var(--mono);
    color: var(--green);
    line-height: 1;
    transition: all 0.3s;
  }
  .stat-block.red   .big { color: var(--red);   }
  .stat-block.amber .big { color: var(--amber); }

  .stat-block .sub {
    font-size: 0.7rem;
    color: #4a6a4a;
    font-family: var(--mono);
  }

  /* ── Progress bar ── */
  .progress-wrap {
    grid-column: 1 / -1;
    height: 3px;
    background: var(--dim);
    border-radius: 2px;
    overflow: hidden;
  }
  #progress-bar {
    height: 100%;
    background: linear-gradient(90deg, var(--green2), var(--green));
    width: 0%;
    transition: width 0.6s ease;
    box-shadow: 0 0 8px var(--green);
  }

  /* ── Log bar ── */
  .log-bar {
    grid-column: 1 / -1;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 12px 16px;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .log-label {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--green2);
    letter-spacing: 0.15em;
    white-space: nowrap;
  }
  #log-text {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: #4a6a4a;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
  }

  /* ── Number flip animation ── */
  @keyframes flip-in {
    from { opacity: 0; transform: translateY(-8px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .updated { animation: flip-in 0.3s ease; }
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">
      <svg viewBox="0 0 24 24">
        <path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11
                 11-7.5c-1.73-4.39-6-7.5-11-7.5zm0 12.5c-2.76 0-5-2.24-5-5s2.24-5
                 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3
                 3-1.34 3-3-1.34-3-3-3z"/>
      </svg>
    </div>
    <div><h1>Surveillance Monitor</h1></div>
  </div>
  <div class="header-right">
    <span id="clock">--:--:--</span>
    <span class="badge badge-idle" id="status-badge">● IDLE</span>
    <!-- Header Start / Restart button -->
    <button id="start-btn" onclick="startPipeline()">
      <span class="btn-icon">▶</span>
      <span id="start-btn-label">START</span>
    </button>
  </div>
</header>

<div class="main-grid">

  <!-- Progress bar (always visible) -->
  <div class="progress-wrap" style="grid-column:1/-1">
    <div id="progress-bar"></div>
  </div>

  <!-- ── IDLE OVERLAY — shown before first start ── -->
  <div id="idle-overlay">
    <div class="idle-card">

      <!-- Title -->
      <div class="idle-card-title">
        <span class="emoji">📷</span>
        Video Surveillance System
        <span class="emoji">🎥</span>
      </div>

      <!-- Status pill -->
      <div class="idle-status-pill">
        <span class="dot"></span>
        SYSTEM READY · IDLE
      </div>

      <!-- Big Start button -->
      <button id="idle-start-btn" onclick="startPipeline()">
        <div class="idle-btn-circle">⏺</div>
        <div class="idle-btn-text">
          <span class="btn-main">Start Surveillance</span>
          <span class="btn-sub">CLICK TO ACTIVATE</span>
        </div>
      </button>

      <!-- Feature badges -->
      <div class="idle-badges">
        <div class="idle-badge">
          🔒&nbsp; Encrypted Stream
        </div>
        <div class="idle-badge">
          <span class="badge-dot blue"></span>
          24/7 Recording Ready
        </div>
        <div class="idle-badge">
          🔔&nbsp; Motion Detection
        </div>
      </div>

      <!-- Footer hint -->
      <div class="idle-footer-text">
        Secure video pipeline &bull; Click start to activate monitoring
      </div>

    </div>
  </div>

  <!-- ── CAMERA GRID — shown after start ── -->
  <div id="cam-grid" class="hidden">

    <!-- CAM 1 -->
    <div class="feed-panel">
      <div class="feed-header">
        <span class="feed-title">◈ CAM-01 / ENTRANCE</span>
        <span class="feed-meta" id="cam1-status">WAITING</span>
      </div>
      <div class="video-wrap">
        <img id="cam1-img" alt="Camera 1 Feed" />
        <div class="cam-label">CAM-01 ● REC</div>
      </div>
      <div class="feed-footer">
        <div class="stat-chip">
          <span class="label">Active Now</span>
          <span class="value" id="cam1-active">0</span>
        </div>
        <div class="stat-chip">
          <span class="label">Total Seen</span>
          <span class="value" id="cam1-total">0</span>
        </div>
      </div>
    </div>

    <!-- CAM 2 -->
    <div class="feed-panel">
      <div class="feed-header">
        <span class="feed-title">◈ CAM-02 / EXIT</span>
        <span class="feed-meta" id="cam2-status">WAITING</span>
      </div>
      <div class="video-wrap">
        <img id="cam2-img" alt="Camera 2 Feed" />
        <div class="cam-label">CAM-02 ● REC</div>
      </div>
      <div class="feed-footer">
        <div class="stat-chip">
          <span class="label">Active Now</span>
          <span class="value" id="cam2-active">0</span>
        </div>
        <div class="stat-chip">
          <span class="label">Total Seen</span>
          <span class="value" id="cam2-total">0</span>
        </div>
      </div>
    </div>

    <!-- Bottom stat cards -->
    <div class="stats-bar">
      <div class="stat-block">
        <span class="label">CAM-01 Persons</span>
        <span class="big" id="s-cam1">0</span>
        <span class="sub">total detected</span>
      </div>
      <div class="stat-block">
        <span class="label">CAM-02 Persons</span>
        <span class="big" id="s-cam2">0</span>
        <span class="sub">total detected</span>
      </div>
      <div class="stat-block amber">
        <span class="label">Cross-Cam Matches</span>
        <span class="big" id="s-matches">0</span>
        <span class="sub">same person, both cams</span>
      </div>
      <div class="stat-block red">
        <span class="label">Total Unique People</span>
        <span class="big" id="s-unique">0</span>
        <span class="sub">across all cameras</span>
      </div>
    </div>

    <!-- Log bar -->
    <div class="log-bar">
      <span class="log-label">SYS ›</span>
      <span id="log-text">Initializing pipeline…</span>
    </div>

  </div><!-- /#cam-grid -->

</div><!-- /.main-grid -->

<script>
  // ── Clock ──────────────────────────────────────────────────────────────────
  function updateClock() {
    document.getElementById('clock').textContent =
      new Date().toLocaleTimeString('en-GB', { hour12: false });
  }
  setInterval(updateClock, 1000);
  updateClock();

  // ── Animate value change ───────────────────────────────────────────────────
  function setVal(id, val) {
    const el = document.getElementById(id);
    if (!el) return;
    const str = String(val);
    if (el.textContent !== str) {
      el.textContent = str;
      el.classList.remove('updated');
      void el.offsetWidth;
      el.classList.add('updated');
    }
  }

  function log(msg) {
    const el = document.getElementById('log-text');
    if (el) el.textContent = msg;
  }

  // ── UI state machine ───────────────────────────────────────────────────────
  // States: 'idle' | 'running' | 'done'
  let uiState   = 'idle';
  let pollTimer = null;
  let prevStats = {};

  function setUIState(state) {
    uiState = state;
    const badge     = document.getElementById('status-badge');
    const btn       = document.getElementById('start-btn');
    const btnLabel  = document.getElementById('start-btn-label');
    const btnIcon   = btn.querySelector('.btn-icon');
    const idleOv    = document.getElementById('idle-overlay');
    const camGrid   = document.getElementById('cam-grid');

    if (state === 'idle') {
      badge.textContent = '● IDLE';
      badge.className   = 'badge badge-idle';
      btn.disabled      = false;
      btn.classList.remove('running');
      btnIcon.textContent  = '▶';
      btnLabel.textContent = 'START';
      idleOv.style.display = 'flex';
      camGrid.classList.add('hidden');
      document.getElementById('progress-bar').style.width = '0%';
    }

    if (state === 'running') {
      badge.textContent = '● LIVE';
      badge.className   = 'badge badge-live';
      btn.disabled      = true;
      btn.classList.add('running');
      btnIcon.textContent  = '↻';
      btnLabel.textContent = 'RUNNING';
      idleOv.style.display = 'none';
      camGrid.classList.remove('hidden');
      // Connect MJPEG streams NOW (avoids loading before pipeline starts)
      document.getElementById('cam1-img').src = '/stream/cam1?t=' + Date.now();
      document.getElementById('cam2-img').src = '/stream/cam2?t=' + Date.now();
      log('Pipeline started — processing both cameras in parallel…');
    }

    if (state === 'done') {
      badge.textContent = '✓ DONE';
      badge.className   = 'badge badge-done';
      btn.disabled      = false;
      btn.classList.remove('running');
      btnIcon.textContent  = '↺';
      btnLabel.textContent = 'RESTART';
      document.getElementById('progress-bar').style.width = '100%';
      document.getElementById('cam1-status').textContent  = 'COMPLETE';
      document.getElementById('cam2-status').textContent  = 'COMPLETE';
    }
  }

  // ── Start pipeline ─────────────────────────────────────────────────────────
  async function startPipeline() {
    if (uiState === 'running') return;

    // Reset all displayed values
    ['cam1-active','cam1-total','cam2-active','cam2-total',
     's-cam1','s-cam2','s-matches','s-unique'].forEach(id => setVal(id, 0));
    prevStats = {};

    setUIState('running');

    try {
      const res = await fetch('/api/start', { method: 'POST' });
      const d   = await res.json();
      if (!d.ok) {
        log('Error: ' + (d.error || 'could not start pipeline'));
        setUIState('idle');
        return;
      }
    } catch (e) {
      log('Could not reach server — is Flask running?');
      setUIState('idle');
      return;
    }

    // Begin polling
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(pollStats, 500);
    pollStats();
  }

  // ── Stats polling ──────────────────────────────────────────────────────────
  async function pollStats() {
    try {
      const res  = await fetch('/api/stats');
      const data = await res.json();

      setVal('cam1-active', data.cam1_active);
      setVal('cam1-total',  data.cam1_total);
      setVal('cam2-active', data.cam2_active);
      setVal('cam2-total',  data.cam2_total);
      setVal('s-cam1',      data.cam1_total);
      setVal('s-cam2',      data.cam2_total);
      setVal('s-matches',   data.matches);
      setVal('s-unique',    data.unique);

      document.getElementById('cam1-status').textContent =
        data.cam1_done ? 'COMPLETE' : 'PROCESSING';
      document.getElementById('cam2-status').textContent =
        data.cam2_done ? 'COMPLETE' : 'PROCESSING';

      const done = (data.cam1_done ? 1 : 0) + (data.cam2_done ? 1 : 0);
      if (!data.pipeline_done)
        document.getElementById('progress-bar').style.width =
          `${done * 40 + 10}%`;

      // Log messages
      if (data.cam1_total !== prevStats.cam1_total)
        log(`CAM-01 › New person #${data.cam1_total} detected! Matches: ${data.matches} | Unique: ${data.unique}`);
      else if (data.cam2_total !== prevStats.cam2_total)
        log(`CAM-02 › New person #${data.cam2_total} detected! Matches: ${data.matches} | Unique: ${data.unique}`);
      else if (data.matches !== prevStats.matches)
        log(`Cross-camera match found — ${data.matches} person(s) seen in both feeds. Unique: ${data.unique}`);
      else if (uiState === 'running')
        log(`Processing… CAM-01: ${data.cam1_total} | CAM-02: ${data.cam2_total} | Matches: ${data.matches} | Unique: ${data.unique}`);

      prevStats = { ...data };

      if (data.pipeline_done && uiState !== 'done') {
        setUIState('done');
        clearInterval(pollTimer);
        log(`✓ Pipeline complete — CAM-01: ${data.cam1_total} | CAM-02: ${data.cam2_total} | Matches: ${data.matches} | Total Unique: ${data.unique}`);
      }

    } catch (e) {
      log('Polling error — retrying…');
    }
  }

  // ── MJPEG reconnect on error ───────────────────────────────────────────────
  ['cam1-img', 'cam2-img'].forEach(id => {
    const img = document.getElementById(id);
    if (!img) return;
    img.addEventListener('error', () => {
      if (uiState === 'running') {
        setTimeout(() => {
          img.src = img.src.split('?')[0] + '?t=' + Date.now();
        }, 2000);
      }
    });
  });
</script>
</body>
</html>
"""


# ─── Pipeline state ───────────────────────────────────────────────────────────
_pipeline_lock    = threading.Lock()   # prevents double-start
_pipeline_started = False


# ─── MJPEG helpers ────────────────────────────────────────────────────────────
PLACEHOLDER = None

def _placeholder_frame():
    global PLACEHOLDER
    if PLACEHOLDER is None:
        import numpy as np
        img = cv2.imencode(
            '.jpg',
            np.zeros((360, 640, 3), dtype='uint8'),
            [cv2.IMWRITE_JPEG_QUALITY, 60]
        )[1]
        PLACEHOLDER = img.tobytes()
    return PLACEHOLDER


def _gen_stream(cam: str):
    while True:
        frame = (pipeline.latest_frame_cam1
                 if cam == "cam1"
                 else pipeline.latest_frame_cam2)
        if frame is not None:
            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            jpeg = buf.tobytes() if ok else _placeholder_frame()
        else:
            jpeg = _placeholder_frame()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
        time.sleep(0.07)   # ~14 fps


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route('/stream/cam1')
def stream_cam1():
    return Response(_gen_stream('cam1'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stream/cam2')
def stream_cam2():
    return Response(_gen_stream('cam2'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/stats')
def api_stats():
    """Thread-safe snapshot of live_counts from main.py."""
    with pipeline.counts_lock:
        c = dict(pipeline.live_counts)

    return jsonify({
        "cam1_active":    c.get("cam1_active",    0),
        "cam1_total":     c.get("cam1_total",     0),
        "cam2_active":    c.get("cam2_active",    0),
        "cam2_total":     c.get("cam2_total",     0),
        "matches":        c.get("matches",         0),
        "unique":         c.get("unique",          0),
        "cam1_done":      c.get("cam1_done",       False),
        "cam2_done":      c.get("cam2_done",       False),
        "pipeline_done":  pipeline.pipeline_done,
        "pipeline_running": pipeline.pipeline_running,
    })


@app.route('/api/start', methods=['POST'])
def api_start():
    """
    Starts (or restarts) the pipeline in a background thread.
    Rejects the request if a run is already in progress.
    """
    global _pipeline_started

    with _pipeline_lock:
        if pipeline.pipeline_running:
            return jsonify({"ok": False, "error": "Pipeline already running"}), 409

        # Reset all state before starting fresh
        pipeline.reset_pipeline()
        _pipeline_started = True

    def _run():
        pipeline.run_pipeline()

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return jsonify({"ok": True, "message": "Pipeline started"})


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Pipeline does NOT auto-start — user must press the button
    print("\n>>> Dashboard: http://127.0.0.1:5000  <<<\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)