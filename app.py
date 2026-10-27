# ============================================================
# QUANTUM CIRCUIT OPTIMIZER
# Streamlit frontend styled from the Stitch "Quantum Precision" design
# ============================================================

import json
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from src.gate_optimizer import (
    GateOptimizerNN,
    predict_gate_mask,
    ml_optimize_circuit,
)
from src.optimizer import validate_optimization
from src.circuit_utils import check_circuit_equivalence
from src.model_utils import load_model
from src.benchmarks import get_benchmark_names, create_benchmark


# ============================================================
# SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent
MODEL_PATH = str(ROOT / "models/gate_optimizer_cnn.pth")
METRICS_PATH = str(ROOT / "models/evaluation_metrics.json")
ML_THRESHOLD = 0.60
OPERATOR_EQ_TOLERANCE_LABEL = "Exact operator equivalence"
MAX_MODEL_QUBITS = 4
MAX_MODEL_GATES = 50

DEFAULT_USER_CIRCUIT = """x 0
x 0
h 0
z 0
z 0"""

DEFAULT_QASM = """OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];

x q[0];
x q[0];
h q[0];
cx q[0],q[1];
z q[1];
z q[1];
"""




# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Quantum Circuit Optimizer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# STITCH-INSPIRED DESIGN SYSTEM + ANIMATIONS
# ============================================================

st.markdown(
    """
<style>
/* ---------- ROOT ---------- */
:root {
    --q-bg: #090C0B;
    --q-surface: #101614;
    --q-surface-2: #151D1A;
    --q-surface-3: #1B2421;
    --q-surface-4: #202B27;
    --q-border: #26332E;
    --q-border-2: #4C5D55;
    --q-text: #ECE8DF;
    --q-muted: #9FA89C;
    --q-primary: #B7F7A8;
    --q-primary-2: #64C69A;
    --q-secondary: #E9A15B;
    --q-green: #6FD08C;
    --q-red: #E76E56;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
    background: var(--q-bg) !important;
    color: var(--q-text) !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
    backdrop-filter: none !important;
    border-bottom: 0 !important;
    height: 0 !important;
    min-height: 0 !important;
}
[data-testid="stToolbar"] {
    top: 10px !important;
    right: 14px !important;
    opacity: .30;
    transition: opacity .35s ease;
}
[data-testid="stToolbar"]:hover { opacity: 1; }

[data-testid="stMainBlockContainer"] {
    padding-top: 1.05rem;
    max-width: 1800px;
}

[data-testid="stSidebar"] {
    background: var(--q-surface) !important;
    border-right: 1px solid var(--q-border);
}

* {
    scrollbar-color: #34463E #090C0B;
}

h1, h2, h3, h4, p, label, span, div {
    letter-spacing: normal;
}

h1, h2, h3 {
    color: var(--q-text) !important;
}

/* ---------- ANIMATIONS FROM STITCH ---------- */
@keyframes qFadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes qPulseGlow {
    0%, 100% { box-shadow: 0 0 0 rgba(183,247,168,0.0); }
    50% { box-shadow: 0 0 24px rgba(183,247,168,0.16); }
}

@keyframes qPulseDot {
    0%, 100% { opacity: .42; transform: scale(.85); }
    50% { opacity: 1; transform: scale(1.18); }
}

@keyframes qShimmer {
    0% { transform: translateX(-160%); }
    100% { transform: translateX(180%); }
}

@keyframes qScanline {
    0% { top: 3%; }
    50% { top: 94%; }
    100% { top: 3%; }
}

@keyframes qProgress {
    from { width: 0; }
}

@keyframes qBorderFlow {
    0%, 100% { border-color: rgba(183,247,168,0.22); }
    50% { border-color: rgba(183,247,168,0.62); }
}
@keyframes qHeroSheen {
    0% { transform: translateX(-140%) skewX(-18deg); opacity:0; }
    18% { opacity:.26; }
    48% { opacity:.08; }
    100% { transform: translateX(230%) skewX(-18deg); opacity:0; }
}
@keyframes qOrbFloat {
    0%,100% { transform:translate3d(0,0,0) scale(1); opacity:.30; }
    50% { transform:translate3d(18px,-10px,0) scale(1.08); opacity:.46; }
}
@keyframes qSoftBreath {
    0%,100% { opacity:.56; }
    50% { opacity:1; }
}
@keyframes qVerifiedSweep {
    0% { transform:translateX(-130%); opacity:0; }
    20% { opacity:.24; }
    100% { transform:translateX(180%); opacity:0; }
}

/* Slower entrances: deliberate enough to attract attention without feeling laggy. */
.q-fade-1 { animation: qFadeUp .92s cubic-bezier(.16,.84,.28,1) both; animation-delay: .08s; }
.q-fade-2 { animation: qFadeUp .92s cubic-bezier(.16,.84,.28,1) both; animation-delay: .20s; }
.q-fade-3 { animation: qFadeUp .92s cubic-bezier(.16,.84,.28,1) both; animation-delay: .32s; }
.q-fade-4 { animation: qFadeUp .92s cubic-bezier(.16,.84,.28,1) both; animation-delay: .44s; }
.q-fade-5 { animation: qFadeUp .92s cubic-bezier(.16,.84,.28,1) both; animation-delay: .56s; }
.q-fade-6 { animation: qFadeUp .92s cubic-bezier(.16,.84,.28,1) both; animation-delay: .68s; }

/* ---------- CLEAN HERO HEADER ---------- */
.q-topbar {
    position:relative;
    overflow:hidden;
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:18px;
    min-height:96px;
    padding:18px 22px;
    border:1px solid rgba(183,247,168,.16);
    border-radius:14px;
    margin:0 0 24px 0;
    background:
      radial-gradient(circle at 8% 20%, rgba(100,198,154,.10), transparent 31%),
      linear-gradient(180deg, #0A0E14 0%, #070A0F 100%);
    box-shadow:0 16px 48px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.025);
    isolation:isolate;
}
.q-topbar::before {
    content:"";
    position:absolute;
    width:300px; height:300px;
    left:-120px; top:-190px;
    border-radius:50%;
    background:radial-gradient(circle, rgba(100,198,154,.22), transparent 68%);
    filter:blur(14px);
    animation:qOrbFloat 8s ease-in-out infinite;
    z-index:-1;
}
.q-topbar::after {
    content:"";
    position:absolute;
    inset:-30% auto -30% -24%;
    width:24%;
    background:linear-gradient(90deg,transparent,rgba(225,226,255,.18),transparent);
    animation:qHeroSheen 8.5s ease-in-out infinite;
    pointer-events:none;
}
.q-brand-wrap { display:flex; align-items:center; gap:13px; min-width:0; flex-wrap:wrap; }
.q-brand-block { min-width:260px; margin-right:4px; }
.q-kicker {
    color:#96A096;
    font:700 10px/1.2 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    letter-spacing:.15em;
    text-transform:uppercase;
    margin-bottom:7px;
}
.q-brand {
    font-size:1.63rem;
    line-height:1.12;
    font-weight:790;
    color:#F2EEE5;
    white-space:nowrap;
    letter-spacing:-.025em;
    text-shadow:0 0 24px rgba(183,247,168,.08);
}
.q-brand-sub {
    color:#96A096;
    font-size:.78rem;
    margin-top:5px;
}
.q-chip {
    display:inline-flex;
    align-items:center;
    gap:8px;
    padding:7px 11px;
    border:1px solid rgba(183,247,168,.16);
    border-radius:999px;
    background:rgba(12,15,20,.72);
    color:#CFD5CD;
    font:700 10px/1.1 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    letter-spacing:.08em;
    backdrop-filter:blur(10px);
}
.q-dot {
    width:8px; height:8px; border-radius:999px;
    background:#7E82FF;
    box-shadow:0 0 12px rgba(100,198,154,.62);
    animation:qPulseDot 2.8s ease-in-out infinite;
}
.q-verified-chip {
    color:#F2C98C;
    border-color:rgba(233,161,91,.18);
    background:rgba(111,79,38,.12);
}
.q-head-meta {
    display:flex; align-items:center; gap:9px; flex-wrap:wrap; justify-content:flex-end;
}
.q-head-badge {
    padding:8px 10px;
    border:1px solid rgba(70,76,90,.65);
    border-radius:7px;
    color:#AEB6AC;
    background:rgba(8,11,15,.66);
    font:600 10px/1.2 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
.q-head-badge strong { color:#ECE8DF; font-weight:700; }

/* ---------- GLASS CARDS ---------- */
.q-card {
    background:linear-gradient(180deg, rgba(12,15,20,.98), rgba(8,11,15,.98));
    border:1px solid var(--q-border);
    border-radius:10px;
    padding:20px;
    backdrop-filter:blur(12px);
    transition:transform .42s ease, border-color .42s ease, box-shadow .42s ease;
}
.q-card:hover {
    border-color:#3B5148;
    transform:translateY(-2px);
}
.q-card-title {
    color:var(--q-text);
    font-size:1.03rem;
    font-weight:650;
    margin-bottom:4px;
}
.q-eyebrow {
    color:var(--q-muted);
    font:700 10px/1.3 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    letter-spacing:.09em;
    text-transform:uppercase;
}

/* ---------- METRIC CARDS ---------- */
.q-metrics {
    display:grid;
    grid-template-columns:repeat(4,minmax(0,1fr));
    gap:14px;
    margin:4px 0 18px 0;
}
.q-metric {
    min-height:120px;
    display:flex;
    flex-direction:column;
    justify-content:space-between;
}
.q-metric-value {
    color:var(--q-text);
    font-size:2.55rem;
    font-weight:750;
    line-height:1;
    margin-top:12px;
}
.q-metric-value.secondary { color:var(--q-secondary); }
.q-metric-value.primary { color:var(--q-primary); }
.q-progress {
    height:4px;
    overflow:hidden;
    border-radius:999px;
    background:#25302B;
    margin-top:15px;
}
.q-progress > span {
    display:block;
    height:100%;
    width:var(--w);
    background:linear-gradient(90deg,var(--q-primary-2),var(--q-primary));
    border-radius:999px;
    animation:qProgress 1.85s cubic-bezier(.16,.84,.28,1) both;
}

/* ---------- SAFETY BANNER ---------- */
.q-safety {
    position:relative;
    overflow:hidden;
    border:1px solid rgba(183,247,168,.38);
    background:linear-gradient(90deg,rgba(12,15,20,.98),rgba(17,20,23,.98));
    border-radius:10px;
    padding:20px 22px;
    display:flex;
    align-items:center;
    gap:16px;
    animation:qPulseGlow 5.8s ease-in-out infinite, qBorderFlow 5.8s ease-in-out infinite;
    margin-bottom:16px;
}
.q-safety::after {
    content:"";
    position:absolute; inset:0 auto 0 -28%; width:24%;
    background:linear-gradient(90deg,transparent,rgba(183,247,168,.12),transparent);
    animation:qVerifiedSweep 7.5s ease-in-out infinite;
    pointer-events:none;
}
.q-safety.rejected {
    border-color:rgba(231,110,86,.42);
    animation:qBorderFlow 7s ease-in-out infinite;
}
.q-safety.rejected::after { background:linear-gradient(90deg,transparent,rgba(231,110,86,.10),transparent); }
.q-safety-icon {
    display:flex; align-items:center; justify-content:center;
    width:46px; height:46px; border-radius:999px;
    border:1px solid rgba(183,247,168,.35);
    background:rgba(100,198,154,.10);
    font-size:1.35rem;
}
.q-safety.rejected .q-safety-icon {
    border-color:rgba(231,110,86,.35);
    background:rgba(231,110,86,.09);
}
.q-safety-title { color:var(--q-primary); font-size:1.04rem; font-weight:760; }
.q-safety.rejected .q-safety-title { color:#FFAA94; }
.q-safety-meta { color:#9FA89C; font:500 12px/1.5 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; margin-top:4px; }

/* ---------- SCANNING OVERLAY ---------- */
.q-scan {
    position:relative;
    min-height:210px;
    overflow:hidden;
    border:1px solid rgba(183,247,168,.28);
    border-radius:10px;
    background:rgba(12,15,20,.92);
    display:flex;
    align-items:center;
    justify-content:center;
    flex-direction:column;
    gap:10px;
    animation:qPulseGlow 4.2s ease-in-out infinite;
}
.q-scan::after {
    content:"";
    position:absolute;
    left:0; right:0;
    height:2px;
    background:linear-gradient(90deg,transparent,var(--q-primary),transparent);
    box-shadow:0 0 16px rgba(183,247,168,.8);
    animation:qScanline 2.75s cubic-bezier(.45,0,.55,1) infinite;
}
.q-scan-icon { font-size:2.1rem; color:var(--q-primary); animation:qPulseDot 2.2s ease-in-out infinite; }
.q-scan-text { color:var(--q-primary); font:700 14px/1 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; letter-spacing:.13em; }
.q-scan-sub { color:#929C94; font-size:.8rem; }


/* ---------- LIVE OPTIMIZATION PIPELINE ---------- */
.q-live-pipeline {
    position:relative;
    overflow:hidden;
    min-height:235px;
    border:1px solid rgba(183,247,168,.30);
    border-radius:12px;
    padding:22px;
    background:radial-gradient(circle at 85% 15%, rgba(100,198,154,.10), transparent 34%),linear-gradient(180deg,rgba(12,15,20,.98),rgba(7,10,15,.98));
    box-shadow:0 18px 48px rgba(0,0,0,.24);
}
.q-live-head { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:18px; }
.q-live-title { color:#F2F2F7; font-size:1.05rem; font-weight:760; }
.q-live-status { color:var(--q-primary); font:700 10px/1 ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:.10em; }
.q-live-track { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }
.q-live-step { min-height:112px; padding:14px 12px; border:1px solid rgba(70,76,90,.52); border-radius:8px; background:rgba(10,13,18,.72); transition:.3s ease; }
.q-live-step.active { border-color:rgba(183,247,168,.62); background:rgba(100,198,154,.09); box-shadow:0 0 24px rgba(100,198,154,.09); }
.q-live-step.done { border-color:rgba(16,185,129,.34); background:rgba(16,185,129,.035); }
.q-live-num { color:#7E8290; font:700 10px/1 ui-monospace,SFMono-Regular,Menlo,monospace; margin-bottom:10px; }
.q-live-step.active .q-live-num { color:var(--q-primary); }
.q-live-step.done .q-live-num { color:#74D7B1; }
.q-live-label { color:#D8D8E1; font-size:.82rem; font-weight:680; margin-bottom:7px; }
.q-live-desc { color:#8F909D; font-size:.72rem; line-height:1.45; }
.q-live-progress { height:4px; margin-top:16px; background:#20252E; border-radius:999px; overflow:hidden; }
.q-live-progress span { display:block; height:100%; width:var(--w); background:linear-gradient(90deg,#7074EE,#B7F7A8); border-radius:999px; transition:width .45s ease; }
.q-live-pulse { display:inline-block; width:7px; height:7px; border-radius:99px; background:#AEB0FF; box-shadow:0 0 12px rgba(183,247,168,.8); margin-right:7px; animation:qPulseDot 1.2s ease-in-out infinite; }

/* ---------- FINAL OUTCOME ---------- */
.q-outcome { display:grid; grid-template-columns:minmax(0,1.3fr) minmax(190px,.7fr); gap:16px; align-items:stretch; margin:0 0 16px 0; }
.q-outcome-main, .q-outcome-score { border:1px solid rgba(183,247,168,.22); border-radius:10px; background:linear-gradient(180deg,rgba(13,16,22,.98),rgba(8,11,15,.98)); padding:18px 20px; }
.q-outcome.rejected .q-outcome-main, .q-outcome.rejected .q-outcome-score { border-color:rgba(231,110,86,.28); }
.q-outcome-kicker { color:#96A096; font:700 10px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:.10em; text-transform:uppercase; }
.q-outcome-title { margin-top:8px; color:#F2F2F7; font-size:1.2rem; font-weight:780; line-height:1.25; }
.q-outcome-copy { margin-top:7px; color:#9C9CA8; font-size:.82rem; line-height:1.5; }
.q-outcome-score { display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; }
.q-outcome-number { color:var(--q-primary); font-size:2.6rem; line-height:1; font-weight:800; }
.q-outcome.rejected .q-outcome-number { color:#FFAA94; font-size:1.55rem; }
.q-outcome-label { margin-top:7px; color:#9293A0; font:700 10px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:.08em; text-transform:uppercase; }
@media (max-width: 900px) { .q-live-track { grid-template-columns:repeat(2,minmax(0,1fr)); } .q-outcome { grid-template-columns:1fr; } }

/* ---------- CONFIDENCE ROWS ---------- */
.q-confidence-row {
    display:grid;
    grid-template-columns:56px 88px minmax(130px,1fr) 92px;
    gap:10px;
    align-items:center;
    padding:9px 2px;
    border-bottom:1px solid rgba(30,41,59,.62);
    font:500 12px/1.2 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
.q-confidence-row:last-child { border-bottom:0; }
.q-bar { height:6px; border-radius:999px; overflow:hidden; background:#25302B; }
.q-bar > span {
    display:block; height:100%; width:var(--w);
    border-radius:999px;
    background:linear-gradient(90deg,#4DAA7D,#B7F7A8);
    animation:qProgress 1.65s cubic-bezier(.16,.84,.28,1) both;
}
.q-decision {
    justify-self:end;
    padding:5px 8px;
    border-radius:5px;
    font-weight:700;
    letter-spacing:.03em;
}
.q-remove { color:#FFB29F; background:rgba(231,110,86,.10); border:1px solid rgba(231,110,86,.18); }
.q-keep { color:var(--q-primary); background:rgba(183,247,168,.08); border:1px solid rgba(183,247,168,.18); }

/* ---------- PERFORMANCE ---------- */
.q-perf-row {
    display:flex;
    justify-content:space-between;
    gap:16px;
    padding:10px 0;
    border-bottom:1px solid rgba(30,41,59,.70);
}
.q-perf-row:last-child { border-bottom:0; }
.q-perf-label { color:#B7B9B1; font-size:.83rem; }
.q-perf-value { color:var(--q-text); font:650 13px/1 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
.q-perf-value.accent { color:var(--q-primary); }

/* ---------- PIPELINE ---------- */
.q-step {
    display:grid;
    grid-template-columns:28px 1fr;
    gap:12px;
    align-items:start;
    margin:8px 0;
    animation:qFadeUp .82s cubic-bezier(.16,.84,.28,1) both;
}
.q-step:nth-child(1){animation-delay:.16s}
.q-step:nth-child(2){animation-delay:.34s}
.q-step:nth-child(3){animation-delay:.52s}
.q-step:nth-child(4){animation-delay:.70s}
.q-step-num {
    width:26px; height:26px; display:flex; align-items:center; justify-content:center;
    border-radius:5px; border:1px solid var(--q-border-2); background:#26312D;
    color:#D7DAD3; font:700 11px/1 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
.q-step:last-child .q-step-num { color:var(--q-primary); border-color:rgba(183,247,168,.42); background:rgba(100,198,154,.10); }
.q-step-text { color:#D3D5CE; font-size:.84rem; line-height:1.5; padding-top:2px; }

/* ---------- STREAMLIT CONTROL RESTYLE ---------- */
[data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input,
[data-baseweb="select"] > div {
    background:#0A0E0C !important;
    color:var(--q-text) !important;
    border-color:#35463F !important;
    border-radius:5px !important;
}
[data-testid="stTextArea"] textarea:focus,
[data-testid="stNumberInput"] input:focus {
    border-color:var(--q-primary-2) !important;
    box-shadow:0 0 0 1px var(--q-primary-2), 0 0 14px rgba(100,198,154,.12) !important;
}

[data-testid="stRadio"] label { color:#C7C4D7 !important; }

div.stButton > button[kind="primary"] {
    position:relative;
    overflow:hidden;
    min-height:46px;
    border-radius:5px;
    border:1px solid rgba(183,247,168,.26);
    background:#B7F7A8;
    color:#102219;
    font-weight:760;
    letter-spacing:.02em;
    transition:all .24s ease;
    box-shadow:0 0 15px rgba(183,247,168,.12);
}
div.stButton > button[kind="primary"]:hover {
    transform:translateY(-1px);
    background:#D8FFC9;
    box-shadow:0 0 20px rgba(183,247,168,.22);
}
div.stButton > button[kind="primary"]:active { transform:scale(.985); }
div.stButton > button[kind="primary"]::after {
    content:"";
    position:absolute;
    inset:-20% auto -20% -40%;
    width:32%;
    background:linear-gradient(90deg,transparent,rgba(255,255,255,.45),transparent);
    transform:skewX(-18deg);
    animation:qShimmer 4.6s ease-in-out infinite;
}

[data-testid="stExpander"] {
    background:var(--q-surface) !important;
    border:1px solid var(--q-border) !important;
    border-radius:8px !important;
}

[data-testid="stCodeBlock"] {
    border:1px solid var(--q-border) !important;
    border-radius:7px !important;
}

[data-testid="stDataFrame"] {
    border:1px solid var(--q-border);
    border-radius:8px;
    overflow:hidden;
}

/* ---------- THEME SWITCH ---------- */
[data-testid="stToggle"] label {
    color: var(--q-muted) !important;
    font: 700 10px/1.2 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
    letter-spacing: .10em !important;
    text-transform: uppercase;
}
[data-testid="stToggle"] [role="switch"] {
    border: 1px solid var(--q-border-2) !important;
}

/* Hide default decoration/menu while keeping Streamlit requirement */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}

@media (max-width: 1050px) {
    .q-metrics { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .q-head-meta { display:none; }
}
@media (max-width: 650px) {
    .q-metrics { grid-template-columns:1fr; }
    .q-chip.q-verified-chip { display:none; }
    .q-brand { font-size:1.15rem; white-space:normal; }
    .q-topbar { padding:16px; min-height:auto; }
    .q-confidence-row { grid-template-columns:44px 60px 1fr 76px; font-size:10px; }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration:.01ms !important;
        animation-iteration-count:1 !important;
        transition-duration:.01ms !important;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

@st.cache_resource
def load_cnn_model():
    model = GateOptimizerNN()
    model = load_model(model, MODEL_PATH)
    model.eval()
    return model


@st.cache_data
def load_saved_metrics():
    # Legacy artifact has no training/data provenance. Never invent results.
    try:
        with Path(METRICS_PATH).open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        required = {"gate_accuracy", "f1_score", "exact_mask_accuracy",
                    "operator_equivalence_rate", "safe_optimization_rate",
                    "overall_gate_reduction", "test_circuits"}
        return loaded if required.issubset(loaded) else None
    except (OSError, ValueError, TypeError):
        return None


def build_circuit_from_qasm(qasm_text):
    if not qasm_text.strip():
        raise ValueError("OpenQASM input cannot be empty.")
    try:
        return QuantumCircuit.from_qasm_str(qasm_text)
    except Exception as error:
        raise ValueError(f"Invalid OpenQASM: {error}") from error


def build_circuit_from_text(circuit_text, num_qubits):
    """Parse the lightweight manual syntax used in the Streamlit UI."""
    circuit = QuantumCircuit(num_qubits)

    for line_number, raw_line in enumerate(circuit_text.strip().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.lower().split()
        gate = parts[0]

        if gate in {"x", "y", "z", "h", "s", "t"}:
            if len(parts) != 2:
                raise ValueError(
                    f"Line {line_number}: {gate.upper()} requires exactly one qubit."
                )
            try:
                qubit = int(parts[1])
            except ValueError as error:
                raise ValueError(
                    f"Line {line_number}: qubit index must be an integer."
                ) from error

            if qubit < 0 or qubit >= num_qubits:
                raise ValueError(
                    f"Line {line_number}: qubit {qubit} does not exist."
                )

            getattr(circuit, gate)(qubit)

        elif gate == "cx":
            if len(parts) != 3:
                raise ValueError(f"Line {line_number}: CX requires control and target qubits.")
            try:
                control = int(parts[1])
                target = int(parts[2])
            except ValueError as error:
                raise ValueError(
                    f"Line {line_number}: CX qubit indices must be integers."
                ) from error

            if not (0 <= control < num_qubits and 0 <= target < num_qubits):
                raise ValueError(f"Line {line_number}: invalid qubit index.")
            if control == target:
                raise ValueError(
                    f"Line {line_number}: control and target cannot be the same."
                )
            circuit.cx(control, target)

        else:
            raise ValueError(f"Line {line_number}: unsupported gate '{gate}'.")

    return circuit

def scroll_to_results():
    """
    Automatically scroll the browser to the optimization
    results after the ML pipeline finishes.
    """

    components.html(
        """
        <script>
        setTimeout(function() {

            const parentDocument =
                window.parent.document;

            const target =
                parentDocument.getElementById(
                    "optimization-results"
                );

            if (target) {
                target.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });
            }

        }, 350);
        </script>
        """,
        height=0,
    )

def circuit_to_text(circuit):
    return str(circuit.draw(output="text"))

def simulate_probabilities(circuit):
    """
    Simulate a quantum circuit using Qiskit's Statevector
    and return basis-state probabilities.

    This is intended for circuits without measurements.
    """
    try:
        # Create quantum state after applying the circuit
        state = Statevector.from_instruction(circuit)

        # Convert statevector into probabilities
        probabilities = state.probabilities_dict()

        # Convert values to normal floats
        probabilities = {
            str(state_name): float(probability)
            for state_name, probability in probabilities.items()
        }

        return probabilities

    except Exception as error:
        raise ValueError(
            f"Quantum simulation failed: {error}"
        ) from error


def simulation_html(probabilities, title="Simulation Result"):

    if not probabilities:
        return '<div class="q-card"><div class="q-card-title">No simulation data</div></div>'

    items = sorted(probabilities.items())

    rows = ""

    for state, probability in items:

        percentage = probability * 100

        if percentage < 0.0001:
            percentage = 0.0

        rows += f"""
<div style="display:grid;grid-template-columns:70px 1fr 72px;gap:12px;align-items:center;margin:11px 0;">
<div style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;color:var(--q-text);">
|{state}⟩
</div>

<div style="height:8px;background:#232833;border-radius:999px;overflow:hidden;">
<div style="width:{percentage:.2f}%;height:100%;border-radius:999px;background:linear-gradient(90deg,var(--q-primary-2),var(--q-primary));">
</div>
</div>

<div style="text-align:right;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--q-primary);font-weight:700;font-size:12px;">
{percentage:.2f}%
</div>
</div>
"""

    return f"""
<div class="q-card q-fade-4">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">

<div>
<div class="q-card-title">{title}</div>
<div class="q-eyebrow">STATEVECTOR PROBABILITY DISTRIBUTION</div>
</div>

<div class="q-chip">QUANTUM SIMULATION</div>

</div>

{rows}

</div>
"""

def circuit_preflight(circuit):
    """Protect the fixed-size CNN encoder from unsupported oversized inputs."""
    issues = []
    if circuit.num_qubits > MAX_MODEL_QUBITS:
        issues.append(
            f"This saved CNN supports at most {MAX_MODEL_QUBITS} qubits; "
            f"this circuit has {circuit.num_qubits}."
        )
    if len(circuit.data) > MAX_MODEL_GATES:
        issues.append(
            f"This saved CNN supports at most {MAX_MODEL_GATES} gate positions; "
            f"this circuit has {len(circuit.data)} gates."
        )
    result = check_circuit_equivalence(circuit, circuit)
    if result["status"] == "inconclusive":
        issues.append(result["reason"])
    return issues


def metric_html(original, proposed=None, final=None, reduction=None):
    proposed_text = "—" if proposed is None else str(proposed)
    final_text = "—" if final is None else str(final)
    reduction_text = "—" if reduction is None else f"{reduction:.1f}%"
    width = 0 if reduction is None else max(0, min(100, reduction))

    return f"""
<div class="q-metrics">
  <div class="q-card q-metric q-fade-1">
    <div class="q-eyebrow">Original Gates</div>
    <div class="q-metric-value">{original}</div>
  </div>
  <div class="q-card q-metric q-fade-2">
    <div class="q-eyebrow">ML Proposed Gates</div>
    <div class="q-metric-value secondary">{proposed_text}</div>
  </div>
  <div class="q-card q-metric q-fade-3">
    <div class="q-eyebrow">Final Safe Gates</div>
    <div class="q-metric-value primary">{final_text}</div>
  </div>
  <div class="q-card q-metric q-fade-4">
    <div class="q-eyebrow">Safe Gate Reduction</div>
    <div class="q-metric-value primary">{reduction_text}</div>
    <div class="q-progress"><span style="--w:{width:.1f}%"></span></div>
  </div>
</div>
"""


def performance_html(metrics):
    if metrics is None:
        return "<div class='q-card'>No saved evaluation results are available.</div>"
    rows = [
        ("Gate Accuracy", f"{metrics['gate_accuracy'] * 100:.2f}%", False),
        ("Precision", f"{metrics.get('precision', 0) * 100:.2f}%", False),
        ("Recall", f"{metrics.get('recall', 0) * 100:.2f}%", False),
        ("F1 Score", f"{metrics['f1_score'] * 100:.2f}%", False),
        ("Exact Mask Accuracy", f"{metrics['exact_mask_accuracy'] * 100:.2f}%", False),
        ("Operator Equivalence", f"{metrics['operator_equivalence_rate'] * 100:.2f}%", True),
        ("Safe Optimization Rate", f"{metrics['safe_optimization_rate'] * 100:.2f}%", True),
        ("Overall Gate Reduction", f"{metrics['overall_gate_reduction']:.2f}%", True),
    ]

    body = "".join(
        f'<div class="q-perf-row"><span class="q-perf-label">{label}</span>'
        f'<span class="q-perf-value {"accent" if accent else ""}">{value}</span></div>'
        for label, value, accent in rows
    )

    return f"""
<div class="q-card q-fade-5">
  <div class="q-card-title">Model Performance</div>
  <div class="q-eyebrow">{int(metrics['test_circuits']):,} circuits in the legacy report — dataset provenance unverified</div>
  <div style="margin-top:10px">{body}</div>
  <div style="margin-top:13px;padding:10px;border:1px solid var(--q-border);border-radius:6px;background:#111815;color:#9DA69E;font-size:.75rem;line-height:1.5">
    Historical CNN results, not GNN results. Labels have since been corrected; rerun evaluation before making performance claims.
  </div>
</div>
"""


def safety_html(accepted):
    if accepted:
        return """
<div class="q-safety q-fade-2">
  <div class="q-safety-icon">✓</div>
  <div>
    <div class="q-safety-title">OPERATOR EQUIVALENCE VERIFIED — OPTIMIZATION ACCEPTED</div>
    <div class="q-safety-meta">U<sub>proposed</sub> ≡ e<sup>iφ</sup> U<sub>original</sub> &nbsp; • &nbsp; full operator verification passed</div>
  </div>
</div>
"""
    return """
<div class="q-safety rejected q-fade-2">
  <div class="q-safety-icon">✕</div>
  <div>
    <div class="q-safety-title">OPERATOR EQUIVALENCE FAILED — ML PROPOSAL REJECTED</div>
    <div class="q-safety-meta">Unsafe proposal discarded &nbsp; • &nbsp; original circuit restored automatically</div>
  </div>
</div>
"""


def live_pipeline_html(active_step, status_text):
    steps = [
        ("Encode Circuit", "Convert gates and qubit positions into the model input tensor."),
        ("CNN Inference", "Predict a removal score for every real gate position."),
        ("Build Proposal", "Apply the threshold and construct the ML candidate circuit."),
        ("Verify Operator", "Check the candidate against the original up to global phase."),
    ]
    cards = []
    for i, (title, desc) in enumerate(steps, start=1):
        if i < active_step:
            state, symbol = "done", "✓"
        elif i == active_step:
            state, symbol = "active", f"0{i}"
        else:
            state, symbol = "", f"0{i}"
        cards.append(
            f'<div class="q-live-step {state}"><div class="q-live-num">{symbol}</div>'
            f'<div class="q-live-label">{title}</div><div class="q-live-desc">{desc}</div></div>'
        )
    progress = max(8, min(100, active_step * 25))
    return f"""
<div class="q-live-pipeline q-fade-2">
  <div class="q-live-head">
    <div><div class="q-eyebrow">LIVE OPTIMIZATION PIPELINE</div><div class="q-live-title">ML proposes. Mathematics verifies.</div></div>
    <div class="q-live-status"><span class="q-live-pulse"></span>{status_text}</div>
  </div>
  <div class="q-live-track">{''.join(cards)}</div>
  <div class="q-live-progress"><span style="--w:{progress}%"></span></div>
</div>
"""


def outcome_html(result):
    if result["accepted"]:
        removed = result["original_gate_count"] - result["final_gate_count"]
        return f"""
<div class="q-outcome q-fade-2">
  <div class="q-outcome-main">
    <div class="q-outcome-kicker">Optimization Result</div>
    <div class="q-outcome-title">Verified circuit reduction completed successfully.</div>
    <div class="q-outcome-copy">The CNN proposed removing {removed} gate(s), and the resulting circuit passed full operator-equivalence verification. The optimized circuit is safe to return.</div>
  </div>
  <div class="q-outcome-score"><div class="q-outcome-number">{result['reduction']:.1f}%</div><div class="q-outcome-label">Safe Gate Reduction</div></div>
</div>
"""
    return """
<div class="q-outcome rejected q-fade-2">
  <div class="q-outcome-main">
    <div class="q-outcome-kicker">Safety Intervention</div>
    <div class="q-outcome-title">ML proposal rejected. Original circuit restored.</div>
    <div class="q-outcome-copy">The neural network suggested a smaller circuit, but the complete quantum operation changed. The verification layer blocked the unsafe optimization automatically.</div>
  </div>
  <div class="q-outcome-score"><div class="q-outcome-number">BLOCKED</div><div class="q-outcome-label">Unsafe Proposal</div></div>
</div>
"""


def confidence_html(circuit, probabilities, threshold):
    rows = []
    for index, probability in enumerate(probabilities):
        instruction = circuit.data[index]
        gate_name = instruction.operation.name.upper()
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        qtext = ",".join(str(q) for q in qubits)
        p = float(probability)
        decision = "REMOVE" if p >= threshold else "KEEP"
        cls = "q-remove" if decision == "REMOVE" else "q-keep"
        rows.append(
            f"""
<div class="q-confidence-row" style="animation:qFadeUp .72s cubic-bezier(.16,.84,.28,1) both;animation-delay:{0.08 + index * 0.09:.2f}s">
  <span style="color:#7E8881">{index:02d}</span>
  <span>{gate_name} (q{qtext})</span>
  <div class="q-bar" title="{p:.4f}"><span style="--w:{p * 100:.2f}%"></span></div>
  <span class="q-decision {cls}">{decision}</span>
</div>
"""
        )

    return f"""
<div class="q-card q-fade-5">
  <div style="display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:10px">
    <div>
      <div class="q-card-title">Removal Confidence</div>
      <div class="q-eyebrow">Gate-level CNN inference</div>
    </div>
    <div class="q-chip">THRESHOLD {threshold:.2f}</div>
  </div>
  {''.join(rows)}
</div>
"""


def pipeline_html():
    steps = [
        ("Encoding", "Circuit gates are converted into the fixed numerical tensor representation used by the CNN."),
        ("CNN Inference", "The 1D CNN predicts a removal probability for every gate position."),
        ("Proposal", f"Gates with P(remove) ≥ {ML_THRESHOLD:.2f} form the ML candidate circuit."),
        ("Verification", "The original and candidate circuits are converted to complete operators and checked up to global phase."),
    ]
    body = "".join(
        f'<div class="q-step"><div class="q-step-num">{i}</div><div class="q-step-text"><b>{title}:</b> {text}</div></div>'
        for i, (title, text) in enumerate(steps, start=1)
    )
    return f"<div class='q-card'><div class='q-card-title'>Pipeline Overview</div>{body}</div>"


# ============================================================
# SESSION STATE
# ============================================================

if "optimization_result" not in st.session_state:
    st.session_state.optimization_result = None

if "scroll_to_results" not in st.session_state:
    st.session_state.scroll_to_results = False

# ============================================================
# LOAD MODEL + METRICS
# ============================================================

metrics = load_saved_metrics()

try:
    model = load_cnn_model()
    model_loaded = True
    model_error = None
except Exception as error:
    model_loaded = False
    model_error = str(error)


# ============================================================
# APPEARANCE
# ============================================================

if "light_mode" not in st.session_state:
    st.session_state.light_mode = False

_theme_spacer, _theme_control = st.columns([0.84, 0.16])
with _theme_control:
    light_mode = st.toggle("Light mode", key="light_mode")

if light_mode:
    st.markdown(
        """
<style>
/* Warm-paper light theme: scientific/editorial rather than generic SaaS. */
:root {
    --q-bg: #F1EEE5;
    --q-surface: #FAF8F1;
    --q-surface-2: #F5F1E8;
    --q-surface-3: #E9E5DA;
    --q-surface-4: #DDD8CC;
    --q-border: #D1CABB;
    --q-border-2: #A99F8E;
    --q-text: #19211D;
    --q-muted: #667069;
    --q-primary: #286B55;
    --q-primary-2: #3C8A6D;
    --q-secondary: #B66532;
    --q-green: #2E7D5B;
    --q-red: #B64A38;
}
html, body, [data-testid="stAppViewContainer"], .stApp {
    background:#F1EEE5 !important;
    color:var(--q-text) !important;
}
[data-testid="stSidebar"] { background:#FAF8F1 !important; border-right:1px solid var(--q-border); }
* { scrollbar-color:#AAA18F #F1EEE5; }

.q-topbar {
    background:
      radial-gradient(circle at 8% 20%, rgba(40,107,85,.10), transparent 31%),
      linear-gradient(180deg,#FBF9F3 0%,#F5F1E8 100%);
    border-color:rgba(40,107,85,.20);
    box-shadow:0 18px 50px rgba(63,52,36,.09), inset 0 1px 0 rgba(255,255,255,.8);
}
.q-topbar::before { background:radial-gradient(circle,rgba(40,107,85,.13),transparent 68%); }
.q-topbar::after { background:linear-gradient(90deg,transparent,rgba(182,101,50,.10),transparent); }
.q-brand { color:#19211D !important; text-shadow:none; }
.q-brand-sub,.q-kicker { color:#68716A; }
.q-chip { background:rgba(250,248,241,.88); color:#36423C; border-color:#D4CCBD; }
.q-verified-chip { color:#8A4D27; background:rgba(182,101,50,.08); border-color:rgba(182,101,50,.20); }
.q-head-badge { background:#F8F5ED; color:#6F756F; border-color:#D9D2C6; }
.q-head-badge strong { color:#19211D; }
.q-dot { background:#286B55; box-shadow:0 0 12px rgba(40,107,85,.35); }

.q-card,
.q-live-pipeline,
.q-outcome,
.q-scan {
    background:linear-gradient(180deg,#FBF9F3,#F6F2E9) !important;
    border-color:#D5CEBF !important;
    box-shadow:0 12px 34px rgba(69,58,41,.05);
}
.q-card:hover { border-color:#B9AF9E !important; }
.q-card-title,.q-live-title,.q-outcome-title,.q-step-text { color:#202823 !important; }
.q-eyebrow,.q-outcome-kicker { color:#6A746D !important; }
.q-metric-value { color:#19211D !important; }
.q-metric-value.primary { color:#286B55 !important; }
.q-metric-value.secondary { color:#B66532 !important; }
.q-progress,.q-bar { background:#DDD7CB !important; }
.q-progress > span,.q-bar > span { background:linear-gradient(90deg,#3C8A6D,#286B55) !important; }

.q-safety {
    background:linear-gradient(90deg,#F8F7EF,#F2F5EC) !important;
    border-color:rgba(40,107,85,.35) !important;
}
.q-safety-title { color:#286B55 !important; }
.q-safety-meta { color:#68726B !important; }
.q-safety-icon { background:rgba(40,107,85,.08); border-color:rgba(40,107,85,.28); }
.q-safety.rejected { background:linear-gradient(90deg,#FBF4EF,#F8ECE6) !important; border-color:rgba(182,74,56,.34) !important; }
.q-safety.rejected .q-safety-title { color:#A54132 !important; }
.q-safety.rejected .q-safety-icon { background:rgba(182,74,56,.07); border-color:rgba(182,74,56,.25); }

.q-live-card { background:#F4F0E7 !important; border-color:#D5CEBF !important; }
.q-live-card.active { background:#EDF5EF !important; border-color:rgba(40,107,85,.38) !important; }
.q-live-card.done { background:#F1F3E8 !important; }
.q-live-status { color:#286B55 !important; }
.q-live-progress { background:#DCD6CA !important; }
.q-live-progress > span { background:linear-gradient(90deg,#B66532,#286B55) !important; }

.q-outcome-score { border-left-color:#D6CFC1 !important; }
.q-outcome-number { color:#286B55 !important; }
.q-outcome.rejected .q-outcome-number { color:#B64A38 !important; }
.q-outcome-copy { color:#626C65 !important; }

.q-confidence-row { border-bottom-color:#DDD6CA !important; color:#28312C !important; }
.q-remove { color:#A94432 !important; background:rgba(182,74,56,.08) !important; border-color:rgba(182,74,56,.17) !important; }
.q-keep { color:#286B55 !important; background:rgba(40,107,85,.07) !important; border-color:rgba(40,107,85,.18) !important; }
.q-step-num { background:#E8E3D8 !important; border-color:#C9C1B3 !important; color:#414A44 !important; }
.q-step:last-child .q-step-num { color:#286B55 !important; background:rgba(40,107,85,.08) !important; border-color:rgba(40,107,85,.28) !important; }

[data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input,
[data-baseweb="select"] > div {
    background:#FFFDF8 !important;
    color:#19211D !important;
    border-color:#CFC7B8 !important;
}
[data-testid="stTextArea"] textarea:focus,
[data-testid="stNumberInput"] input:focus {
    border-color:#3C8A6D !important;
    box-shadow:0 0 0 1px #3C8A6D, 0 0 14px rgba(40,107,85,.08) !important;
}
[data-testid="stRadio"] label,[data-testid="stToggle"] label { color:#4F5953 !important; }
[data-testid="stExpander"],[data-testid="stCodeBlock"],[data-testid="stDataFrame"] {
    background:#FAF8F1 !important;
    border-color:#D4CCBD !important;
}
[data-testid="stCodeBlock"] pre { background:#F5F1E8 !important; color:#202823 !important; }

div.stButton > button[kind="primary"] {
    background:#286B55 !important;
    color:#FFF9ED !important;
    border-color:#286B55 !important;
    box-shadow:0 8px 20px rgba(40,107,85,.12) !important;
}
div.stButton > button[kind="primary"]:hover {
    background:#347A62 !important;
    box-shadow:0 10px 24px rgba(40,107,85,.18) !important;
}
</style>
""",
        unsafe_allow_html=True,
    )

# ============================================================
# TOP BAR
# ============================================================

model_status = "ACTIVE" if model_loaded else "OFFLINE"
status_dot = "<span class='q-dot'></span>" if model_loaded else "<span style='width:8px;height:8px;border-radius:999px;background:#E76E56;display:inline-block'></span>"

st.markdown(
    f"""
<div class="q-topbar q-fade-1">
  <div class="q-brand-wrap">
    <div class="q-brand-block">
      <div class="q-kicker">ML-DRIVEN QUANTUM OPTIMIZATION</div>
      <div class="q-brand">Quantum Circuit Optimizer</div>
      <div class="q-brand-sub">CNN gate analysis with exact operator-equivalence verification</div>
    </div>
    <div class="q-chip">{status_dot} CNN {model_status}</div>
    <div class="q-chip q-verified-chip">◆ SAFETY LAYER ACTIVE</div>
  </div>
  <div class="q-head-meta">
    <div class="q-head-badge">THRESHOLD <strong>{ML_THRESHOLD:.2f}</strong></div>
    <div class="q-head-badge">MODEL LIMIT <strong>{MAX_MODEL_QUBITS}Q / {MAX_MODEL_GATES}G</strong></div>
    <div class="q-head-badge">VERIFY <strong>OPERATOR ≡φ</strong></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if not model_loaded:
    st.error(f"CNN model could not be loaded: {model_error}")


# ============================================================
# INPUT WORKSPACE
# ============================================================

left, right = st.columns([0.31, 0.69], gap="large")

with left:
    st.markdown("<div class='q-eyebrow'>CIRCUIT INPUT</div>", unsafe_allow_html=True)
    st.markdown("### Build or import a circuit")
    st.page_link("pages/1_Graph_Lab.py", label="Open Graph Optimizer Lab", icon="🔬")

    circuit_source = st.radio(
        "Input method",
        ["Manual", "Benchmark", "OpenQASM"],
        horizontal=True,
        label_visibility="collapsed",
    )

    user_circuit = None
    circuit_valid = False

    if circuit_source == "Manual":
        num_qubits = st.number_input(
            "Number of qubits",
            min_value=1,
            max_value=MAX_MODEL_QUBITS,
            value=1,
            step=1,
        )
        circuit_text = st.text_area(
            "Gate instructions",
            value=DEFAULT_USER_CIRCUIT,
            height=220,
            help="One gate per line. Example: x 0, h 1, cx 0 1",
        )

        with st.expander("Supported manual syntax"):
            st.code("x 0\ny 0\nz 0\nh 0\ns 0\nt 0\ncx 0 1", language="text")

        try:
            user_circuit = build_circuit_from_text(circuit_text, int(num_qubits))
            circuit_valid = True
        except ValueError as error:
            st.error(str(error))

    elif circuit_source == "Benchmark":
        benchmark_name = st.selectbox("Benchmark", get_benchmark_names())
        benchmark_qubits = st.slider(
            "Number of qubits",
            min_value=2,
            max_value=MAX_MODEL_QUBITS,
            value=3,
        )
        try:
            user_circuit = create_benchmark(
                benchmark_name,
                num_qubits=benchmark_qubits,
            )
            circuit_valid = True
        except Exception as error:
            st.error(f"Could not create benchmark: {error}")

    else:
        qasm_text = st.text_area(
            "OpenQASM 2.0",
            value=DEFAULT_QASM,
            height=300,
        )
        with st.expander("OpenQASM example"):
            st.code(DEFAULT_QASM, language="text")
        try:
            user_circuit = build_circuit_from_qasm(qasm_text)
            circuit_valid = True
        except ValueError as error:
            st.error(str(error))

    if circuit_valid:
        issues = circuit_preflight(user_circuit)
        if issues:
            for issue in issues:
                st.error(issue)
            circuit_valid = False

        st.markdown("<div class='q-eyebrow' style='margin-top:10px'>PARSED CIRCUIT</div>", unsafe_allow_html=True)
        st.code(circuit_to_text(user_circuit), language="text")
        st.caption(
            f"{user_circuit.num_qubits} qubit(s) • {len(user_circuit.data)} gate(s) • model limit {MAX_MODEL_QUBITS} qubits / {MAX_MODEL_GATES} gates"
        )

    optimize_clicked = st.button(
        "⚡ OPTIMIZE CIRCUIT",
        type="primary",
        use_container_width=True,
        disabled=(not circuit_valid or not model_loaded),
    )

    st.markdown(performance_html(metrics), unsafe_allow_html=True)


# ============================================================
# RUN OPTIMIZATION
# ============================================================

if optimize_clicked and circuit_valid and model_loaded:
    scan_slot = right.empty()

    try:
        scan_slot.markdown(live_pipeline_html(1, "ENCODING CIRCUIT"), unsafe_allow_html=True)
        time.sleep(0.55)

        scan_slot.markdown(live_pipeline_html(2, "RUNNING CNN INFERENCE"), unsafe_allow_html=True)
        candidate_circuit, predicted_mask, probabilities = ml_optimize_circuit(
            user_circuit, model, threshold=ML_THRESHOLD,
        )
        time.sleep(0.55)

        scan_slot.markdown(live_pipeline_html(4, "VERIFYING EQUIVALENCE"), unsafe_allow_html=True)
        final_circuit, equivalence_metric, accepted = validate_optimization(
            user_circuit,
            candidate_circuit,
            fidelity_threshold=0.99,
        )
        time.sleep(0.80)

        original_gate_count = len(user_circuit.data)
        proposed_gate_count = len(candidate_circuit.data)
        final_gate_count = len(final_circuit.data)
        safe_removed = original_gate_count - final_gate_count
        reduction = (
            safe_removed / original_gate_count * 100
            if original_gate_count
            else 0.0
        )

        st.session_state.optimization_result = {
            "source": circuit_source,
            "original": user_circuit,
            "candidate": candidate_circuit,
            "final": final_circuit,
            "predicted_mask": predicted_mask,
            "probabilities": probabilities,
            "accepted": accepted,
            "equivalence_metric": equivalence_metric,
            "original_gate_count": original_gate_count,
            "proposed_gate_count": proposed_gate_count,
            "final_gate_count": final_gate_count,
            "reduction": reduction,
        }

    except Exception as error:
        st.session_state.optimization_result = None
        scan_slot.empty()
        right.error(f"Optimization failed: {error}")
    else:
        completion = "VERIFIED — OPTIMIZATION ACCEPTED" if accepted else "SAFETY CHECK BLOCKED PROPOSAL"
        scan_slot.markdown(live_pipeline_html(4, completion), unsafe_allow_html=True)
        time.sleep(0.35)
        scan_slot.empty()

    st.session_state.scroll_to_results = True


# ============================================================
# RESULTS WORKSPACE
# ============================================================

with right:

    st.markdown(
        """
        <div
            id="optimization-results"
            style="scroll-margin-top: 25px;">
        </div>
        """,
        unsafe_allow_html=True,
    )

    result = st.session_state.optimization_result

    # Do not show stale results when the current circuit shape differs.
    if result is not None and user_circuit is not None:
        current_signature = (user_circuit.num_qubits, len(user_circuit.data), circuit_to_text(user_circuit))
        result_signature = (
            result["original"].num_qubits,
            len(result["original"].data),
            circuit_to_text(result["original"]),
        )
        if current_signature != result_signature:
            result = None

    original_count_for_top = len(user_circuit.data) if user_circuit is not None else 0

    if result is None:
        st.markdown(
            metric_html(original_count_for_top),
            unsafe_allow_html=True,
        )

        st.markdown(
            """
<div class="q-card q-fade-5" style="min-height:250px;display:flex;align-items:center;justify-content:center;text-align:center">
  <div>
    <div style="font-size:2rem;color:var(--q-primary);margin-bottom:10px">◈</div>
    <div class="q-card-title">Optimization workspace ready</div>
    <div style="color:#929C94;font-size:.86rem;max-width:520px;line-height:1.55;margin-top:8px">
      Build, benchmark, or paste an OpenQASM circuit, then run the CNN optimizer. The candidate will be accepted only after full operator-equivalence verification.
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(pipeline_html(), unsafe_allow_html=True)

    else:
        st.markdown(
            metric_html(
                result["original_gate_count"],
                result["proposed_gate_count"],
                result["final_gate_count"],
                result["reduction"],
            ),
            unsafe_allow_html=True,
        )

        st.markdown(safety_html(result["accepted"]), unsafe_allow_html=True)
        st.markdown(outcome_html(result), unsafe_allow_html=True)

        st.markdown("<div class='q-card-title q-fade-3'>Circuit Visualization</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3, gap="medium")

        with c1:
            st.markdown("<div class='q-eyebrow'>ORIGINAL CIRCUIT</div>", unsafe_allow_html=True)
            st.code(circuit_to_text(result["original"]), language="text")
            st.caption(f"{result['original_gate_count']} gates")

        with c2:
            st.markdown("<div class='q-eyebrow' style='color:#E9A15B'>ML PROPOSAL</div>", unsafe_allow_html=True)
            st.code(circuit_to_text(result["candidate"]), language="text")
            st.caption(f"{result['proposed_gate_count']} gates")

        with c3:
            title = "FINAL VERIFIED" if result["accepted"] else "ORIGINAL RESTORED"
            st.markdown(
                f"<div class='q-eyebrow' style='color:#B7F7A8'>{title}</div>",
                unsafe_allow_html=True,
            )
            st.code(
                circuit_to_text(result["final"]),
                language="text"
            )
            st.caption(
                f"{result['final_gate_count']} gates"
            )


        # ============================================================
        # QUANTUM SIMULATION
        # ============================================================

        st.markdown(
            """
            <div class='q-card-title q-fade-4'
                 style='margin-top:24px'>
                Quantum Simulation
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "Statevector simulation showing the probability distribution "
            "of the original and final verified quantum circuits."
        )

        try:
            original_probabilities = simulate_probabilities(
                result["original"]
            )

            final_probabilities = simulate_probabilities(
                result["final"]
            )

            sim_left, sim_right = st.columns(
                2,
                gap="large"
            )

            with sim_left:
                st.markdown(
                    simulation_html(
                        original_probabilities,
                        "Original Circuit"
                    ),
                    unsafe_allow_html=True,
                )

            with sim_right:
                st.markdown(
                    simulation_html(
                        final_probabilities,
                        "Final Verified Circuit"
                    ),
                    unsafe_allow_html=True,
                )

        except Exception as error:
            st.warning(
                f"Simulation unavailable: {error}"
            )


        # ============================================================
        # CNN CONFIDENCE + PIPELINE
        # ============================================================

        bottom_left, bottom_right = st.columns(
            [1.05, 0.95],
            gap="large"
        )

        with bottom_left:
            st.markdown(
                confidence_html(
                    result["original"],
                    result["probabilities"],
                    ML_THRESHOLD,
                ),
                unsafe_allow_html=True,
            )

            with st.expander("Raw gate-removal mask"):
                st.code(
                    str(result["predicted_mask"]),
                    language="text",
                )
                st.caption(
                    "1 = REMOVE • 0 = KEEP"
                )

        with bottom_right:
            st.markdown(
                pipeline_html(),
                unsafe_allow_html=True,
            )

            if result["accepted"]:
                st.success(
                    "The ML proposal passed exact operator-equivalence "
                    "verification and is returned as the final circuit."
                )
            else:
                st.warning(
                    "The candidate was not operator-equivalent to the "
                    "original circuit, so the system rejected it and "
                    "restored the original."
                )

       # AUTO-SCROLL TO NEW RESULTS
# ============================================================

if st.session_state.scroll_to_results:

    scroll_to_results()

    st.session_state.scroll_to_results = False

# ============================================================
# TECHNICAL DETAILS
# ============================================================

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

with st.expander("Technical details & current architecture"):
    st.markdown(
        f"""
**Current production path**

1. User circuit → Qiskit `QuantumCircuit`
2. Fixed gate/qubit feature encoding
3. 1D CNN gate-level removal prediction
4. Threshold `{ML_THRESHOLD:.2f}` generates the candidate mask
5. Candidate circuit is constructed
6. Exact operator-equivalence verification is performed up to global phase
7. Equivalent candidates are accepted; non-equivalent candidates are rejected and the original circuit is restored

**Prototype constraints**

- Maximum supported qubits in this saved CNN: **{MAX_MODEL_QUBITS}**
- Maximum encoded gate positions: **{MAX_MODEL_GATES}**
- Exact operator construction scales exponentially, so this verification strategy is intentionally limited to small circuits

**Experimental modules also present in the repository**

- Simple redundancy-count neural network
- Autoencoder for compressed circuit representations
- Rule-based teacher used to generate supervised gate-removal labels
"""
    )