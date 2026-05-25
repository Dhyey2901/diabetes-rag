# src/web_ui.py
from __future__ import annotations

import json
import os
import time
from collections import deque, defaultdict, Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, render_template_string, request, session, redirect, url_for, jsonify, Response
from markupsafe import escape
from src.qa import answer as qa_answer


# =========================
# Metrics store
# =========================
class MetricsStore:
    def __init__(self, path: str = "/tmp/rag_metrics.jsonl", maxlen: int = 20000):
        self.path = path
        self.buffer: deque = deque(maxlen=maxlen)
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self.buffer.append(json.loads(line))
        except Exception:
            self.buffer.clear()

    def _append_file(self, rec: Dict[str, Any]) -> None:
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def add(self, rec: Dict[str, Any]) -> None:
        self.buffer.append(rec)
        self._append_file(rec)

    def _parse_range(self, range_key: str) -> Tuple[datetime, datetime]:
        now = datetime.now(timezone.utc)
        key = (range_key or "today").lower()
        if key in ("last_hour", "1h"):
            start = now - timedelta(hours=1)
        elif key in ("today", "24h", "day"):
            start = now - timedelta(hours=24)
        elif key in ("7d", "last_7_days", "week"):
            start = now - timedelta(days=7)
        elif key in ("30d", "last_30_days", "month"):
            start = now - timedelta(days=30)
        else:
            start = now - timedelta(hours=24)
        return start, now

    def window(self, range_key: str) -> List[Dict[str, Any]]:
        start, end = self._parse_range(range_key)
        out = []
        for r in self.buffer:
            try:
                ts = datetime.fromisoformat(r["ts"])
            except Exception:
                continue
            if start <= ts <= end:
                out.append(r)
        return out

    def perf_timeseries(self, range_key: str) -> Dict[str, Any]:
        rows = self.window(range_key)
        bins: dict = defaultdict(list)
        for r in rows:
            ts = datetime.fromisoformat(r["ts"])
            bucket = ts.replace(minute=(ts.minute // 5) * 5, second=0, microsecond=0)
            bins[bucket.isoformat()] += [r]
        labels, accuracy, abstention_rate, latency_avg = [], [], [], []
        for k in sorted(bins.keys()):
            batch = bins[k]
            correct_vals = [b.get("correct") for b in batch if b.get("correct") is not None]
            acc = sum(correct_vals) / len(correct_vals) if correct_vals else None
            abst = sum(1 for b in batch if b.get("abstained")) / len(batch) if batch else 0
            lat = sum(b.get("latency_ms", 0) for b in batch) / len(batch) if batch else 0
            labels.append(k)
            accuracy.append(None if acc is None else round(acc, 3))
            abstention_rate.append(round(abst, 3))
            latency_avg.append(round(lat, 1))
        lats = sorted([r.get("latency_ms", 0) for r in rows if r.get("latency_ms") is not None])
        def pct(p):
            if not lats: return None
            idx = max(0, min(len(lats) - 1, int(len(lats) * p / 100)))
            return round(lats[idx], 1)
        return {"labels": labels, "accuracy": accuracy, "abstention_rate": abstention_rate,
                "latency_avg": latency_avg, "latency_p50": pct(50), "latency_p90": pct(90),
                "latency_p95": pct(95), "count": len(rows)}

    def accuracy_by_category(self, range_key: str) -> Dict[str, Any]:
        rows = self.window(range_key)
        by_cat: dict = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in rows:
            cat = r.get("category") or "Uncategorized"
            by_cat[cat]["total"] += 1
            if r.get("correct") is True:
                by_cat[cat]["correct"] += 1
        labels, data = [], []
        for cat, v in sorted(by_cat.items(), key=lambda x: x[0].lower()):
            labels.append(cat)
            data.append(round(v["correct"] / v["total"], 3) if v["total"] else 0)
        return {"labels": labels, "data": data}

    def business_metrics(self, range_key: str) -> Dict[str, Any]:
        rows = self.window(range_key)
        token_cost = sum(r.get("token_cost", 0) or 0 for r in rows)
        time_saved_s = sum((r.get("abstained") is False) * 30 for r in rows)
        return {"cost_saved_usd": round(max(0.0, 100.0 - token_cost), 2),
                "time_saved_minutes": round(time_saved_s / 60, 1),
                "answered": sum(1 for r in rows if not r.get("abstained")),
                "total": len(rows)}

    def usage_metrics(self, range_key: str) -> Dict[str, Any]:
        rows = self.window(range_key)
        cats = Counter([r.get("category") or "Uncategorized" for r in rows])
        heat = [[0] * 24 for _ in range(7)]
        for r in rows:
            ts = datetime.fromisoformat(r["ts"])
            heat[ts.weekday()][ts.hour] += 1
        bins2: dict = defaultdict(int)
        for r in rows:
            ts = datetime.fromisoformat(r["ts"])
            bucket = ts.replace(minute=0, second=0, microsecond=0).isoformat()
            bins2[bucket] += 1
        tl_labels = sorted(bins2.keys())
        return {"category": {"labels": list(cats.keys()), "counts": [cats[c] for c in cats]},
                "heatmap": {"matrix": heat},
                "timeline": {"labels": tl_labels, "counts": [bins2[k] for k in tl_labels]}}

    def health_metrics(self, range_key: str) -> Dict[str, Any]:
        rows = self.window(range_key)
        errors = sum(1 for r in rows if r.get("error") is True)
        rate = (errors / len(rows)) if rows else 0.0
        last_ts = max((r.get("ts") for r in self.buffer), default=None)
        return {"error_rate": round(rate, 3), "queue_len": 0,
                "index_freshness": last_ts, "cpu": None, "mem": None}


metrics_store = MetricsStore()


# =========================
# QA helper
# =========================
def _classify_category(ql: str) -> str:
    if any(k in ql for k in ["glucose", "cgm", "reading"]): return "Glucose"
    if any(k in ql for k in ["insulin", "dose", "bolus", "basal"]): return "Insulin"
    if any(k in ql for k in ["diet", "carb", "food", "meal"]): return "Nutrition"
    if any(k in ql for k in ["bp", "blood pressure", "hypertension"]): return "Cardio"
    return "General"


def _run_qa(question: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    error_flag = False
    try:
        res = qa_answer(question)
        answer = getattr(res, "answer", "")
        sources = getattr(res, "citations", []) or []
        confidence = getattr(res, "confidence", None)
        category = getattr(res, "category", None) or _classify_category(question.lower())
        correct = getattr(res, "correct", None)
        token_cost = getattr(res, "token_cost", None)
    except Exception as e:
        answer = f"⚠️ {e}"
        sources, confidence, category, correct, token_cost = [], None, _classify_category(question.lower()), None, None
        error_flag = True
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    abstained = answer.strip().lower().startswith("i don't know")
    return {
        "q": question, "a": answer,
        "sources": [s for s in sources if isinstance(s, dict)],
        "confidence": round(confidence, 2) if confidence is not None else None,
        "latency_ms": latency_ms, "abstained": abstained,
        "category": category, "correct": correct,
        "token_cost": token_cost, "error": error_flag,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def _record_metrics(turn: Dict[str, Any]) -> None:
    metrics_store.add({
        "ts": turn["ts"], "latency_ms": turn["latency_ms"],
        "confidence": turn["confidence"], "abstained": turn["abstained"],
        "category": turn["category"],
        "citation_count": len(turn["sources"]),
        "sources": list({s.get("source_id") for s in turn["sources"] if s.get("source_id")}),
        "correct": turn["correct"], "token_cost": turn["token_cost"], "error": turn["error"],
    })


# =========================
# Chat render helpers
# =========================
_SUGGESTIONS = [
    "What is the A1C target for most adults with type 2 diabetes?",
    "What blood pressure target is recommended for adults with diabetes?",
    "What annual screening is recommended for diabetic kidney disease?",
    "What physical activity is recommended for people with diabetes?",
    "Are SGLT2 inhibitors recommended for patients with type 2 diabetes and CKD?",
]


def _welcome_html() -> str:
    pills = "".join(f'<button class="welcome-pill">{escape(p)}</button>' for p in _SUGGESTIONS)
    return f"""<div class="welcome" id="welcome-screen">
  <div class="welcome-icon">⚕️</div>
  <h2>ADA 2025 Clinical Q&amp;A</h2>
  <p>Ask about the ADA Standards of Care in Diabetes 2025. The system cites its sources and abstains when the answer isn't in the guidelines.</p>
  <div class="welcome-pills">{pills}</div>
</div>"""


def _conf_badge_html(conf: Optional[float]) -> str:
    if conf is None:
        return ""
    cls = "conf-high" if conf >= 0.6 else ("conf-mid" if conf >= 0.35 else "conf-low")
    return f'<span class="conf-badge {cls}">conf {conf:.2f}</span>'


def _render_history(history: List[Dict[str, Any]]) -> str:
    if not history:
        return _welcome_html()
    html = ""
    for item in history:
        q = item.get("q", "")
        a = item.get("a", "")
        conf = item.get("confidence")
        abstained = item.get("abstained", False)
        latency_ms = item.get("latency_ms")
        ts = item.get("ts", "")
        sources = item.get("sources") or []

        # user bubble
        html += f"""<div class="msg-row user">
  <div class="avatar avatar-user">U</div>
  <div class="msg-body">
    <div class="bubble bubble-user">{escape(q)}</div>
    <div class="msg-meta"><span class="ts" data-ts="{escape(ts)}"></span></div>
  </div>
</div>"""

        # bot bubble
        bubble_cls = "bubble-abstained" if abstained else "bubble-bot"
        conf_badge = _conf_badge_html(conf)
        lat_html = f'<span class="latency-tag">{round(latency_ms / 1000, 1)}s</span>' if latency_ms else ""
        prefix = "⚠️ " if abstained else ""

        seen: set = set()
        pills_html = ""
        for s in sources:
            sec = s.get("section", "")
            if sec and sec not in seen and not abstained:
                seen.add(sec)
                pills_html += f'<span class="source-pill">{escape(sec)}</span>'
        sources_block = ""
        if pills_html:
            n = len(seen)
            label = f"{n} source{'s' if n > 1 else ''}"
            sources_block = f"""<div class="sources-row">
  <button class="sources-toggle" onclick="this.nextElementSibling.hidden=!this.nextElementSibling.hidden">📚 {label}</button>
  <div class="source-pills" hidden>{pills_html}</div>
</div>"""

        html += f"""<div class="msg-row">
  <div class="avatar avatar-bot">⚕</div>
  <div class="msg-body">
    <div class="bubble {bubble_cls}">{escape(prefix + a)}</div>
    <div class="msg-meta">
      {conf_badge}
      {lat_html}
      <button class="copy-btn" data-text="{escape(a)}" onclick="copyText(this,this.dataset.text)">Copy</button>
      <span class="ts" data-ts="{escape(ts)}"></span>
    </div>
    {sources_block}
  </div>
</div>"""
    return html


# =========================
# Base template
# =========================
BASE_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Diabetes RAG — ADA 2025</title>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"/>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"/>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
:root{
  --bg:#0d0d14;--bg2:#13131e;--bg3:#1a1a28;--border:#252535;
  --text:#e2e8f0;--muted:#64748b;
  --blue:#38bdf8;--indigo:#6366f1;--purple:#a78bfa;--teal:#2dd4bf;
  --green:#10b981;--amber:#f59e0b;--red:#ef4444;
  --user-grad:linear-gradient(135deg,#3b82f6,#06b6d4);
  --bot-grad:linear-gradient(135deg,#7c3aed,#6366f1);
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--text);display:flex;height:100vh;overflow:hidden;font-size:14px}

/* sidebar */
.sidebar{width:230px;min-width:230px;background:var(--bg2);border-right:1px solid var(--border);display:flex;flex-direction:column;padding:16px 10px;gap:2px;transition:all .25s cubic-bezier(.4,0,.2,1);overflow:hidden}
.sidebar.collapsed{width:0;min-width:0;padding:0}
.sidebar-logo{display:flex;align-items:center;gap:10px;padding:6px 10px 18px;color:var(--blue);font-weight:700;font-size:15px;white-space:nowrap}
.nav-link{display:flex;align-items:center;gap:10px;padding:9px 12px;border-radius:10px;color:var(--muted);text-decoration:none;font-size:13px;font-weight:500;transition:all .15s;white-space:nowrap}
.nav-link:hover{color:var(--text);background:var(--bg3)}
.nav-link.active{color:var(--blue);background:rgba(56,189,248,.1)}
.nav-danger{color:#f87171!important}
.nav-danger:hover{background:rgba(239,68,68,.08)!important;color:#f87171!important}
.sidebar-spacer{flex:1}

/* main */
.main{flex:1;display:flex;flex-direction:column;min-width:0;background:var(--bg)}

/* header */
.header{display:flex;align-items:center;gap:12px;padding:11px 18px;border-bottom:1px solid var(--border);background:var(--bg2)}
.toggle-btn{background:none;border:none;color:var(--muted);cursor:pointer;padding:4px;border-radius:6px;line-height:1;transition:color .15s;flex-shrink:0}
.toggle-btn:hover{color:var(--blue)}
.header-title{font-size:14px;font-weight:600;color:var(--text)}
.header-sub{font-size:11px;color:var(--muted);margin-left:auto}

/* chat scroll area */
.chat-wrap{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:14px;scroll-behavior:smooth}
.chat-wrap::-webkit-scrollbar{width:4px}
.chat-wrap::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px}

/* welcome */
.welcome{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;gap:20px;padding:40px 20px}
.welcome-icon{font-size:48px;line-height:1}
.welcome h2{font-size:20px;font-weight:700}
.welcome p{color:var(--muted);max-width:400px;line-height:1.65}
.welcome-pills{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;max-width:540px}
.welcome-pill{background:var(--bg3);border:1px solid var(--border);border-radius:20px;padding:7px 14px;font-size:12.5px;color:var(--text);cursor:pointer;transition:all .15s;text-align:left}
.welcome-pill:hover{border-color:var(--blue);color:var(--blue);background:rgba(56,189,248,.06)}

/* message rows */
.msg-row{display:flex;gap:10px;animation:fadeUp .22s ease}
.msg-row.user{flex-direction:row-reverse}
@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.avatar{width:30px;height:30px;border-radius:50%;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:600}
.avatar-bot{background:var(--bot-grad)}
.avatar-user{background:var(--user-grad)}
.msg-body{display:flex;flex-direction:column;gap:5px;max-width:74%}
.msg-row.user .msg-body{align-items:flex-end}

/* bubbles */
.bubble{padding:11px 15px;border-radius:16px;line-height:1.65;word-break:break-word}
.bubble-user{background:var(--user-grad);color:#fff;border-bottom-right-radius:4px}
.bubble-bot{background:var(--bg3);border:1px solid var(--border);color:var(--text);border-bottom-left-radius:4px}
.bubble-abstained{background:var(--bg2);border:1px solid var(--border);color:var(--muted);border-bottom-left-radius:4px}

/* meta row */
.msg-meta{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.msg-row.user .msg-meta{justify-content:flex-end}
.ts{font-size:11px;color:var(--muted)}
.latency-tag{font-size:11px;color:var(--muted)}

/* confidence badge */
.conf-badge{font-size:11px;font-weight:600;padding:2px 8px;border-radius:20px}
.conf-high{background:rgba(16,185,129,.15);color:#6ee7b7;border:1px solid rgba(16,185,129,.3)}
.conf-mid{background:rgba(245,158,11,.15);color:#fcd34d;border:1px solid rgba(245,158,11,.3)}
.conf-low{background:rgba(239,68,68,.15);color:#fca5a5;border:1px solid rgba(239,68,68,.3)}

/* copy button */
.copy-btn{background:none;border:none;color:var(--muted);cursor:pointer;font-size:11px;padding:2px 7px;border-radius:6px;transition:all .15s}
.copy-btn:hover{color:var(--blue);background:rgba(56,189,248,.1)}
.copy-btn.copied{color:var(--green)}

/* sources */
.sources-row{margin-top:4px}
.sources-toggle{background:none;border:1px solid var(--border);color:var(--muted);border-radius:20px;font-size:11px;padding:3px 10px;cursor:pointer;transition:all .15s}
.sources-toggle:hover{border-color:var(--blue);color:var(--blue)}
.source-pills{display:flex;flex-wrap:wrap;gap:5px;margin-top:6px}
.source-pill{background:rgba(99,102,241,.1);border:1px solid rgba(99,102,241,.25);color:#a5b4fc;border-radius:20px;padding:2px 10px;font-size:11px}

/* typing */
.typing-dots{display:flex;gap:4px;align-items:center;padding:6px 2px}
.typing-dots span{width:6px;height:6px;border-radius:50%;background:var(--muted);animation:bounce .9s infinite ease-in-out}
.typing-dots span:nth-child(2){animation-delay:.15s}
.typing-dots span:nth-child(3){animation-delay:.3s}
@keyframes bounce{0%,80%,100%{transform:translateY(0);opacity:.4}40%{transform:translateY(-5px);opacity:1}}

/* input area */
.input-area{padding:12px 18px 14px;border-top:1px solid var(--border);background:var(--bg2)}
.input-inner{display:flex;gap:8px;align-items:flex-end;background:var(--bg3);border:1px solid var(--border);border-radius:14px;padding:8px 8px 8px 14px;transition:border-color .2s}
.input-inner:focus-within{border-color:var(--blue)}
.input-ta{flex:1;background:none;border:none;color:var(--text);font-family:inherit;font-size:14px;line-height:1.5;resize:none;max-height:130px;overflow-y:auto}
.input-ta:focus{outline:none}
.input-ta::placeholder{color:var(--muted)}
.send-btn{width:34px;height:34px;border-radius:10px;border:none;background:var(--user-grad);color:#fff;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:opacity .15s;flex-shrink:0}
.send-btn:hover{opacity:.85}
.send-btn:disabled{opacity:.35;cursor:not-allowed}
.input-hint{font-size:11px;color:var(--muted);text-align:right;margin-top:5px}

/* dashboard */
.dash-wrap{flex:1;overflow-y:auto;padding:18px}
.dash-wrap::-webkit-scrollbar{width:4px}
.dash-wrap::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px}
.grid{display:grid;gap:12px;grid-template-columns:repeat(12,1fr)}
.card{background:var(--bg2);border:1px solid var(--border);border-radius:14px;padding:14px}
.card-title{font-size:12px;font-weight:600;color:var(--blue);margin-bottom:10px;text-transform:uppercase;letter-spacing:.04em}
.kpi-row{display:flex;gap:10px}
.kpi-box{flex:1;background:var(--bg3);border:1px solid var(--border);border-radius:12px;padding:14px 16px}
.kpi-label{font-size:10px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px}
.kpi-value{font-size:26px;font-weight:700;color:var(--text)}
.muted{color:var(--muted);font-size:12px}
canvas{background:transparent;border-radius:8px}
</style>
</head>
<body>

<div class="sidebar" id="sidebar">
  <div class="sidebar-logo">
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" fill="rgba(56,189,248,.12)" stroke="#38bdf8" stroke-width="1.5"/><path d="M12 7v5l3 2" stroke="#38bdf8" stroke-width="1.5" stroke-linecap="round"/></svg>
    Diabetes RAG
  </div>
  <a href="/" class="nav-link" id="nav-chat">
    <svg width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.8" viewBox="0 0 24 24"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
    Chat
  </a>
  <a href="/dashboard" class="nav-link" id="nav-dash">
    <svg width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.8" viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/></svg>
    Dashboard
  </a>
  <a href="/history" class="nav-link" id="nav-hist">
    <svg width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.8" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
    History
  </a>
  <div class="sidebar-spacer"></div>
  <a href="/clear_history" class="nav-link nav-danger">
    <svg width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.8" viewBox="0 0 24 24"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4h6v2"/></svg>
    Clear History
  </a>
</div>

<div class="main">
  <div class="header">
    <button class="toggle-btn" onclick="toggleSidebar()" aria-label="Toggle sidebar">
      <svg width="17" height="17" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
    </button>
    <span class="header-title">ADA 2025 Clinical Q&amp;A</span>
    <span class="header-sub">Grounded · cited · abstains when unsure</span>
  </div>

  <div class="{% if show_input %}chat-wrap{% else %}dash-wrap{% endif %}" id="chatbox">
    {{ content|safe }}
  </div>

  {% if show_input %}
  <div class="input-area">
    <div class="input-inner">
      <textarea class="input-ta" id="inputTa" rows="1" placeholder="Ask about ADA 2025 guidelines…"></textarea>
      <button class="send-btn" id="sendBtn" aria-label="Send">
        <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" viewBox="0 0 24 24"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
      </button>
    </div>
    <div class="input-hint">Ctrl + Enter to send</div>
  </div>
  {% endif %}
</div>

<script>
// sidebar
function toggleSidebar(){ document.getElementById('sidebar').classList.toggle('collapsed'); }

// active nav
(function(){
  const p = location.pathname;
  if(p==='/'||p==='') document.getElementById('nav-chat').classList.add('active');
  else if(p.startsWith('/dashboard')) document.getElementById('nav-dash').classList.add('active');
  else if(p.startsWith('/history')) document.getElementById('nav-hist').classList.add('active');
})();

// relative timestamps
function relTime(iso){
  if(!iso) return '';
  const diff = Math.round((Date.now() - new Date(iso)) / 1000);
  if(diff < 5)   return 'just now';
  if(diff < 60)  return diff + 's ago';
  if(diff < 3600) return Math.round(diff/60) + 'm ago';
  return new Date(iso).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});
}
function refreshTs(){ document.querySelectorAll('[data-ts]').forEach(el=>{ el.textContent=relTime(el.dataset.ts); }); }
setInterval(refreshTs, 30000);

// copy to clipboard
function copyText(btn, text){
  navigator.clipboard.writeText(text).then(()=>{
    btn.textContent='✓ Copied'; btn.classList.add('copied');
    setTimeout(()=>{ btn.textContent='Copy'; btn.classList.remove('copied'); }, 1800);
  });
}

// chatbox ref
const chatbox = document.getElementById('chatbox');

// build a message turn from API data
function buildTurn(q, answer, sources, conf, abstained, latency_ms, ts){
  // user row
  const userRow = document.createElement('div');
  userRow.className = 'msg-row user';
  userRow.innerHTML = `<div class="avatar avatar-user">U</div><div class="msg-body"><div class="bubble bubble-user"></div><div class="msg-meta"><span class="ts" data-ts="${ts||''}"></span></div></div>`;
  userRow.querySelector('.bubble-user').textContent = q;

  // bot row
  const bubbleCls = abstained ? 'bubble-abstained' : 'bubble-bot';
  let confBadge = '';
  if(conf != null){
    const cls = conf>=0.6?'conf-high':conf>=0.35?'conf-mid':'conf-low';
    confBadge = `<span class="conf-badge ${cls}">conf ${conf.toFixed(2)}</span>`;
  }
  const latTag = latency_ms ? `<span class="latency-tag">${(latency_ms/1000).toFixed(1)}s</span>` : '';

  const botRow = document.createElement('div');
  botRow.className = 'msg-row';
  botRow.innerHTML = `
    <div class="avatar avatar-bot">⚕</div>
    <div class="msg-body">
      <div class="bubble ${bubbleCls}"></div>
      <div class="msg-meta">${confBadge}${latTag}<button class="copy-btn" onclick="copyText(this,this.dataset.text)">Copy</button><span class="ts" data-ts="${ts||''}"></span></div>
    </div>`;
  botRow.querySelector(`.${bubbleCls}`).textContent = (abstained ? '⚠️ ' : '') + answer;
  botRow.querySelector('.copy-btn').dataset.text = answer;

  // citation pills
  if(sources && sources.length && !abstained){
    const seen = new Set();
    const pills = sources.map(s=>s.section||'').filter(s=>{ if(!s||seen.has(s)) return false; seen.add(s); return true; });
    if(pills.length){
      const wrap = document.createElement('div');
      wrap.className = 'sources-row';
      const n = pills.length;
      const toggle = document.createElement('button');
      toggle.className = 'sources-toggle';
      toggle.textContent = `📚 ${n} source${n>1?'s':''}`;
      const pillsDiv = document.createElement('div');
      pillsDiv.className = 'source-pills';
      pillsDiv.hidden = true;
      toggle.onclick = () => { pillsDiv.hidden = !pillsDiv.hidden; };
      pills.forEach(p => { const el=document.createElement('span'); el.className='source-pill'; el.textContent=p; pillsDiv.appendChild(el); });
      wrap.appendChild(toggle);
      wrap.appendChild(pillsDiv);
      botRow.querySelector('.msg-body').appendChild(wrap);
    }
  }

  refreshTs();
  return [userRow, botRow];
}

// typing indicator
function addTyping(){
  const row = document.createElement('div');
  row.className = 'msg-row'; row.id = 'typing-row';
  row.innerHTML = `<div class="avatar avatar-bot">⚕</div><div class="bubble bubble-bot"><div class="typing-dots"><span></span><span></span><span></span></div></div>`;
  chatbox.appendChild(row);
  chatbox.scrollTop = chatbox.scrollHeight;
  return row;
}

// send question via AJAX
async function sendQuestion(q){
  if(!q || !q.trim()) return;
  const ta = document.getElementById('inputTa');
  const btn = document.getElementById('sendBtn');
  if(ta){ ta.value=''; ta.style.height='auto'; }
  if(btn) btn.disabled = true;

  const welcome = document.getElementById('welcome-screen');
  if(welcome) welcome.remove();

  const ts = new Date().toISOString();
  const typing = addTyping();

  try {
    const res = await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
    const data = await res.json();
    typing.remove();
    const [ur, br] = buildTurn(q, data.answer, data.sources, data.confidence, data.abstained, data.latency_ms, ts);
    chatbox.appendChild(ur);
    chatbox.appendChild(br);
  } catch(e) {
    typing.remove();
    const [ur, br] = buildTurn(q, '⚠️ Server error — please try again.', [], null, false, null, ts);
    chatbox.appendChild(ur);
    chatbox.appendChild(br);
  } finally {
    if(btn) btn.disabled = false;
    if(ta) ta.focus();
    chatbox.scrollTop = chatbox.scrollHeight;
  }
}

// wire input
const ta = document.getElementById('inputTa');
const sendBtn = document.getElementById('sendBtn');
if(ta){
  ta.addEventListener('input', ()=>{ ta.style.height='auto'; ta.style.height=Math.min(ta.scrollHeight,130)+'px'; });
  ta.addEventListener('keydown', e=>{
    if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)){ e.preventDefault(); sendQuestion(ta.value.trim()); }
  });
}
if(sendBtn){ sendBtn.addEventListener('click', ()=>{ if(ta) sendQuestion(ta.value.trim()); }); }

// wire welcome pills
document.querySelectorAll('.welcome-pill').forEach(p=>{ p.addEventListener('click',()=>sendQuestion(p.textContent.trim())); });

// scroll to bottom
if(chatbox) chatbox.scrollTop = chatbox.scrollHeight;
refreshTs();
</script>
</body>
</html>"""


# =========================
# Flask app
# =========================
def create_app():
    app = Flask(__name__)
    app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(24)

    # ── Chat (GET + POST fallback for no-JS) ──
    @app.route("/", methods=["GET", "POST"])
    def rag():
        if "history" not in session:
            session["history"] = []
        if request.method == "POST":
            q = request.form.get("question", "").strip()
            if q:
                turn = _run_qa(q)
                session["history"].append({k: turn[k] for k in ["q", "a", "sources", "confidence", "latency_ms", "abstained", "ts"]})
                session.modified = True
                _record_metrics(turn)
            return redirect(url_for("rag"))
        content = _render_history(session.get("history", []))
        return render_template_string(BASE_TEMPLATE, content=content, show_input=True)

    # ── AJAX ask endpoint ──
    @app.post("/ask")
    def ask():
        body = request.get_json(force=True, silent=True) or {}
        q = (body.get("question") or "").strip()
        if not q:
            return jsonify({"error": "empty question"}), 400
        if "history" not in session:
            session["history"] = []
        turn = _run_qa(q)
        session["history"].append({k: turn[k] for k in ["q", "a", "sources", "confidence", "latency_ms", "abstained", "ts"]})
        session.modified = True
        _record_metrics(turn)
        return jsonify({
            "answer": turn["a"],
            "sources": turn["sources"],
            "confidence": turn["confidence"],
            "abstained": turn["abstained"],
            "latency_ms": turn["latency_ms"],
        })

    # ── History ──
    @app.route("/history")
    def history():
        hist = session.get("history", [])
        content = f"<h4 style='color:var(--blue);font-size:14px;font-weight:600;padding:4px 0 12px'>🕒 Chat History ({len(hist)} turns)</h4>"
        content += _render_history(hist)
        return render_template_string(BASE_TEMPLATE, content=content, show_input=False)

    # ── Clear ──
    @app.route("/clear_history")
    def clear_history():
        session.pop("history", None)
        return redirect(url_for("rag"))

    # ── Dashboard ──
    @app.route("/dashboard")
    def dashboard():
        content = """
<div class="grid">

  <!-- Controls -->
  <div class="card" style="grid-column:1/span 12">
    <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;justify-content:space-between">
      <div style="display:flex;align-items:center;gap:8px">
        <span class="muted">Range</span>
        <select id="rangeSel" class="form-select form-select-sm" style="width:150px;background:var(--bg3);color:var(--text);border-color:var(--border)">
          <option value="last_hour">Last Hour</option>
          <option value="today" selected>Today</option>
          <option value="7d">Last 7 Days</option>
          <option value="30d">Last 30 Days</option>
        </select>
      </div>
      <div style="display:flex;align-items:center;gap:10px">
        <label style="display:flex;align-items:center;gap:6px;font-size:12px;color:var(--muted)">
          <input type="checkbox" id="auto"/> Auto-refresh
          <select id="autoInt" class="form-select form-select-sm" style="width:90px;background:var(--bg3);color:var(--text);border-color:var(--border)">
            <option value="5">5s</option>
            <option value="30" selected>30s</option>
            <option value="60">1m</option>
          </select>
        </label>
        <button class="btn btn-sm" style="background:var(--bg3);color:var(--text);border:1px solid var(--border)" id="exportCsv">Export CSV</button>
      </div>
    </div>
  </div>

  <!-- KPIs -->
  <div class="card" style="grid-column:1/span 12">
    <div class="kpi-row">
      <div class="kpi-box"><div class="kpi-label">Total Requests</div><div class="kpi-value" id="k_count">0</div></div>
      <div class="kpi-box"><div class="kpi-label">Avg Latency</div><div class="kpi-value" id="k_lat">–</div></div>
      <div class="kpi-box"><div class="kpi-label">p90 Latency</div><div class="kpi-value" id="k_p90">–</div></div>
      <div class="kpi-box"><div class="kpi-label">Abstention Rate</div><div class="kpi-value" id="k_abs">–</div></div>
    </div>
  </div>

  <!-- Performance -->
  <div class="card" style="grid-column:1/span 8">
    <div class="card-title">Performance Over Time</div>
    <canvas id="perfChart" height="140"></canvas>
  </div>
  <div class="card" style="grid-column:9/span 4">
    <div class="card-title">Latency Percentiles</div>
    <canvas id="latGauge" height="140"></canvas>
    <div class="muted" style="margin-top:6px">p50 · p90 · p95</div>
  </div>

  <!-- Usage -->
  <div class="card" style="grid-column:1/span 4">
    <div class="card-title">Question Categories</div>
    <canvas id="catDonut" height="180"></canvas>
  </div>
  <div class="card" style="grid-column:5/span 8">
    <div class="card-title">Volume Timeline</div>
    <canvas id="volumeChart" height="180"></canvas>
  </div>

  <!-- Quality -->
  <div class="card" style="grid-column:1/span 6">
    <div class="card-title">Confidence Distribution</div>
    <canvas id="confHist" height="160"></canvas>
  </div>
  <div class="card" style="grid-column:7/span 6">
    <div class="card-title">Citation Coverage</div>
    <canvas id="citeBars" height="160"></canvas>
  </div>

  <!-- Business -->
  <div class="card" style="grid-column:1/span 6">
    <div class="card-title">Business Impact</div>
    <canvas id="bizChart" height="160"></canvas>
  </div>
  <div class="card" style="grid-column:7/span 6">
    <div class="card-title">ROI Progress (demo)</div>
    <div class="muted" style="margin-bottom:8px">Toward monthly target of 100 answered</div>
    <div class="progress" style="height:18px;background:var(--bg3);border-radius:9px">
      <div id="roiBar" class="progress-bar" style="width:0%;background:var(--green);border-radius:9px;transition:width .5s"></div>
    </div>
  </div>

  <!-- Health -->
  <div class="card" style="grid-column:1/span 12">
    <div class="card-title">Health</div>
    <div class="kpi-row">
      <div class="kpi-box"><div class="kpi-label">Error Rate</div><div class="kpi-value" id="h_err">–</div></div>
      <div class="kpi-box"><div class="kpi-label">Index Freshness</div><div class="kpi-value" style="font-size:14px;padding-top:4px" id="h_idx">–</div></div>
      <div class="kpi-box"><div class="kpi-label">Queue</div><div class="kpi-value" id="h_q">0</div></div>
    </div>
  </div>

</div>

<script>
const CHART_DEFAULTS = {
  color: '#e2e8f0',
  plugins: { legend: { labels: { color: '#e2e8f0', font: { size: 11 } } } },
  scales: { x: { ticks: { color: '#64748b' }, grid: { color: '#1a1a28' } }, y: { ticks: { color: '#64748b' }, grid: { color: '#1a1a28' } } }
};
const COLORS = ['#38bdf8','#a78bfa','#10b981','#f59e0b','#ef4444','#6366f1','#2dd4bf','#84cc16'];

let perfChart, latGauge, catDonut, volumeChart, confHist, citeBars, bizChart;

function mkChart(id, type, data, opts){
  const ctx = document.getElementById(id).getContext('2d');
  return new Chart(ctx, {type, data, options: opts});
}
function updateChart(ch, data){ ch.data=data; ch.update(); }

async function fetchJ(url){ const r=await fetch(url); return r.json(); }

async function refreshAll(){
  const range = document.getElementById('rangeSel').value;
  const [perf, usage, biz, health, raw, cov] = await Promise.all([
    fetchJ(`/api/metrics/performance?range=${range}`),
    fetchJ(`/api/metrics/usage?range=${range}`),
    fetchJ(`/api/metrics/business?range=${range}`),
    fetchJ(`/api/metrics/health?range=${range}`),
    fetchJ(`/api/metrics/raw_confidence?range=${range}`),
    fetchJ(`/api/metrics/citation_coverage?range=${range}`),
  ]);

  // KPIs
  document.getElementById('k_count').textContent = perf.count ?? 0;
  const lastLat = perf.latency_avg?.slice(-1)[0];
  document.getElementById('k_lat').textContent = lastLat != null ? lastLat + ' ms' : '–';
  document.getElementById('k_p90').textContent = perf.latency_p90 != null ? perf.latency_p90 + ' ms' : '–';
  const lastAbs = perf.abstention_rate?.slice(-1)[0];
  document.getElementById('k_abs').textContent = lastAbs != null ? (lastAbs*100).toFixed(0) + '%' : '–';

  // Performance line
  const perfData = {
    labels: perf.labels,
    datasets: [
      {label:'Abstention',data:perf.abstention_rate,yAxisID:'y1',borderColor:'#f59e0b',backgroundColor:'rgba(245,158,11,.1)',fill:true,tension:.35,borderWidth:2,pointRadius:2},
      {label:'Avg Latency (ms)',data:perf.latency_avg,yAxisID:'y2',borderColor:'#38bdf8',tension:.35,borderWidth:2,pointRadius:2}
    ]
  };
  const perfOpts = {responsive:true,interaction:{mode:'index'},plugins:{legend:{labels:{color:'#e2e8f0',font:{size:11}}}},scales:{x:{ticks:{color:'#64748b'},grid:{color:'#1a1a28'}},y1:{type:'linear',position:'left',min:0,max:1,ticks:{color:'#64748b'},grid:{color:'#1a1a28'}},y2:{type:'linear',position:'right',grid:{drawOnChartArea:false},ticks:{color:'#64748b'}}}};
  if(perfChart) updateChart(perfChart, perfData); else perfChart = mkChart('perfChart','line',perfData,perfOpts);

  // Latency gauge
  const p50=perf.latency_p50??0,p90=perf.latency_p90??0,p95=perf.latency_p95??0;
  const gData = {labels:['p50','p90','p95'],datasets:[{data:[p50,Math.max(0,p90-p50),Math.max(0,p95-p90)],backgroundColor:['#10b981','#f59e0b','#ef4444'],borderWidth:0}]};
  const gOpts = {cutout:'68%',plugins:{legend:{position:'bottom',labels:{color:'#e2e8f0',font:{size:11}}},tooltip:{callbacks:{label:(c)=>`${c.label}: ${[p50,p90,p95][c.dataIndex]} ms`}}}};
  if(latGauge) updateChart(latGauge,gData); else latGauge=mkChart('latGauge','doughnut',gData,gOpts);

  // Category donut
  const cData={labels:usage.category.labels,datasets:[{data:usage.category.counts,backgroundColor:COLORS,borderWidth:0}]};
  const cOpts={plugins:{legend:{position:'bottom',labels:{color:'#e2e8f0',font:{size:11}}}}};
  if(catDonut) updateChart(catDonut,cData); else catDonut=mkChart('catDonut','doughnut',cData,cOpts);

  // Volume
  const vData={labels:usage.timeline.labels,datasets:[{label:'Queries',data:usage.timeline.counts,backgroundColor:'rgba(56,189,248,.25)',borderColor:'#38bdf8',borderWidth:1.5}]};
  const vOpts={responsive:true,plugins:{legend:{labels:{color:'#e2e8f0',font:{size:11}}}},scales:{x:{ticks:{color:'#64748b'},grid:{color:'#1a1a28'}},y:{ticks:{color:'#64748b'},grid:{color:'#1a1a28'}}}};
  if(volumeChart) updateChart(volumeChart,vData); else volumeChart=mkChart('volumeChart','bar',vData,vOpts);

  // Confidence histogram
  const bins=new Array(10).fill(0);
  raw.values.forEach(v=>{ if(v==null)return; bins[Math.max(0,Math.min(9,Math.floor(v*10)))]++; });
  const hLabels=bins.map((_,i)=>`${(i*.1).toFixed(1)}`);
  const hColors=bins.map((_,i)=>i>=6?'rgba(16,185,129,.6)':i>=3?'rgba(245,158,11,.6)':'rgba(239,68,68,.6)');
  const hData={labels:hLabels,datasets:[{label:'Responses',data:bins,backgroundColor:hColors,borderRadius:4}]};
  const hOpts={responsive:true,plugins:{legend:{labels:{color:'#e2e8f0',font:{size:11}}}},scales:{x:{ticks:{color:'#64748b'},grid:{color:'#1a1a28'}},y:{ticks:{color:'#64748b'},grid:{color:'#1a1a28'}}}};
  if(confHist) updateChart(confHist,hData); else confHist=mkChart('confHist','bar',hData,hOpts);

  // Citation coverage
  const covData={labels:cov.labels,datasets:[{label:'Responses',data:cov.counts,backgroundColor:['rgba(239,68,68,.5)','rgba(245,158,11,.5)','rgba(16,185,129,.5)'],borderRadius:4}]};
  if(citeBars) updateChart(citeBars,covData); else citeBars=mkChart('citeBars','bar',covData,hOpts);

  // Business
  const bData={labels:['Cost Saved ($)','Time Saved (min)','Answered'],datasets:[{data:[biz.cost_saved_usd,biz.time_saved_minutes,biz.answered],backgroundColor:['rgba(56,189,248,.5)','rgba(99,102,241,.5)','rgba(16,185,129,.5)'],borderRadius:4}]};
  if(bizChart) updateChart(bizChart,bData); else bizChart=mkChart('bizChart','bar',bData,hOpts);

  // ROI bar
  const pct=Math.min(100,Math.round((biz.answered/100)*100));
  document.getElementById('roiBar').style.width=pct+'%';

  // Health
  const errPct=(health.error_rate*100).toFixed(1);
  const errEl=document.getElementById('h_err');
  errEl.textContent=errPct+'%';
  errEl.style.color=health.error_rate>0.05?'#ef4444':health.error_rate>0?'#f59e0b':'#10b981';
  document.getElementById('h_idx').textContent=health.index_freshness?new Date(health.index_freshness).toLocaleString():'–';
  document.getElementById('h_q').textContent=health.queue_len??0;
}

let timer=null;
function setAuto(){
  const on=document.getElementById('auto').checked;
  const sec=parseInt(document.getElementById('autoInt').value,10);
  if(timer){clearInterval(timer);timer=null;}
  if(on) timer=setInterval(refreshAll,sec*1000);
}
document.getElementById('auto').addEventListener('change',setAuto);
document.getElementById('autoInt').addEventListener('change',setAuto);
document.getElementById('rangeSel').addEventListener('change',refreshAll);
document.getElementById('exportCsv').addEventListener('click',()=>{ window.location=`/api/metrics/export_csv?range=${document.getElementById('rangeSel').value}`; });

refreshAll(); setAuto();
</script>"""
        return render_template_string(BASE_TEMPLATE, content=content, show_input=False)

    # ── Metrics API (unchanged) ──
    def _range_arg() -> str:
        return (request.args.get("range") or "today").lower()

    @app.get("/api/metrics/performance")
    def api_perf(): return jsonify(metrics_store.perf_timeseries(_range_arg()))

    @app.get("/api/metrics/accuracy")
    def api_acc(): return jsonify(metrics_store.accuracy_by_category(_range_arg()))

    @app.get("/api/metrics/business")
    def api_biz(): return jsonify(metrics_store.business_metrics(_range_arg()))

    @app.get("/api/metrics/usage")
    def api_usage(): return jsonify(metrics_store.usage_metrics(_range_arg()))

    @app.get("/api/metrics/health")
    def api_health(): return jsonify(metrics_store.health_metrics(_range_arg()))

    @app.get("/api/metrics/raw_confidence")
    def api_raw_conf():
        rows = metrics_store.window(_range_arg())
        return jsonify({"values": [r.get("confidence") for r in rows if r.get("confidence") is not None]})

    @app.get("/api/metrics/citation_coverage")
    def api_cite_cov():
        rows = metrics_store.window(_range_arg())
        b0  = sum(1 for r in rows if (r.get("citation_count") or 0) == 0)
        b12 = sum(1 for r in rows if 1 <= (r.get("citation_count") or 0) <= 2)
        b3p = sum(1 for r in rows if (r.get("citation_count") or 0) >= 3)
        return jsonify({"labels": ["0", "1–2", "3+"], "counts": [b0, b12, b3p]})

    @app.get("/api/metrics/export_csv")
    def api_export_csv():
        rows = metrics_store.window(_range_arg())
        header = ["ts","latency_ms","confidence","abstained","category","citation_count","sources","correct","token_cost","error"]
        def gen():
            yield ",".join(header) + "\n"
            for r in rows:
                vals = [
                    r.get("ts",""), str(r.get("latency_ms","")),
                    "" if r.get("confidence") is None else str(r.get("confidence")),
                    str(r.get("abstained","")), (r.get("category") or ""),
                    str(r.get("citation_count","")), "|".join(r.get("sources") or []),
                    "" if r.get("correct") is None else str(r.get("correct")),
                    "" if r.get("token_cost") is None else str(r.get("token_cost")),
                    str(r.get("error","")),
                ]
                yield ",".join(v.replace(",",";") for v in vals) + "\n"
        return Response(gen(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=metrics.csv"})

    return app


def run_web_ui():
    app = create_app()
    print("🌐 Web UI running at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
