# src/web_ui.py
from __future__ import annotations

import json
import os
import time
from collections import deque, defaultdict, Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, render_template_string, request, session, redirect, url_for, jsonify, Response
from src.qa import answer as qa_answer


# =========================
# Metrics collection / store
# =========================
class MetricsStore:
    """
    Simple JSONL-backed time-series buffer for RAG metrics.
    Fields we try to capture per interaction:
      - ts (UTC ISO)
      - latency_ms
      - confidence (0..1 or None)
      - abstained (bool)
      - category (str)
      - citation_count (int)
      - sources (list[str])
      - correct (bool|None)  # if you have labels
      - token_cost (float|None)  # if you track spend
    """

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
            # If file is corrupt, start fresh
            self.buffer.clear()

    def _append_file(self, rec: Dict[str, Any]) -> None:
        # Best-effort persistence
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

    # ---------- aggregations for endpoints ----------

    def perf_timeseries(self, range_key: str) -> Dict[str, Any]:
        rows = self.window(range_key)
        # bucket to 5-minute bins for charts
        bins = defaultdict(list)
        for r in rows:
            ts = datetime.fromisoformat(r["ts"])
            bucket = ts.replace(minute=(ts.minute // 5) * 5, second=0, microsecond=0)
            bins[bucket.isoformat()] += [r]

        # accuracy proxy: mean(correct) ignoring None
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

        # percentiles for gauges
        lats = sorted([r.get("latency_ms", 0) for r in rows if r.get("latency_ms") is not None])

        def pct(p):
            if not lats:
                return None
            idx = max(0, min(len(lats) - 1, int(len(lats) * p / 100)))
            return round(lats[idx], 1)

        return {
            "labels": labels,
            "accuracy": accuracy,
            "abstention_rate": abstention_rate,
            "latency_avg": latency_avg,
            "latency_p50": pct(50),
            "latency_p90": pct(90),
            "latency_p95": pct(95),
            "count": len(rows),
        }

    def accuracy_by_category(self, range_key: str) -> Dict[str, Any]:
        rows = self.window(range_key)
        by_cat = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in rows:
            cat = r.get("category") or "Uncategorized"
            by_cat[cat]["total"] += 1
            if r.get("correct") is True:
                by_cat[cat]["correct"] += 1
        labels, data = [], []
        for cat, v in sorted(by_cat.items(), key=lambda x: x[0].lower()):
            acc = v["correct"] / v["total"] if v["total"] else 0
            labels.append(cat)
            data.append(round(acc, 3))
        return {"labels": labels, "data": data}

    def business_metrics(self, range_key: str) -> Dict[str, Any]:
        rows = self.window(range_key)
        # very rough placeholders; wire real values if you have them
        token_cost = sum(r.get("token_cost", 0) or 0 for r in rows)
        time_saved_s = sum((r.get("abstained") is False) * 30 for r in rows)  # pretend 30s saved per answered
        return {
            "cost_saved_usd": round(max(0.0, 100.0 - token_cost), 2),
            "time_saved_minutes": round(time_saved_s / 60, 1),
            "answered": sum(1 for r in rows if not r.get("abstained")),
            "total": len(rows),
        }

    def usage_metrics(self, range_key: str) -> Dict[str, Any]:
        rows = self.window(range_key)
        # category distribution
        cats = Counter([r.get("category") or "Uncategorized" for r in rows])
        cat_labels = list(cats.keys())
        cat_counts = [cats[c] for c in cat_labels]

        # heatmap: hour(0-23) x day(0-6)
        heat = [[0] * 24 for _ in range(7)]
        for r in rows:
            ts = datetime.fromisoformat(r["ts"])
            heat[ts.weekday()][ts.hour] += 1

        # timeline volume
        bins = defaultdict(int)
        for r in rows:
            ts = datetime.fromisoformat(r["ts"])
            bucket = ts.replace(minute=0, second=0, microsecond=0).isoformat()
            bins[bucket] += 1
        tl_labels = sorted(bins.keys())
        tl_counts = [bins[k] for k in tl_labels]

        return {
            "category": {"labels": cat_labels, "counts": cat_counts},
            "heatmap": {"matrix": heat},  # client renders grid
            "timeline": {"labels": tl_labels, "counts": tl_counts},
        }

    def health_metrics(self, range_key: str) -> Dict[str, Any]:
        rows = self.window(range_key)
        errors = sum(1 for r in rows if r.get("error") is True)
        rate = (errors / len(rows)) if rows else 0.0
        last_index_update_iso = max((r.get("ts") for r in self.buffer), default=None)
        return {
            "error_rate": round(rate, 3),
            "queue_len": 0,
            "index_freshness": last_index_update_iso,
            "cpu": None,
            "mem": None,
        }


metrics_store = MetricsStore()


# =========================
# Global Base Template (visible to all routes)
# =========================
BASE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>⚕️ Diabetes RAG Assistant</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet" />
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet" />
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg:#181825; --bg-2:#12121b; --text:#f5f5f7; --line:#2d2d3a;
      --blue:#38bdf8; --accent-1:#9333ea; --accent-2:#6366f1; --accent-3:#06b6d4; --good:#10b981; --warn:#f59e0b; --bad:#ef4444;
    }
    * { box-sizing: border-box; }
    body { margin:0; font-family:'Inter',sans-serif; background:var(--bg); color:var(--text); display:flex; height:100vh; }
    .sidebar { width:250px; background:var(--bg-2); padding:20px; display:flex; flex-direction:column; border-right:1px solid var(--line); transition:width .3s ease; }
    .sidebar.collapsed { width:0; padding:0; overflow:hidden; }
    .logo { font-size:32px; font-weight:800; color:var(--blue); margin-bottom:30px; text-align:center; }
    .sidebar a { color:var(--text); text-decoration:none; padding:10px 0; font-size:15px; transition:.2s; }
    .sidebar a:hover { color:var(--blue); }
    .main { flex:1; display:flex; flex-direction:column; background:var(--bg); }
    .header { padding:10px 15px; border-bottom:1px solid var(--line); text-align:left; background:var(--bg); font-size:18px; font-weight:600; color:var(--blue); display:flex; align-items:center; gap:10px; }
    .toggle-btn { cursor:pointer; font-size:18px; background:none; border:none; color:var(--blue); }
    .chat-container { flex:1; overflow-y:auto; padding:20px; display:flex; flex-direction:column; gap:12px; font-size:14px; }
    .chat-bubble { max-width:75%; padding:10px 14px; border-radius:12px; line-height:1.5; word-wrap:break-word; }
    .user-msg { align-self:flex-end; background: linear-gradient(135deg, #3b82f6, var(--accent-3)); color:#fff; border-top-right-radius:0; }
    .assistant-msg { align-self:flex-start; background: linear-gradient(135deg, var(--accent-1), var(--accent-2)); color:#fff; border-top-left-radius:0; }
    .input-box { padding:12px; border-top:1px solid var(--line); background:var(--bg-2); display:flex; }
    .input-wrapper { position:relative; flex:1; }
    textarea { width:100%; border-radius:10px; padding:12px 45px 12px 12px; background:#2d2d3a; border:none; color:var(--text); font-size:15px; resize:none; height:55px; }
    textarea:focus { outline:none; border:1px solid var(--blue); }
    .send-btn { position:absolute; right:10px; top:50%; transform:translateY(-50%); background:none; border:none; cursor:pointer; font-size:20px; color:var(--blue); }
    .send-btn:hover { color:#1d9bf0; }
    /* Dashboard layout */
    .grid { display:grid; gap:16px; grid-template-columns: repeat(12, 1fr); }
    .card { background:var(--bg-2); border:1px solid var(--line); border-radius:12px; padding:12px; }
    .card h5 { margin:0 0 8px 0; color:var(--blue); font-weight:600; }
    .kpi { display:flex; gap:12px; }
    .kpi .box { flex:1; background:#1f1f2e; border:1px solid var(--line); border-radius:12px; padding:12px; }
    .muted { color:#cbd5e1; font-size:12px; }
    .switch-row { display:flex; gap:8px; align-items:center; justify-content:flex-end; }
    .btn-row { display:flex; gap:8px; justify-content:flex-end; }
    /* Transparent canvases (no big white blocks) */
    canvas { background:transparent; border-radius:10px; padding:6px; }

    /* KPI numbers: force white + bold */
    #k_count, #k_lat, #k_p90, #k_abs,
    #h_err, #h_idx, #h_q {
      color:#fff !important; font-size:22px; font-weight:700;
    }
  </style>
</head>
<body>
  <div class="sidebar" id="sidebar">
    <div class="logo">⚕️</div>
    <a href="/">💬 Chat</a>
    <a href="/dashboard">📊 Dashboard</a>
    <a href="/history">🕒 History</a>
    <a href="/clear_history" style="color:#f87171;">🗑️ Clear History</a>
  </div>

  <div class="main">
    <div class="header">
      <button class="toggle-btn" onclick="toggleSidebar()">☰</button>
      Diabetes RAG Assistant
    </div>
    <div class="chat-container" id="chatbox">
      {{ content|safe }}
    </div>
    {% if show_input %}
    <form method="POST" class="input-box">
      <div class="input-wrapper">
        <textarea name="question" placeholder="Ask anything about diabetes..."></textarea>
        <button type="submit" class="send-btn">➤</button>
      </div>
    </form>
    {% endif %}
  </div>

  <script>
    function toggleSidebar(){ document.getElementById("sidebar").classList.toggle("collapsed"); }
    var chatbox=document.getElementById("chatbox"); if(chatbox){ chatbox.scrollTop = chatbox.scrollHeight; }
  </script>
</body>
</html>"""


# =========================
# Flask app + routes
# =========================
def create_app():
    app = Flask(__name__)
    app.secret_key = "supersecretkey"  # sessions

    # ------------------------
    # Chat route
    # ------------------------
    @app.route("/", methods=["GET", "POST"])
    def rag():
        if "history" not in session:
            session["history"] = []

        if request.method == "POST":
            question = request.form.get("question", "").strip()
            if question:
                t0 = time.perf_counter()
                error_flag = False
                try:
                    res = qa_answer(question)
                    answer = getattr(res, "answer", "")
                    sources = getattr(res, "citations", []) or []
                    confidence = getattr(res, "confidence", None)
                    category = getattr(res, "category", None)
                    correct = getattr(res, "correct", None)
                    token_cost = getattr(res, "token_cost", None)
                except Exception as e:
                    answer = f"⚠️ {e}"
                    sources, confidence, category, correct, token_cost = [], None, None, None, None
                    error_flag = True
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)

                # save turn
                session["history"].append({"q": question, "a": answer, "sources": sources})
                session.modified = True

                # classify question category (unchanged logic)
                if not category:
                    ql = question.lower()
                    if any(k in ql for k in ["glucose", "cgm", "reading"]):
                        category = "Glucose"
                    elif any(k in ql for k in ["insulin", "dose", "bolus", "basal"]):
                        category = "Insulin"
                    elif any(k in ql for k in ["diet", "carb", "food", "meal"]):
                        category = "Nutrition"
                    elif any(k in ql for k in ["bp", "blood pressure", "hypertension"]):
                        category = "Cardio"
                    else:
                        category = "General"

                # record metrics safely
                rec = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "latency_ms": latency_ms,
                    "confidence": confidence,
                    "abstained": (confidence is None or confidence < 0.2 or not answer or answer.strip().lower().startswith("i don't know")),
                    "category": category,
                    "citation_count": len(sources),
                    "sources": list({s.get("source_id") for s in sources if isinstance(s, dict) and s.get("source_id")}),
                    "correct": correct,
                    "token_cost": token_cost,
                    "error": error_flag,
                }
                metrics_store.add(rec)

        # 🔹 rebuild full chat history for display
        content = ""
        for item in session.get("history", []):
            content += f'<div class="chat-bubble user-msg">{item["q"]}</div>'
            content += f'<div class="chat-bubble assistant-msg">{item["a"]}</div>'
            if item.get("sources"):
                sources_html = "<ul>" + "".join(
                    f"<li>{s.get('section','')} ({s.get('source_id','')}, chunk {s.get('chunk_idx','')})</li>"
                    for s in item["sources"]
                ) + "</ul>"
                content += f'<div class="chat-bubble assistant-msg"><b>📚 Sources:</b> {sources_html}</div>'

        return render_template_string(BASE_TEMPLATE, content=content, show_input=True)


    # ------------------------
    # History
    # ------------------------
    @app.route("/history")
    def history():
        content = "<h4>🕒 Chat History</h4><div class='chat-container'>"
        for item in session.get("history", []):
            content += f'<div class="chat-bubble user-msg">{item["q"]}</div>'
            content += f'<div class="chat-bubble assistant-msg">{item["a"]}</div>'
            if item.get("sources"):
                sources_html = "<ul>" + "".join(
                    f"<li>{s.get('section','')} ({s.get('source_id','')}, chunk {s.get('chunk_idx','')})</li>"
                    for s in item["sources"]
                ) + "</ul>"
                content += f'<div class="chat-bubble assistant-msg"><b>📚 Sources:</b> {sources_html}</div>'
        content += "</div>"
        return render_template_string(BASE_TEMPLATE, content=content, show_input=False)

    # ------------------------
    # Clear history
    # ------------------------
    @app.route("/clear_history")
    def clear_history():
        session.pop("history", None)
        return redirect(url_for("rag"))

    # =========================
    # Dashboard (clean + white KPI text)
    # =========================
    @app.route("/dashboard")
    def dashboard():
        content = """
        <div class="grid">
          <div class="card" style="grid-column: 1 / span 12;">
            <div class="switch-row">
              <div>
                <label class="muted">Time Range</label>
                <select id="rangeSel" class="form-select form-select-sm" style="width:160px; display:inline-block;">
                  <option value="last_hour">Last Hour</option>
                  <option value="today" selected>Today</option>
                  <option value="7d">Last 7 Days</option>
                  <option value="30d">Last 30 Days</option>
                </select>
              </div>
              <div>
                <label class="muted">Auto-refresh</label>
                <input type="checkbox" id="auto" />
                <select id="autoInt" class="form-select form-select-sm" style="width:120px; display:inline-block;">
                  <option value="5">5s</option>
                  <option value="30" selected>30s</option>
                  <option value="60">1m</option>
                  <option value="300">5m</option>
                </select>
                <button class="btn btn-sm btn-outline-light" id="exportCsv">Export CSV</button>
              </div>
            </div>
          </div>

          <!-- KPIs -->
          <div class="card kpi" style="grid-column: 1 / span 12;">
            <div class="box"><div class="muted">Requests</div><div id="k_count">0</div></div>
            <div class="box"><div class="muted">Avg Latency (ms)</div><div id="k_lat">–</div></div>
            <div class="box"><div class="muted">p90 Latency (ms)</div><div id="k_p90">–</div></div>
            <div class="box"><div class="muted">Abstention Rate</div><div id="k_abs">–</div></div>
          </div>

          <!-- Overview -->
          <div class="card" style="grid-column: 1 / span 8;">
            <h5>Performance Over Time</h5>
            <canvas id="perfChart" height="140"></canvas>
          </div>
          <div class="card" style="grid-column: 9 / span 4;">
            <h5>Latency Percentiles</h5>
            <canvas id="latGauge" height="140"></canvas>
            <div class="muted">p50, p90, p95</div>
          </div>

          <!-- Usage -->
          <div class="card" style="grid-column: 1 / span 4;">
            <h5>Question Categories</h5>
            <canvas id="catDonut" height="180"></canvas>
          </div>
          <div class="card" style="grid-column: 5 / span 8;">
            <h5>Volume Timeline</h5>
            <canvas id="volumeChart" height="180"></canvas>
          </div>

          <!-- Quality -->
          <div class="card" style="grid-column: 1 / span 6;">
            <h5>Confidence Distribution</h5>
            <canvas id="confHist" height="160"></canvas>
          </div>
          <div class="card" style="grid-column: 7 / span 6;">
            <h5>Citation Coverage</h5>
            <canvas id="citeBars" height="160"></canvas>
          </div>

          <!-- Business -->
          <div class="card" style="grid-column: 1 / span 6;">
            <h5>Business Impact</h5>
            <canvas id="bizChart" height="160"></canvas>
          </div>
          <div class="card" style="grid-column: 7 / span 6;">
            <h5>ROI Progress (demo)</h5>
            <div class="muted">Toward monthly target</div>
            <div class="progress" role="progressbar" aria-label="ROI" style="height:20px;">
              <div id="roiBar" class="progress-bar progress-bar-striped" style="width: 0%; background-color: var(--good);"></div>
            </div>
          </div>

          <!-- Health -->
          <div class="card" style="grid-column: 1 / span 12;">
            <h5>Health</h5>
            <div class="kpi">
              <div class="box"><div class="muted">Error Rate</div><div id="h_err">–</div></div>
              <div class="box"><div class="muted">Index Freshness</div><div id="h_idx">–</div></div>
              <div class="box"><div class="muted">Queue</div><div id="h_q">0</div></div>
            </div>
          </div>
        </div>

        <script>
        // --- charts ---
        let perfChart, latGauge, catDonut, volumeChart, confHist, citeBars, bizChart;

        function makeOrUpdateChart(ctx, type, data, options, instanceRef) {
          if (instanceRef && instanceRef.data) {
            instanceRef.data = data;
            instanceRef.options = options || {};
            instanceRef.update();
            return instanceRef;
          }
          return new Chart(ctx, {type, data, options});
        }

        function donutColors(n){
          const base=['#38bdf8','#9333ea','#10b981','#f59e0b','#ef4444','#6366f1','#06b6d4','#a78bfa','#22d3ee','#84cc16'];
          const out=[]; for(let i=0;i<n;i++){ out.push(base[i%base.length]); } return out;
        }

        async function fetchJSON(url){
          const res = await fetch(url);
          return await res.json();
        }

        async function refreshAll(){
          const range = document.getElementById('rangeSel').value;
          const perf = await fetchJSON(`/api/metrics/performance?range=${range}`);
          const usage = await fetchJSON(`/api/metrics/usage?range=${range}`);
          const acc   = await fetchJSON(`/api/metrics/accuracy?range=${range}`);
          const biz   = await fetchJSON(`/api/metrics/business?range=${range}`);
          const health= await fetchJSON(`/api/metrics/health?range=${range}`);

          // KPIs
          document.getElementById('k_count').innerText = perf.count ?? 0;
          document.getElementById('k_lat').innerText = (perf.latency_avg?.slice(-1)[0] ?? '–');
          document.getElementById('k_p90').innerText = (perf.latency_p90 ?? '–');
          document.getElementById('k_abs').innerText = (perf.abstention_rate?.slice(-1)[0] ?? '–');

          // Performance line (accuracy, abstention, latency)
          const ctxPerf = document.getElementById('perfChart').getContext('2d');
          perfChart = makeOrUpdateChart(ctxPerf, 'line', {
            labels: perf.labels,
            datasets: [
              {label:'Accuracy', data: perf.accuracy, yAxisID:'y1', borderWidth:2, tension:0.3},
              {label:'Abstention', data: perf.abstention_rate, yAxisID:'y1', borderWidth:2, tension:0.3},
              {label:'Avg Latency (ms)', data: perf.latency_avg, yAxisID:'y2', borderWidth:2, tension:0.3}
            ]
          },{
            responsive:true,
            plugins:{ legend:{labels:{color:'#fff'}} },
            scales:{
              x:{ticks:{color:'#fff'}},
              y1:{type:'linear', position:'left', min:0, max:1, ticks:{color:'#fff'}},
              y2:{type:'linear', position:'right', grid:{drawOnChartArea:false}, ticks:{color:'#fff'}}
            }
          }, perfChart);

          // Latency "gauge": doughnut with three segments (p50,p90,p95)
          const ctxGauge = document.getElementById('latGauge').getContext('2d');
          const p50 = perf.latency_p50 ?? 0, p90 = perf.latency_p90 ?? 0, p95 = perf.latency_p95 ?? 0;
          latGauge = makeOrUpdateChart(ctxGauge, 'doughnut', {
            labels:['p50','p90','p95'],
            datasets:[{data:[p50, Math.max(0,p90-p50), Math.max(0,p95-p90)], borderWidth:0}]
          }, {cutout:'70%', plugins:{legend:{position:'bottom', labels:{color:'#fff'}}}}, latGauge);

          // Category donut
          const ctxCat = document.getElementById('catDonut').getContext('2d');
          catDonut = makeOrUpdateChart(ctxCat, 'doughnut', {
            labels: usage.category.labels,
            datasets:[{data: usage.category.counts, backgroundColor: donutColors(usage.category.labels.length)}]
          }, {plugins:{legend:{position:'bottom', labels:{color:'#fff'}}}}, catDonut);

          // Volume timeline
          const ctxVol = document.getElementById('volumeChart').getContext('2d');
          volumeChart = makeOrUpdateChart(ctxVol, 'bar', {
            labels: usage.timeline.labels,
            datasets:[{label:'Queries', data: usage.timeline.counts}]
          }, {responsive:true, plugins:{legend:{labels:{color:'#fff'}}}, scales:{x:{ticks:{color:'#fff'}}, y:{ticks:{color:'#fff'}}}}, volumeChart);

          // Confidence histogram
          const raw = await fetchJSON(`/api/metrics/raw_confidence?range=${range}`);
          const bins = new Array(10).fill(0);
          raw.values.forEach(v => {
            if (v==null) return;
            const i = Math.max(0, Math.min(9, Math.floor(v*10)));
            bins[i] += 1;
          });
          const labels = bins.map((_,i)=>`${(i*0.1).toFixed(1)}-${((i+1)*0.1).toFixed(1)}`);
          const ctxConf = document.getElementById('confHist').getContext('2d');
          confHist = makeOrUpdateChart(ctxConf, 'bar', {
            labels, datasets:[{label:'Count', data: bins}]
          }, {responsive:true, plugins:{legend:{labels:{color:'#fff'}}}, scales:{x:{ticks:{color:'#fff'}}, y:{ticks:{color:'#fff'}}}}, confHist);

          // Citation coverage (0, 1-2, 3+)
          const cov = await fetchJSON(`/api/metrics/citation_coverage?range=${range}`);
          const ctxCite = document.getElementById('citeBars').getContext('2d');
          citeBars = makeOrUpdateChart(ctxCite, 'bar', {
            labels: cov.labels,
            datasets:[{label:'Responses', data: cov.counts}]
          }, {responsive:true, plugins:{legend:{labels:{color:'#fff'}}}, scales:{x:{ticks:{color:'#fff'}}, y:{ticks:{color:'#fff'}}}}, citeBars);

          // Business impact
          const ctxBiz = document.getElementById('bizChart').getContext('2d');
          bizChart = makeOrUpdateChart(ctxBiz, 'bar', {
            labels: ['Cost Saved (USD)', 'Time Saved (min)', 'Answered'],
            datasets:[{data:[biz.cost_saved_usd, biz.time_saved_minutes, biz.answered]}]
          }, {responsive:true, plugins:{legend:{labels:{color:'#fff'}}}, scales:{x:{ticks:{color:'#fff'}}, y:{ticks:{color:'#fff'}}}}, bizChart);

          // ROI progress (demo: answered/target)
          const target = 100;
          const pct = Math.min(100, Math.round((biz.answered/target)*100));
          document.getElementById('roiBar').style.width = pct + '%';

          // Health
          document.getElementById('h_err').innerText = (health.error_rate*100).toFixed(1)+'%';
          document.getElementById('h_idx').innerText = health.index_freshness ? new Date(health.index_freshness).toLocaleString() : '–';
          document.getElementById('h_q').innerText = health.queue_len ?? 0;
        }

        // auto refresh
        let timer=null;
        function setAuto(){
          const enabled = document.getElementById('auto').checked;
          const sec = parseInt(document.getElementById('autoInt').value, 10);
          if (timer) { clearInterval(timer); timer=null; }
          if (enabled) { timer = setInterval(refreshAll, sec*1000); }
        }
        document.getElementById('auto').addEventListener('change', setAuto);
        document.getElementById('autoInt').addEventListener('change', setAuto);
        document.getElementById('rangeSel').addEventListener('change', refreshAll);

        // export CSV
        document.getElementById('exportCsv').addEventListener('click', ()=> {
          const range = document.getElementById('rangeSel').value;
          window.location = `/api/metrics/export_csv?range=${range}`;
        });

        // initial
        refreshAll(); setAuto();
        </script>
        """
        return render_template_string(BASE_TEMPLATE, content=content, show_input=False)

    # =========================
    # Metrics API endpoints
    # =========================
    def _range_arg() -> str:
        return (request.args.get("range") or "today").lower()

    @app.get("/api/metrics/performance")
    def api_perf():
        return jsonify(metrics_store.perf_timeseries(_range_arg()))

    @app.get("/api/metrics/accuracy")
    def api_acc():
        return jsonify(metrics_store.accuracy_by_category(_range_arg()))

    @app.get("/api/metrics/business")
    def api_biz():
        return jsonify(metrics_store.business_metrics(_range_arg()))

    @app.get("/api/metrics/usage")
    def api_usage():
        return jsonify(metrics_store.usage_metrics(_range_arg()))

    @app.get("/api/metrics/health")
    def api_health():
        return jsonify(metrics_store.health_metrics(_range_arg()))

    # extras used by dashboard
    @app.get("/api/metrics/raw_confidence")
    def api_raw_conf():
        rows = metrics_store.window(_range_arg())
        vals = [r.get("confidence") for r in rows if r.get("confidence") is not None]
        return jsonify({"values": vals})

    @app.get("/api/metrics/citation_coverage")
    def api_cite_cov():
        rows = metrics_store.window(_range_arg())
        b0 = sum(1 for r in rows if (r.get("citation_count") or 0) == 0)
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
                    r.get("ts",""),
                    str(r.get("latency_ms","")),
                    "" if r.get("confidence") is None else str(r.get("confidence")),
                    str(r.get("abstained","")),
                    (r.get("category") or ""),
                    str(r.get("citation_count","")),
                    "|".join(r.get("sources") or []),
                    "" if r.get("correct") is None else str(r.get("correct")),
                    "" if r.get("token_cost") is None else str(r.get("token_cost")),
                    str(r.get("error","")),
                ]
                yield ",".join(v.replace(",", ";") for v in vals) + "\n"
        return Response(gen(), mimetype="text/csv",
                        headers={"Content-Disposition":"attachment; filename=metrics.csv"})

    return app


def run_web_ui():
    app = create_app()
    print("🌐 Web UI running at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
