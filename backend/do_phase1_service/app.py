from __future__ import annotations

import os
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from backend.do_phase1_service.jobs import create_job, get_job
from backend.do_phase1_service.models import JobCreatePayload
from backend.do_phase1_service.state_store import SQLiteJobStore


DEFAULT_STATE_ROOT = Path(os.getenv("DO_PHASE1_STATE_ROOT", "/var/lib/clypt/do_phase1_service"))
DEFAULT_DB_PATH = Path(os.getenv("DO_PHASE1_DB_PATH", str(DEFAULT_STATE_ROOT / "jobs.db")))
DEFAULT_OUTPUT_ROOT = Path(os.getenv("DO_PHASE1_OUTPUT_ROOT", str(DEFAULT_STATE_ROOT / "workdir")))


def create_app(*, store: SQLiteJobStore | None = None, output_root: str | Path | None = None) -> FastAPI:
    output_root = Path(output_root or DEFAULT_OUTPUT_ROOT)

    app = FastAPI(title="Clypt DO Phase 1 Service")
    app.state.store = store
    app.state.db_path = DEFAULT_DB_PATH
    app.state.output_root = output_root

    def _store() -> SQLiteJobStore:
        if app.state.store is None:
            app.state.store = SQLiteJobStore(app.state.db_path)
        return app.state.store

    def _output_root() -> Path:
        app.state.output_root.mkdir(parents=True, exist_ok=True)
        return app.state.output_root

    @app.get("/healthz")
    def healthz() -> dict:
        live_store = _store()
        live_output_root = _output_root()
        return {
            "status": "ok",
            "sqlite": live_store.healthcheck(),
            "db_path": str(live_store.db_path),
            "output_root": str(live_output_root),
        }

    @app.post("/jobs", status_code=202)
    def post_jobs(payload: JobCreatePayload) -> dict:
        job = create_job(_store(), payload)
        return job.model_dump(mode="json")

    @app.get("/jobs/{job_id}")
    def get_job_status(job_id: str) -> dict:
        job = get_job(_store(), job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job.model_dump(mode="json")

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> str:
        return _dashboard_html()

    @app.get("/dashboard/api/jobs")
    async def dashboard_jobs(
        limit: int = 20,
        remote: bool = Query(default=False),
    ) -> dict:
        if remote:
            return await _proxy_dashboard_json("/dashboard/api/jobs", {"limit": max(1, min(200, limit))})

        jobs = [job.model_dump(mode="json") for job in _store().list_recent_jobs(limit=limit)]
        return {"jobs": jobs}

    @app.get("/dashboard/api/jobs/{job_id}")
    async def dashboard_job_status(job_id: str, remote: bool = Query(default=False)) -> dict:
        if remote:
            return await _proxy_dashboard_json(f"/jobs/{job_id}")
        job = get_job(_store(), job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job.model_dump(mode="json")

    @app.get("/dashboard/api/jobs/{job_id}/logs")
    async def dashboard_job_logs(
        job_id: str,
        tail_lines: int = 200,
        remote: bool = Query(default=False),
    ) -> dict:
        if remote:
            return await _proxy_dashboard_json(
                f"/jobs/{job_id}/logs",
                {"tail_lines": max(1, min(2000, tail_lines))},
            )
        return get_job_logs(job_id=job_id, tail_lines=tail_lines)

    @app.get("/jobs/{job_id}/logs")
    def get_job_logs(job_id: str, tail_lines: int = 200) -> dict:
        job = get_job(_store(), job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")

        log_path = Path(job.log_path) if job.log_path else (_output_root() / "logs" / f"{job_id}.log")
        if not log_path.exists():
            return {"job_id": job_id, "log_path": str(log_path), "lines": []}

        tail_lines = max(1, min(2000, tail_lines))
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return {
            "job_id": job_id,
            "log_path": str(log_path),
            "lines": lines[-tail_lines:],
        }

    @app.get("/jobs/{job_id}/result")
    def get_job_result(job_id: str) -> dict:
        job = get_job(_store(), job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if job.status != "succeeded" or job.manifest is None:
            raise HTTPException(status_code=409, detail="job result not ready")
        return job.manifest

    async def _proxy_dashboard_json(path: str, params: dict | None = None) -> dict:
        remote_base_url = os.getenv("DO_PHASE1_DASHBOARD_REMOTE_BASE_URL", "").strip()
        if not remote_base_url:
            raise HTTPException(
                status_code=400,
                detail="DO_PHASE1_DASHBOARD_REMOTE_BASE_URL is not configured for remote dashboard mode",
            )
        async with httpx.AsyncClient(base_url=remote_base_url, timeout=30.0) as client:
            response = await client.get(path, params=params)
            response.raise_for_status()
            return response.json()

    return app


app = create_app()


def _dashboard_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Clypt Phase 1 Monitor</title>
  <style>
    :root {
      --bg: #0d1117;
      --panel: #161b22;
      --border: #30363d;
      --text: #e6edf3;
      --muted: #8b949e;
      --accent: #58a6ff;
      --good: #3fb950;
      --warn: #d29922;
      --bad: #f85149;
      --mono: "IBM Plex Mono", "SFMono-Regular", Menlo, monospace;
      --sans: "IBM Plex Sans", "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(circle at top, #172033 0%, var(--bg) 48%);
      color: var(--text);
      font-family: var(--sans);
    }
    .wrap {
      max-width: 1480px;
      margin: 0 auto;
      padding: 28px;
    }
    h1 { margin: 0 0 8px; font-size: 32px; }
    .sub { color: var(--muted); margin-bottom: 20px; }
    .controls, .grid { display: grid; gap: 16px; }
    .controls { grid-template-columns: 1.1fr 0.8fr 0.5fr auto auto; margin-bottom: 18px; }
    .grid { grid-template-columns: 0.95fr 2fr; align-items: start; }
    .panel {
      background: color-mix(in srgb, var(--panel) 92%, #0b1220 8%);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 12px 40px rgba(0,0,0,0.2);
      min-height: 120px;
    }
    label { display:block; font-size: 12px; color: var(--muted); margin-bottom: 6px; letter-spacing: 0.04em; text-transform: uppercase; }
    input, button {
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: #0f1724;
      color: var(--text);
      padding: 12px 14px;
      font: inherit;
    }
    button {
      cursor: pointer;
      background: linear-gradient(135deg, #1f6feb, #388bfd);
      border: 0;
      font-weight: 600;
    }
    button.secondary {
      background: #202938;
      border: 1px solid var(--border);
    }
    .jobs { display: grid; gap: 10px; max-height: 70vh; overflow: auto; }
    .job {
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      background: #101722;
      cursor: pointer;
    }
    .job.active { border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }
    .row { display:flex; justify-content:space-between; gap:12px; align-items:center; }
    .mono { font-family: var(--mono); font-size: 12px; }
    .muted { color: var(--muted); }
    .badge {
      border-radius: 999px;
      padding: 4px 9px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .queued { background: rgba(210,153,34,0.16); color: #f2cc60; }
    .running { background: rgba(88,166,255,0.16); color: #79c0ff; }
    .succeeded { background: rgba(63,185,80,0.16); color: #56d364; }
    .failed { background: rgba(248,81,73,0.16); color: #ff7b72; }
    .progress {
      height: 10px;
      border-radius: 999px;
      background: #0b1220;
      border: 1px solid var(--border);
      overflow: hidden;
      margin-top: 10px;
    }
    .progress > div {
      height: 100%;
      background: linear-gradient(90deg, #1f6feb, #56d364);
      width: 0%;
      transition: width 200ms ease;
    }
    .detail-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }
    .stat {
      padding: 12px;
      border-radius: 12px;
      background: #0f1724;
      border: 1px solid var(--border);
    }
    .stat .value { font-weight: 700; font-size: 16px; margin-top: 4px; }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: #0a101a;
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      max-height: 52vh;
      overflow: auto;
      font-family: var(--mono);
      font-size: 12px;
      line-height: 1.5;
    }
    .empty {
      color: var(--muted);
      text-align: center;
      padding: 28px;
      border: 1px dashed var(--border);
      border-radius: 14px;
    }
    @media (max-width: 980px) {
      .controls, .grid, .detail-grid { grid-template-columns: 1fr; }
      .wrap { padding: 16px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Clypt Phase 1 Monitor</h1>
    <div class="sub">Local DO job dashboard with live status, progress, and logs.</div>
    <div class="controls">
      <div>
        <label>Remote Mode</label>
        <input id="remoteToggle" value="0" />
      </div>
      <div>
        <label>Refresh (sec)</label>
        <input id="refreshSeconds" value="5" />
      </div>
      <div>
        <label>Jobs</label>
        <input id="jobLimit" value="20" />
      </div>
      <div style="display:flex;align-items:end;">
        <button id="refreshBtn">Refresh</button>
      </div>
      <div style="display:flex;align-items:end;">
        <button id="autofocusBtn" class="secondary">Auto-select newest</button>
      </div>
    </div>
    <div class="grid">
      <section class="panel">
        <div class="row" style="margin-bottom:12px;">
          <strong>Jobs</strong>
          <span class="muted mono" id="jobsMeta">Loading…</span>
        </div>
        <div class="jobs" id="jobsList"></div>
      </section>
      <section class="panel">
        <div class="row" style="margin-bottom:14px;">
          <strong id="detailTitle">Job detail</strong>
          <span class="muted mono" id="detailUpdated"></span>
        </div>
        <div id="detailBody" class="empty">Choose a job to inspect progress.</div>
      </section>
    </div>
  </div>
  <script>
    const state = {
      selectedJobId: null,
      autoSelectNewest: true,
      timer: null,
    };

    function qs(id) { return document.getElementById(id); }
    function remoteEnabled() { return qs('remoteToggle').value.trim() === '1'; }
    function refreshMs() { return Math.max(1000, Number(qs('refreshSeconds').value || '5') * 1000); }
    function jobLimit() { return Math.max(1, Number(qs('jobLimit').value || '20')); }
    function badgeClass(status) { return ['badge', String(status || 'queued')].join(' '); }
    function escapeHtml(str) {
      return String(str ?? '').replace(/[&<>"]/g, (ch) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[ch]));
    }
    function apiPath(path) {
      const prefix = path.includes('?') ? '&' : '?';
      return `${path}${prefix}remote=${remoteEnabled() ? '1' : '0'}`;
    }

    async function loadJobs() {
      const res = await fetch(apiPath(`/dashboard/api/jobs?limit=${jobLimit()}`));
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || `Failed to load jobs (${res.status})`);
      }
      renderJobs(data.jobs || []);
      return data.jobs || [];
    }

    async function loadJob(jobId) {
      const [statusRes, logsRes] = await Promise.all([
        fetch(apiPath(`/dashboard/api/jobs/${jobId}`)),
        fetch(apiPath(`/dashboard/api/jobs/${jobId}/logs?tail_lines=300`)),
      ]);
      const job = await statusRes.json();
      const logs = await logsRes.json();
      if (!statusRes.ok) {
        throw new Error(job.detail || `Failed to load job (${statusRes.status})`);
      }
      if (!logsRes.ok) {
        throw new Error(logs.detail || `Failed to load logs (${logsRes.status})`);
      }
      renderDetail(job, logs);
    }

    function renderJobs(jobs) {
      qs('jobsMeta').textContent = `${jobs.length} loaded`;
      const list = qs('jobsList');
      if (!jobs.length) {
        list.innerHTML = '<div class="empty">No jobs found yet.</div>';
        return;
      }
      if (!state.selectedJobId || (state.autoSelectNewest && jobs[0])) {
        state.selectedJobId = state.selectedJobId || jobs[0].job_id;
      }
      list.innerHTML = jobs.map((job) => {
        const pct = Math.round((job.progress_pct || 0) * 100);
        return `
          <article class="job ${job.job_id === state.selectedJobId ? 'active' : ''}" data-job-id="${escapeHtml(job.job_id)}">
            <div class="row">
              <div>
                <div><strong>${escapeHtml(job.job_id)}</strong></div>
                <div class="mono muted">${escapeHtml(job.current_step || 'unknown')}</div>
              </div>
              <span class="${badgeClass(job.status)}">${escapeHtml(job.status)}</span>
            </div>
            <div class="muted" style="margin-top:8px;">${escapeHtml(job.progress_message || 'No progress message yet')}</div>
            <div class="progress"><div style="width:${pct}%;"></div></div>
            <div class="row muted mono" style="margin-top:8px;">
              <span>${pct}%</span>
              <span>${escapeHtml(job.updated_at || '')}</span>
            </div>
          </article>
        `;
      }).join('');

      list.querySelectorAll('.job').forEach((node) => {
        node.addEventListener('click', async () => {
          state.selectedJobId = node.dataset.jobId;
          state.autoSelectNewest = false;
          await refresh();
        });
      });
    }

    function renderDetail(job, logs) {
      qs('detailTitle').textContent = job.job_id;
      qs('detailUpdated').textContent = `Updated ${job.updated_at || 'unknown'}`;
      const pct = Math.round((job.progress_pct || 0) * 100);
      const body = `
        <div class="detail-grid">
          <div class="stat"><div class="muted">Status</div><div class="value">${escapeHtml(job.status)}</div></div>
          <div class="stat"><div class="muted">Step</div><div class="value">${escapeHtml(job.current_step || 'unknown')}</div></div>
          <div class="stat"><div class="muted">Progress</div><div class="value">${pct}%</div></div>
          <div class="stat"><div class="muted">Retries</div><div class="value">${escapeHtml(job.retries ?? 0)}</div></div>
        </div>
        <div class="stat" style="margin-bottom:12px;">
          <div class="muted">Message</div>
          <div class="value" style="font-size:14px;">${escapeHtml(job.progress_message || 'No message yet')}</div>
        </div>
        <div class="stat" style="margin-bottom:12px;">
          <div class="muted">Source</div>
          <div class="mono" style="margin-top:8px;">${escapeHtml(job.source_url || '')}</div>
        </div>
        <div class="stat" style="margin-bottom:12px;">
          <div class="muted">Log Path</div>
          <div class="mono" style="margin-top:8px;">${escapeHtml(logs.log_path || job.log_path || '')}</div>
        </div>
        <pre>${escapeHtml((logs.lines || []).join('\\n'))}</pre>
      `;
      qs('detailBody').innerHTML = body;
    }

    async function refresh() {
      try {
        const jobs = await loadJobs();
        if (!jobs.length) {
          qs('detailBody').innerHTML = '<div class="empty">Choose a job to inspect progress.</div>';
          return;
        }
        if (!state.selectedJobId) state.selectedJobId = jobs[0].job_id;
        const selected = jobs.find((job) => job.job_id === state.selectedJobId) || jobs[0];
        state.selectedJobId = selected.job_id;
        await loadJob(selected.job_id);
      } catch (error) {
        const message = escapeHtml(error?.message || String(error));
        qs('jobsMeta').textContent = 'Error';
        qs('jobsList').innerHTML = `<div class="empty">${message}</div>`;
        qs('detailBody').innerHTML = `<div class="empty">${message}</div>`;
      }
    }

    function armTimer() {
      clearInterval(state.timer);
      state.timer = setInterval(refresh, refreshMs());
    }

    qs('refreshBtn').addEventListener('click', refresh);
    qs('autofocusBtn').addEventListener('click', () => {
      state.autoSelectNewest = !state.autoSelectNewest;
      qs('autofocusBtn').textContent = state.autoSelectNewest ? 'Auto-select newest' : 'Pinned selection';
    });
    ['remoteToggle', 'refreshSeconds', 'jobLimit'].forEach((id) => {
      qs(id).addEventListener('change', () => {
        armTimer();
        refresh();
      });
    });

    armTimer();
    refresh();
  </script>
</body>
</html>"""
