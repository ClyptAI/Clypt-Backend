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
      --bg: #050505;
      --bg-soft: #0b0b0c;
      --panel: #111113;
      --panel-strong: #17171a;
      --border: #2a2a2f;
      --border-strong: #3b0f14;
      --text: #fcfcfd;
      --muted: #a9abb3;
      --accent: #ff4d5c;
      --accent-strong: #ff2739;
      --accent-soft: rgba(255, 77, 92, 0.12);
      --good: #6ee7b7;
      --warn: #ffd166;
      --bad: #ff6b76;
      --shadow: 0 20px 60px rgba(0,0,0,0.42);
      --mono: "IBM Plex Mono", "SFMono-Regular", Menlo, monospace;
      --sans: "IBM Plex Sans", "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at top right, rgba(255, 39, 57, 0.14), transparent 26%),
        radial-gradient(circle at top left, rgba(255, 77, 92, 0.09), transparent 22%),
        linear-gradient(180deg, #0b0b0c 0%, var(--bg) 62%);
      color: var(--text);
      font-family: var(--sans);
    }
    .wrap {
      max-width: 1680px;
      margin: 0 auto;
      padding: 28px 24px 32px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 34px;
      letter-spacing: -0.03em;
    }
    .sub {
      color: var(--muted);
      margin-bottom: 22px;
      max-width: 860px;
    }
    .controls, .grid { display: grid; gap: 16px; }
    .controls {
      grid-template-columns: minmax(140px, 0.9fr) minmax(140px, 0.8fr) minmax(120px, 0.8fr) auto auto;
      margin-bottom: 18px;
    }
    .grid {
      grid-template-columns: minmax(300px, 360px) minmax(0, 1fr);
      align-items: stretch;
      min-height: calc(100vh - 190px);
    }
    .panel {
      background: linear-gradient(180deg, color-mix(in srgb, var(--panel) 90%, #000 10%), var(--bg-soft));
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      box-shadow: var(--shadow);
      min-height: 120px;
    }
    .panel-shell {
      display: flex;
      flex-direction: column;
      min-height: 0;
    }
    label {
      display:block;
      font-size: 11px;
      color: var(--muted);
      margin-bottom: 6px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    input, button {
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: #0c0c0f;
      color: var(--text);
      padding: 12px 14px;
      font: inherit;
    }
    input:focus, button:focus {
      outline: 2px solid rgba(255, 77, 92, 0.35);
      outline-offset: 2px;
      border-color: var(--accent);
    }
    button {
      cursor: pointer;
      background: linear-gradient(135deg, var(--accent-strong), var(--accent));
      border: 1px solid rgba(255, 255, 255, 0.06);
      font-weight: 600;
    }
    button.secondary {
      background: #17171b;
      border: 1px solid var(--border);
    }
    .jobs {
      display: grid;
      gap: 10px;
      max-height: calc(100vh - 280px);
      overflow: auto;
      padding-right: 4px;
    }
    .job {
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      background: linear-gradient(180deg, #121216, #0a0a0d);
      cursor: pointer;
      transition: border-color 160ms ease, transform 160ms ease, background 160ms ease;
    }
    .job:hover {
      transform: translateY(-1px);
      border-color: color-mix(in srgb, var(--accent) 48%, var(--border));
    }
    .job.active {
      border-color: var(--accent);
      box-shadow: inset 0 0 0 1px var(--accent);
      background: linear-gradient(180deg, #181217, #0d0b0d);
    }
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
    .queued { background: rgba(255, 209, 102, 0.12); color: var(--warn); }
    .running { background: rgba(255, 77, 92, 0.16); color: #ff9aa3; }
    .succeeded { background: rgba(110, 231, 183, 0.12); color: var(--good); }
    .failed { background: rgba(255, 107, 118, 0.18); color: var(--bad); }
    .progress {
      height: 10px;
      border-radius: 999px;
      background: #09090b;
      border: 1px solid var(--border);
      overflow: hidden;
      margin-top: 10px;
    }
    .progress > div {
      height: 100%;
      background: linear-gradient(90deg, var(--accent-strong), #ff7b72 56%, #ffffff 100%);
      width: 0%;
      transition: width 200ms ease;
    }
    .detail-panel {
      display: flex;
      flex-direction: column;
      min-height: 0;
    }
    .detail-shell {
      display: flex;
      flex-direction: column;
      gap: 14px;
      min-height: calc(100vh - 280px);
    }
    .summary-strip {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      position: sticky;
      top: 0;
      z-index: 1;
      background: linear-gradient(180deg, rgba(17, 17, 19, 0.98), rgba(17, 17, 19, 0.92));
      padding-bottom: 2px;
    }
    .detail-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .stat {
      padding: 12px;
      border-radius: 12px;
      background: linear-gradient(180deg, var(--panel-strong), #0d0d10);
      border: 1px solid var(--border);
    }
    .stat .value { font-weight: 700; font-size: 16px; margin-top: 4px; }
    .detail-meta .value {
      font-size: 13px;
      font-weight: 500;
      line-height: 1.55;
      word-break: break-word;
    }
    .log-toolbar {
      display: grid;
      grid-template-columns: minmax(120px, 160px) auto auto auto;
      gap: 10px;
      align-items: end;
    }
    .checkbox {
      display: flex;
      gap: 10px;
      align-items: center;
      min-height: 46px;
      padding: 0 12px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: #0c0c0f;
      color: var(--text);
      font-size: 13px;
      cursor: pointer;
      user-select: none;
    }
    .checkbox input {
      width: auto;
      margin: 0;
      accent-color: var(--accent);
    }
    .log-card {
      display: flex;
      flex-direction: column;
      min-height: 0;
      border: 1px solid var(--border);
      border-radius: 16px;
      background: linear-gradient(180deg, #0a0a0d, #070709);
      overflow: hidden;
    }
    .log-header {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(180deg, rgba(255, 77, 92, 0.08), rgba(255, 77, 92, 0.02));
    }
    .log-title {
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: #ffd7db;
    }
    .log-meta {
      font-size: 12px;
      color: var(--muted);
    }
    pre {
      margin: 0;
      white-space: pre;
      word-break: normal;
      background: transparent;
      border: 0;
      border-radius: 0;
      padding: 16px 18px 22px;
      min-height: 0;
      height: 100%;
      overflow: auto;
      font-family: var(--mono);
      font-size: 12px;
      line-height: 1.58;
      tab-size: 2;
      color: #fff5f6;
    }
    pre.wrap {
      white-space: pre-wrap;
      word-break: break-word;
    }
    .empty {
      color: var(--muted);
      text-align: center;
      padding: 28px;
      border: 1px dashed var(--border);
      border-radius: 14px;
    }
    .inline-note {
      color: var(--muted);
      font-size: 12px;
    }
    @media (max-width: 980px) {
      .controls,
      .grid,
      .detail-grid,
      .summary-strip,
      .log-toolbar { grid-template-columns: 1fr; }
      .wrap { padding: 16px; }
      .jobs { max-height: none; }
      .detail-shell { min-height: auto; }
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
      <section class="panel detail-panel">
        <div class="row" style="margin-bottom:14px;">
          <strong id="detailTitle">Job detail</strong>
          <span class="muted mono" id="detailUpdated"></span>
        </div>
        <div id="detailBody" class="detail-shell empty">Choose a job to inspect progress.</div>
      </section>
    </div>
  </div>
  <script>
    const state = {
      selectedJobId: null,
      autoSelectNewest: true,
      timer: null,
      wrapLogs: true,
      tailLines: 500,
    };

    function qs(id) { return document.getElementById(id); }
    function remoteEnabled() { return qs('remoteToggle').value.trim() === '1'; }
    function refreshMs() { return Math.max(1000, Number(qs('refreshSeconds').value || '5') * 1000); }
    function jobLimit() { return Math.max(1, Number(qs('jobLimit').value || '20')); }
    function tailLines() { return Math.max(50, Math.min(2000, Number(state.tailLines || 500))); }
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
        fetch(apiPath(`/dashboard/api/jobs/${jobId}/logs?tail_lines=${tailLines()}`)),
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
        <div class="summary-strip">
          <div class="stat"><div class="muted">Status</div><div class="value">${escapeHtml(job.status)}</div></div>
          <div class="stat"><div class="muted">Step</div><div class="value">${escapeHtml(job.current_step || 'unknown')}</div></div>
          <div class="stat"><div class="muted">Progress</div><div class="value">${pct}%</div></div>
          <div class="stat"><div class="muted">Retries</div><div class="value">${escapeHtml(job.retries ?? 0)}</div></div>
        </div>
        <div class="detail-grid">
          <div class="stat detail-meta">
            <div class="muted">Message</div>
            <div class="value">${escapeHtml(job.progress_message || 'No message yet')}</div>
          </div>
          <div class="stat detail-meta">
            <div class="muted">Updated</div>
            <div class="value mono">${escapeHtml(job.updated_at || 'unknown')}</div>
          </div>
        </div>
        <div class="detail-grid">
          <div class="stat detail-meta">
            <div class="muted">Source</div>
            <div class="value mono">${escapeHtml(job.source_url || '')}</div>
          </div>
          <div class="stat detail-meta">
            <div class="muted">Log Path</div>
            <div class="value mono">${escapeHtml(logs.log_path || job.log_path || '')}</div>
          </div>
        </div>
        <div class="log-toolbar">
          <div>
            <label>Tail lines</label>
            <input id="tailLinesInput" value="${tailLines()}" />
          </div>
          <label class="checkbox">
            <input type="checkbox" id="wrapLogsToggle" ${state.wrapLogs ? 'checked' : ''} />
            <span>Wrap lines</span>
          </label>
          <div style="display:flex;align-items:end;">
            <button id="reloadLogsBtn" class="secondary">Reload logs</button>
          </div>
          <div style="display:flex;align-items:end;">
            <button id="copyLogsBtn" class="secondary">Copy logs</button>
          </div>
        </div>
        <div class="log-card">
          <div class="log-header">
            <div>
              <div class="log-title">Live log tail</div>
              <div class="log-meta">${escapeHtml((logs.lines || []).length)} lines loaded</div>
            </div>
            <div class="inline-note">Auto-refresh keeps the log pane current.</div>
          </div>
          <pre id="jobLogPane" class="${state.wrapLogs ? 'wrap' : ''}">${escapeHtml((logs.lines || []).join('\\n'))}</pre>
        </div>
      `;
      qs('detailBody').innerHTML = body;
      const wrapToggle = qs('wrapLogsToggle');
      const tailInput = qs('tailLinesInput');
      const reloadBtn = qs('reloadLogsBtn');
      const copyBtn = qs('copyLogsBtn');
      const logPane = qs('jobLogPane');
      if (wrapToggle) {
        wrapToggle.addEventListener('change', () => {
          state.wrapLogs = wrapToggle.checked;
          if (logPane) {
            logPane.classList.toggle('wrap', state.wrapLogs);
          }
        });
      }
      if (tailInput) {
        tailInput.addEventListener('change', () => {
          const nextValue = Number(tailInput.value || tailLines());
          state.tailLines = Math.max(50, Math.min(2000, nextValue));
          tailInput.value = String(state.tailLines);
          loadJob(job.job_id);
        });
      }
      if (reloadBtn) {
        reloadBtn.addEventListener('click', () => loadJob(job.job_id));
      }
      if (copyBtn) {
        copyBtn.addEventListener('click', async () => {
          try {
            await navigator.clipboard.writeText((logs.lines || []).join('\\n'));
            copyBtn.textContent = 'Copied';
            setTimeout(() => { copyBtn.textContent = 'Copy logs'; }, 1200);
          } catch (_error) {
            copyBtn.textContent = 'Copy failed';
            setTimeout(() => { copyBtn.textContent = 'Copy logs'; }, 1200);
          }
        });
      }
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
