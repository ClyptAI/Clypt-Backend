import { NextResponse } from "next/server";
import { closeSync, existsSync, mkdirSync, openSync, readFileSync } from "fs";
import { resolve } from "path";
import { spawn } from "child_process";

const YOUTUBE_PATTERN =
  /^(https?:\/\/)?(www\.)?(youtube\.com\/watch\?v=|youtu\.be\/)[A-Za-z0-9_-]{6,}/i;

function getRepoRoot() {
  const candidates = [resolve(process.cwd(), "..", ".."), resolve(process.cwd(), "..")];
  return candidates.find((dir) => existsSync(resolve(dir, "pipeline"))) ?? candidates[0];
}

export async function POST(req: Request) {
  try {
    const body = (await req.json()) as { videoUrl?: string };
    const videoUrl = (body.videoUrl ?? "").trim();
    if (!videoUrl || !YOUTUBE_PATTERN.test(videoUrl)) {
      return NextResponse.json(
        { ok: false, error: "Invalid YouTube URL." },
        { status: 400 }
      );
    }

    const repoRoot = getRepoRoot();
    const outputsDir = resolve(repoRoot, "outputs");
    mkdirSync(outputsDir, { recursive: true });

    const statePath = resolve(outputsDir, "pipeline_job_state.json");
    const logPath = resolve(outputsDir, "pipeline_job.log");
    if (existsSync(statePath)) {
      const current = JSON.parse(readFileSync(statePath, "utf-8")) as { status?: string };
      if (current.status === "running") {
        return NextResponse.json(
          { ok: false, error: "A pipeline job is already running." },
          { status: 409 }
        );
      }
    }

    const jobId = `job-${Date.now()}`;
    const pythonBin = process.env.PYTHON_BIN ?? "python3";

    const outFd = openSync(logPath, "a");
    const errFd = openSync(logPath, "a");

    const child = spawn(
      pythonBin,
      [
        resolve(repoRoot, "scripts", "run_pipeline_job.py"),
        "--url",
        videoUrl,
        "--job-id",
        jobId,
        "--state-path",
        statePath,
      ],
      {
        cwd: repoRoot,
        detached: true,
        stdio: ["ignore", outFd, errFd],
      }
    );

    child.unref();
    closeSync(outFd);
    closeSync(errFd);

    return NextResponse.json({ ok: true, jobId });
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        error: error instanceof Error ? error.message : "Failed to start pipeline.",
      },
      { status: 500 }
    );
  }
}
