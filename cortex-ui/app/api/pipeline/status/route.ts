import { NextResponse } from "next/server";
import { existsSync, readFileSync } from "fs";
import { resolve } from "path";

type PipelineState = {
  jobId?: string;
  status?: "idle" | "running" | "completed" | "failed";
  error?: string;
  videoUrl?: string;
  startedAt?: string;
  finishedAt?: string;
  updatedAt?: string;
};

function getRepoRoot() {
  const candidates = [resolve(process.cwd(), "..", ".."), resolve(process.cwd(), "..")];
  return candidates.find((dir) => existsSync(resolve(dir, "pipeline"))) ?? candidates[0];
}

export async function GET() {
  try {
    const repoRoot = getRepoRoot();
    const statePath = resolve(repoRoot, "outputs", "pipeline_job_state.json");

    if (!existsSync(statePath)) {
      return NextResponse.json({ status: "idle" });
    }

    const raw = readFileSync(statePath, "utf-8");
    const state = JSON.parse(raw) as PipelineState;
    return NextResponse.json({
      status: state.status ?? "idle",
      jobId: state.jobId,
      error: state.error,
      videoUrl: state.videoUrl,
      startedAt: state.startedAt,
      finishedAt: state.finishedAt,
      updatedAt: state.updatedAt,
    });
  } catch (error) {
    return NextResponse.json(
      {
        status: "failed",
        error: error instanceof Error ? error.message : "Failed to read pipeline status.",
      },
      { status: 500 }
    );
  }
}
