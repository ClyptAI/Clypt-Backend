import { NextResponse } from "next/server";
import { existsSync, readFileSync } from "fs";
import { join, resolve } from "path";

function getOutputsDir() {
  const candidates = [
    resolve(process.cwd(), "..", "..", "outputs"),
    resolve(process.cwd(), "..", "outputs"),
  ];
  return candidates.find((dir) => existsSync(dir)) ?? candidates[0];
}

export async function GET() {
  const outputsDir = getOutputsDir();

  let nodes = [];
  let edges = [];

  try {
    nodes = JSON.parse(
      readFileSync(join(outputsDir, "phase_1b_nodes.json"), "utf-8")
    );
  } catch {
    // nodes file may not exist yet
  }
  try {
    edges = JSON.parse(
      readFileSync(join(outputsDir, "phase_1c_narrative_edges.json"), "utf-8")
    );
  } catch {
    // edges file may not exist yet
  }

  let payloads = [];
  try {
    payloads = JSON.parse(
      readFileSync(join(outputsDir, "remotion_payloads_array.json"), "utf-8")
    );
  } catch {
    // payloads file may not exist yet
  }

  return NextResponse.json({ nodes, edges, payloads });
}

export async function POST(req: Request) {
  const { nodes, edges } = await req.json();
  const outputsDir = getOutputsDir();
  const { mkdirSync, writeFileSync } = await import("fs");
  mkdirSync(outputsDir, { recursive: true });

  writeFileSync(
    join(outputsDir, "phase_1b_nodes.json"),
    JSON.stringify(nodes, null, 2)
  );
  writeFileSync(
    join(outputsDir, "phase_1c_narrative_edges.json"),
    JSON.stringify(edges, null, 2)
  );

  return NextResponse.json({ ok: true });
}
