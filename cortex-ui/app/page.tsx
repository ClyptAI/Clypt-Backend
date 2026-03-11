"use client";

import { FormEvent, useMemo, useState } from "react";
import { useRouter } from "next/navigation";

const YOUTUBE_PATTERN = /^(https?:\/\/)?(www\.)?(youtube\.com\/watch\?v=|youtu\.be\/)[A-Za-z0-9_-]{6,}/i;

export default function Page() {
  const router = useRouter();
  const [url, setUrl] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const canSubmit = useMemo(() => url.trim().length > 0, [url]);

  const onSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const value = url.trim();

    if (!YOUTUBE_PATTERN.test(value)) {
      setError("Enter a valid YouTube URL (youtube.com/watch?v=... or youtu.be/...).");
      return;
    }

    setSubmitting(true);
    try {
      const response = await fetch("/api/pipeline/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ videoUrl: value }),
      });
      const data = (await response.json()) as { ok?: boolean; error?: string; jobId?: string };
      if (!response.ok || !data.ok) {
        throw new Error(data.error || "Could not start pipeline job.");
      }

      setError("");
      localStorage.setItem("clypt:lastVideoUrl", value);
      router.push(`/graph?video=${encodeURIComponent(value)}&job=${encodeURIComponent(data.jobId ?? "")}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start pipeline.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <main className="relative min-h-screen overflow-hidden bg-[#040a12] text-[#e2e8f0]">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_left,#0f766e4a,transparent_35%),radial-gradient(circle_at_80%_20%,#fb718533,transparent_26%),radial-gradient(circle_at_bottom_right,#0ea5e930,transparent_40%)]" />

      <div className="relative mx-auto flex min-h-screen max-w-5xl flex-col justify-center px-6 py-20">
        <div className="mb-8">
          <p className="text-5xl font-semibold tracking-tight text-white/95 md:text-7xl">CLYPT</p>
        </div>
        <h1 className="max-w-3xl text-4xl font-semibold tracking-tight text-white md:text-6xl">
          Turn any YouTube link into a story map.
        </h1>
        <p className="mt-4 max-w-2xl text-sm text-[#94a3b8] md:text-base">
          Paste a link and jump into <span className="text-[#cbd5e1]">Cortex</span>, the map view for moments, punchlines, pivots, and top clip picks.
        </p>

        <form
          onSubmit={onSubmit}
          className="mt-10 rounded-2xl border border-[#1f2937] bg-[#0b1220d9] p-4 shadow-[0_10px_40px_rgba(2,6,23,0.45)] md:p-5"
        >
          <label htmlFor="youtube-url" className="mb-2 block text-xs uppercase tracking-[0.16em] text-[#94a3b8]">
            YouTube URL
          </label>
          <div className="flex flex-col gap-3 md:flex-row">
            <input
              id="youtube-url"
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://www.youtube.com/watch?v=..."
              className="h-12 flex-1 rounded-xl border border-[#334155] bg-[#111827] px-4 text-sm text-white outline-none transition focus:border-[#14b8a6]"
            />
            <button
              type="submit"
              disabled={!canSubmit || submitting}
              className="h-12 rounded-xl border border-[#0f766e] bg-[#115e59] px-5 text-sm font-medium text-[#ecfeff] transition hover:bg-[#0f766e] disabled:cursor-not-allowed disabled:opacity-60"
            >
              {submitting ? "Starting pipeline..." : "Open Cortex map"}
            </button>
          </div>
          {error && <p className="mt-2 text-xs text-[#fda4af]">{error}</p>}
        </form>
      </div>
    </main>
  );
}
