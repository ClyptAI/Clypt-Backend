import "./index.css";
import { Composition, staticFile } from "remotion";
import { ClyptViralShort } from "./ClyptViralShort";
import type { TrackingFrame, VideoLetterbox } from "./ClyptViralShort";

interface Payload {
  clip_start_ms: number;
  clip_end_ms: number;
  tracking_uris?: string[];
  [key: string]: unknown;
}

interface MergedClipTracking {
  frames?: TrackingFrame[];
  video_letterbox?: VideoLetterbox | null;
}

let payloads: Payload[] = [];
try {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const raw = require("./remotion_payloads_array.json");
  payloads = Array.isArray(raw) ? raw : [raw];
} catch {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const single = require("./remotion_payload.json");
    payloads = [single];
  } catch {
    // no payload found — compositions will be empty
  }
}

let allTracking: Record<string, TrackingFrame[] | MergedClipTracking> = {};
try {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  allTracking = require("../public/merged_tracking.json");
} catch {
  // not yet generated
}

const FPS = 24;

export const RemotionRoot: React.FC = () => {
  return (
    <>
      {payloads.map((payload, i) => {
        const clipDurationMs = payload.clip_end_ms - payload.clip_start_ms;
        const durationInFrames = Math.ceil((clipDurationMs / 1000) * FPS);
        const clipId = payloads.length === 1
          ? "ClyptViralShort"
          : `ClyptViralShort-${i + 1}`;

        const clipTracking = allTracking[String(i)];
        const tracking: TrackingFrame[] = Array.isArray(clipTracking)
          ? clipTracking
          : (clipTracking?.frames || []);
        const videoLetterbox: VideoLetterbox | null = Array.isArray(clipTracking)
          ? null
          : (clipTracking?.video_letterbox || null);

        return (
          <Composition
            key={clipId}
            id={clipId}
            component={ClyptViralShort}
            durationInFrames={durationInFrames}
            fps={FPS}
            width={1080}
            height={1920}
            defaultProps={{
              clipStartMs: payload.clip_start_ms,
              clipEndMs: payload.clip_end_ms,
              videoSrc: staticFile("video.mp4"),
              tracking,
              videoLetterbox,
            }}
          />
        );
      })}
    </>
  );
};
