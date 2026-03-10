import "./index.css";
import { Composition, staticFile } from "remotion";
import { ClyptViralShort } from "./ClyptViralShort";
import type {
  CameraKeyframe,
  SpeakerSegment,
  TrackingFrame,
  VideoLetterbox,
} from "./ClyptViralShort";

interface Payload {
  clip_start_ms: number;
  clip_end_ms: number;
  tracking_uris?: string[];
  active_speaker_timeline?: SpeakerSegment[];
  [key: string]: unknown;
}

interface MergedClipTracking {
  frames?: TrackingFrame[];
  speaker_word_timeline?: SpeakerSegment[];
  asd_active_speaker_timeline?: SpeakerSegment[];
  fused_active_speaker_timeline?: SpeakerSegment[];
  camera_path?: CameraKeyframe[];
  video_letterbox?: VideoLetterbox | null;
}

let payloads: Payload[] = [];
try {
  const raw = require("./remotion_payloads_array.json");
  payloads = Array.isArray(raw) ? raw : [raw];
} catch {
  try {
    const single = require("./remotion_payload.json");
    payloads = [single];
  } catch {
    // no payload found — compositions will be empty
  }
}

let allTracking: Record<string, TrackingFrame[] | MergedClipTracking> = {};
try {
  allTracking = require("../public/merged_tracking.json");
} catch {
  // not yet generated
}

// Most podcast source videos are 23.976 fps; using 24 fps composition
// avoids constant cadence judder from 24 -> 30 conversion.
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
        const speakerTimeline: SpeakerSegment[] =
          payload.active_speaker_timeline || [];
        const wordSpeakerTimeline: SpeakerSegment[] = Array.isArray(clipTracking)
          ? []
          : (clipTracking?.speaker_word_timeline || []);
        const asdSpeakerTimeline: SpeakerSegment[] = Array.isArray(clipTracking)
          ? []
          : (clipTracking?.asd_active_speaker_timeline || []);
        const fusedSpeakerTimeline: SpeakerSegment[] = Array.isArray(clipTracking)
          ? []
          : (clipTracking?.fused_active_speaker_timeline || []);
        const cameraPath: CameraKeyframe[] = Array.isArray(clipTracking)
          ? []
          : (clipTracking?.camera_path || []);
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
              speakerTimeline,
              wordSpeakerTimeline,
              asdSpeakerTimeline,
              fusedSpeakerTimeline,
              cameraPath,
              videoLetterbox,
            }}
          />
        );
      })}
    </>
  );
};
