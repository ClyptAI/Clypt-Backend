import {
  AbsoluteFill,
  OffthreadVideo,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { useMemo } from "react";

export interface TrackingFrame {
  time_ms: number;
  center_x: number;
  center_y: number;
  source?: string;
  speaker_tag?: string | null;
  track_id?: string | number | null;
  face_track_id?: string | number | null;
  bbox_w?: number;
  bbox_h?: number;
  confidence?: number | null;
}

export interface VideoLetterbox {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface ClyptViralShortProps extends Record<string, unknown> {
  clipStartMs: number;
  clipEndMs: number;
  videoSrc: string;
  tracking: TrackingFrame[];
  videoLetterbox?: VideoLetterbox | null;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function findClosestFrame(
  tracking: TrackingFrame[],
  targetMs: number,
): TrackingFrame | null {
  if (tracking.length === 0) return null;

  let lo = 0;
  let hi = tracking.length - 1;

  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (tracking[mid].time_ms < targetMs) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  const right = tracking[lo];
  const left = lo > 0 ? tracking[lo - 1] : right;

  return Math.abs(left.time_ms - targetMs) <= Math.abs(right.time_ms - targetMs)
    ? left
    : right;
}

// 16:9 source into 9:16 viewport baseline.
const SOURCE_ASPECT = 16 / 9;
const VIEWPORT_WIDTH = 1080;
const VIEWPORT_HEIGHT = 1920;
const BASE_ZOOM = 1.12;

export const ClyptViralShort: React.FC<ClyptViralShortProps> = ({
  clipStartMs,
  clipEndMs,
  videoSrc,
  tracking,
  videoLetterbox = null,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const sortedTracking = useMemo(
    () =>
      tracking
        .filter((t) => Number.isFinite(t.time_ms))
        .sort((a, b) => a.time_ms - b.time_ms),
    [tracking],
  );

  const currentMs = clipStartMs + (frame / fps) * 1000;
  const targetFrame = findClosestFrame(sortedTracking, currentMs);

  const targetX = clamp(targetFrame?.center_x ?? 0.5, 0, 1);
  const targetY = clamp(targetFrame?.center_y ?? 0.5, 0, 1);

  const letterboxZoom = useMemo(() => {
    if (!videoLetterbox) return 1;
    const w = clamp(Number(videoLetterbox.width) || 1, 0.05, 1);
    const h = clamp(Number(videoLetterbox.height) || 1, 0.05, 1);
    return clamp(1 / Math.min(w, h), 1, 1.9);
  }, [videoLetterbox]);

  const finalZoom = BASE_ZOOM * letterboxZoom;

  const baseCoverWidth = VIEWPORT_HEIGHT * SOURCE_ASPECT;
  const scaledWidth = baseCoverWidth * finalZoom;
  const scaledHeight = VIEWPORT_HEIGHT * finalZoom;
  const maxShiftX = Math.max(0, (scaledWidth - VIEWPORT_WIDTH) / 2);
  const maxShiftY = Math.max(0, (scaledHeight - VIEWPORT_HEIGHT) / 2);

  // Direct, per-frame camera translation from tracking coordinates.
  const translateX = (0.5 - targetX) * 2 * maxShiftX;
  const translateY = (0.5 - targetY) * 2 * maxShiftY;

  const startFromFrame = Math.round((clipStartMs / 1000) * fps);
  const endAtFrame = Math.round((clipEndMs / 1000) * fps);

  return (
    <AbsoluteFill style={{ backgroundColor: "black", overflow: "hidden" }}>
      <OffthreadVideo
        src={videoSrc}
        trimBefore={startFromFrame}
        trimAfter={endAtFrame}
        style={{
          width: "100%",
          height: "100%",
          objectFit: "cover",
          objectPosition: "50% 50%",
          transform: `translate(${translateX}px, ${translateY}px) scale(${finalZoom})`,
          transformOrigin: "50% 50%",
        }}
      />
    </AbsoluteFill>
  );
};
