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

export interface VideoMetadata {
  width: number;
  height: number;
}

export interface ClyptViralShortProps extends Record<string, unknown> {
  clipStartMs: number;
  clipEndMs: number;
  videoSrc: string;
  tracking: TrackingFrame[];
  videoLetterbox?: VideoLetterbox | null;
  videoMetadata?: VideoMetadata | null;
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

const VIEWPORT_WIDTH = 1080;
const VIEWPORT_HEIGHT = 1920;
const VIEWPORT_ASPECT = VIEWPORT_WIDTH / VIEWPORT_HEIGHT;
const BASE_ZOOM = 1.12;

export const ClyptViralShort: React.FC<ClyptViralShortProps> = ({
  clipStartMs,
  clipEndMs,
  videoSrc,
  tracking,
  videoLetterbox = null,
  videoMetadata = null,
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

  const { effectiveAspect, letterboxZoom } = useMemo(() => {
    const sourceWidth = Math.max(1, Number(videoMetadata?.width) || 1920);
    const sourceHeight = Math.max(1, Number(videoMetadata?.height) || 1080);
    const lbWidth = clamp(Number(videoLetterbox?.width) || 1, 0.05, 1);
    const lbHeight = clamp(Number(videoLetterbox?.height) || 1, 0.05, 1);
    const aspect = (sourceWidth * lbWidth) / (sourceHeight * lbHeight);
    const zoom = videoLetterbox
      ? clamp(1 / Math.min(lbWidth, lbHeight), 1, 1.9)
      : 1;

    return {
      effectiveAspect: clamp(aspect, 0.2, 4),
      letterboxZoom: zoom,
    };
  }, [videoLetterbox, videoMetadata]);

  const finalZoom = BASE_ZOOM * letterboxZoom;
  const sourceIsWiderThanViewport = effectiveAspect >= VIEWPORT_ASPECT;
  const baseCoverWidth = sourceIsWiderThanViewport
    ? VIEWPORT_HEIGHT * effectiveAspect
    : VIEWPORT_WIDTH;
  const baseCoverHeight = sourceIsWiderThanViewport
    ? VIEWPORT_HEIGHT
    : VIEWPORT_WIDTH / effectiveAspect;
  const scaledWidth = baseCoverWidth * finalZoom;
  const scaledHeight = baseCoverHeight * finalZoom;
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
