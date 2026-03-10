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
  source: string;
  speaker_tag?: string | null;
  track_id?: string | number | null;
  face_track_id?: string | number | null;
  bbox_w?: number;
  bbox_h?: number;
  confidence?: number | null;
}

export interface SpeakerSegment {
  start_ms: number;
  end_ms: number;
  speaker_tag?: string | null;
  track_id?: string | number | null;
  confidence?: number;
}

export interface CameraKeyframe {
  time_ms: number;
  x: number;
  y: number;
  zoom: number;
  mode?: string;
  target_speaker?: string | null;
  target_track_id?: string | number | null;
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
  speakerTimeline?: SpeakerSegment[];
  wordSpeakerTimeline?: SpeakerSegment[];
  asdSpeakerTimeline?: SpeakerSegment[];
  fusedSpeakerTimeline?: SpeakerSegment[];
  cameraPath?: CameraKeyframe[];
  videoLetterbox?: VideoLetterbox | null;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/** Find the active timeline segment at a given timestamp. */
function getActiveSegment(
  timeline: SpeakerSegment[],
  timeMs: number,
): SpeakerSegment | null {
  for (const seg of timeline) {
    if (timeMs >= seg.start_ms && timeMs < seg.end_ms) {
      return seg;
    }
  }
  return null;
}

/** Return only frames for active speaker. No fallback to all. */
function filterBySpeaker(
  tracking: TrackingFrame[],
  speakerTag: string | null,
): TrackingFrame[] {
  if (!speakerTag) return tracking;
  return tracking.filter((f) => f.speaker_tag === speakerTag);
}

function filterByTrack(
  tracking: TrackingFrame[],
  trackId: string | number | null,
): TrackingFrame[] {
  if (trackId == null) return tracking;
  const wanted = String(trackId);
  return tracking.filter(
    (f) =>
      (f.track_id != null && String(f.track_id) === wanted) ||
      (f.face_track_id != null && String(f.face_track_id) === wanted),
  );
}

interface TargetPoint {
  x: number;
  y: number;
  trackId: string | number | null;
  timeMs: number;
  bboxW: number;
  bboxH: number;
  confidence: number;
  source: string;
}

const MIN_FACE_AREA = 0.008;
const MIN_FACE_HEIGHT = 0.12;
const AREA_BONUS = 2.5;
const CONFIDENCE_BONUS = 0.04;

/**
 * Pick the most stable candidate near targetMs.
 * If multiple detections share the same timestamp, prefer previous track,
 * otherwise pick by continuity + size + confidence.
 */
function findClosestTrack(
  tracking: TrackingFrame[],
  targetMs: number,
  preferredPos: { x: number; y: number },
  previousTrackId: string | number | null,
): TargetPoint {
  if (tracking.length === 0) {
    return {
      x: 0.5,
      y: 0.5,
      trackId: null,
      timeMs: targetMs,
      bboxW: 0.08,
      bboxH: 0.08,
      confidence: 0,
      source: "none",
    };
  }

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

  const pickAtIndex = (idx: number): TargetPoint => {
    const f = tracking[idx];
    return {
      x: f.center_x,
      y: f.center_y,
      trackId: f.track_id ?? null,
      timeMs: f.time_ms,
      bboxW: f.bbox_w ?? 0.08,
      bboxH: f.bbox_h ?? 0.08,
      confidence: f.confidence ?? 0,
      source: f.source ?? "unknown",
    };
  };

  if (lo === 0) return pickAtIndex(0);
  if (lo >= tracking.length) return pickAtIndex(tracking.length - 1);

  const prev = tracking[lo - 1];
  const next = tracking[lo];
  const closer =
    Math.abs(prev.time_ms - targetMs) <= Math.abs(next.time_ms - targetMs)
      ? prev
      : next;

  const nearestIndex = closer === prev ? lo - 1 : lo;
  const nearestTime = closer.time_ms;

  let start = nearestIndex;
  while (start > 0 && tracking[start - 1].time_ms === nearestTime) {
    start--;
  }
  let end = nearestIndex;
  while (end < tracking.length - 1 && tracking[end + 1].time_ms === nearestTime) {
    end++;
  }
  const candidates = tracking.slice(start, end + 1);

  const areaOf = (c: TrackingFrame) => (c.bbox_w ?? 0) * (c.bbox_h ?? 0);
  const filtered = candidates.filter(
    (c) => !(c.source === "face" && areaOf(c) < MIN_FACE_AREA),
  );
  const usable = filtered.length > 0 ? filtered : candidates;

  if (previousTrackId !== null) {
    const sameTrack = usable.find((c) => c.track_id === previousTrackId);
    if (sameTrack) {
      return {
        x: sameTrack.center_x,
        y: sameTrack.center_y,
        trackId: sameTrack.track_id ?? null,
        timeMs: sameTrack.time_ms,
        bboxW: sameTrack.bbox_w ?? 0.08,
        bboxH: sameTrack.bbox_h ?? 0.08,
        confidence: sameTrack.confidence ?? 0,
        source: sameTrack.source ?? "unknown",
      };
    }
  }

  let best = usable[0];
  let bestScore = Number.POSITIVE_INFINITY;

  for (const c of usable) {
    const continuityCost =
      Math.abs(c.center_x - preferredPos.x) +
      Math.abs(c.center_y - preferredPos.y);
    const areaBonus = areaOf(c) * AREA_BONUS;
    const confBonus = (c.confidence ?? 0) * CONFIDENCE_BONUS;
    const score = continuityCost - areaBonus - confBonus;
    if (score < bestScore) {
      best = c;
      bestScore = score;
    }
  }

  return {
    x: best.center_x,
    y: best.center_y,
    trackId: best.track_id ?? null,
    timeMs: best.time_ms,
    bboxW: best.bbox_w ?? 0.08,
    bboxH: best.bbox_h ?? 0.08,
    confidence: best.confidence ?? 0,
    source: best.source ?? "unknown",
  };
}

const BASE_SMOOTHING = 0.14;
const FAST_SMOOTHING = 0.3;
const LARGE_MOVE_THRESHOLD = 0.18;
const MAX_STEP_X_PER_FRAME = 0.016;
const MAX_STEP_Y_PER_FRAME = 0.012;
const MAX_ZOOM_STEP_PER_FRAME = 0.01;
const MAX_SPEAKER_GAP_MS = 300;
const MAX_ANY_SPEAKER_GAP_MS = 450;
const SPEAKER_LOCK_HOLD_MS = 1600;
const SPEAKER_SEGMENT_MERGE_MS = 180;
const MIN_STABLE_TRACK_FRAMES = 3;
const MIN_ASD_CONFIDENCE = 0.62;
const ASD_MIN_SEGMENTS = 3;
const ASD_MIN_COVERAGE = 0.55;
const ASD_MIN_MEAN_CONFIDENCE = 0.74;
const ASD_MIN_TAGGED_RATIO = 0.35;
const ASD_MAX_SWITCH_RATE = 0.52;
const ASD_STRONG_TARGET_CONFIDENCE = 0.76;
const DUAL_WINDOW_MS = 140;
const DUAL_MIN_FACE_AREA = 0.014;
const DUAL_MIN_SEPARATION_X = 0.22;
const MIN_OBJECT_POS_Y = 34;
const MAX_OBJECT_POS_Y = 62;

// Keep object-position away from pathological extremes.
const MIN_OBJECT_POS_X = 6;
const MAX_OBJECT_POS_X = 94;

// For 16:9 source into 9:16 cover, only ~31.64% source width is visible.
const VISIBLE_WIDTH_FRACTION = (9 / 16) / (16 / 9);
const HALF_VISIBLE_WIDTH = VISIBLE_WIDTH_FRACTION / 2;

function normalizeSpeakerTimeline(
  segments: SpeakerSegment[],
): SpeakerSegment[] {
  const cleaned = segments
    .map((s) => ({
      start_ms: Number(s.start_ms),
      end_ms: Number(s.end_ms),
      speaker_tag:
        s.speaker_tag == null || String(s.speaker_tag).trim() === ""
          ? null
          : String(s.speaker_tag),
      track_id: s.track_id ?? null,
      confidence: s.confidence == null ? undefined : Number(s.confidence),
    }))
    .filter(
      (s) =>
        Number.isFinite(s.start_ms) &&
        Number.isFinite(s.end_ms) &&
        s.end_ms > s.start_ms,
    )
    .sort((a, b) => a.start_ms - b.start_ms);

  const merged: SpeakerSegment[] = [];
  for (const seg of cleaned) {
    if (merged.length === 0) {
      merged.push(seg);
      continue;
    }
    const prev = merged[merged.length - 1];
    const sameSpeaker = prev.speaker_tag === seg.speaker_tag;
    const sameTrack = (prev.track_id ?? null) === (seg.track_id ?? null);
    const smallGap = seg.start_ms <= prev.end_ms + SPEAKER_SEGMENT_MERGE_MS;
    if (sameSpeaker && sameTrack && smallGap) {
      prev.end_ms = Math.max(prev.end_ms, seg.end_ms);
      if (
        typeof prev.confidence === "number" &&
        typeof seg.confidence === "number"
      ) {
        prev.confidence = Math.max(prev.confidence, seg.confidence);
      } else if (typeof seg.confidence === "number") {
        prev.confidence = seg.confidence;
      }
    } else {
      merged.push(seg);
    }
  }
  return merged;
}

function getTracksInWindow(
  tracking: TrackingFrame[],
  targetMs: number,
  windowMs: number,
): TrackingFrame[] {
  if (tracking.length === 0) return [];

  const loTarget = targetMs - windowMs;
  const hiTarget = targetMs + windowMs;

  let lo = 0;
  let hi = tracking.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (tracking[mid].time_ms < loTarget) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  const out: TrackingFrame[] = [];
  for (let i = lo; i < tracking.length; i++) {
    const t = tracking[i].time_ms;
    if (t > hiTarget) break;
    if (t >= loTarget) out.push(tracking[i]);
  }
  return out;
}

export const ClyptViralShort: React.FC<ClyptViralShortProps> = ({
  clipStartMs,
  clipEndMs,
  videoSrc,
  tracking,
  speakerTimeline = [],
  wordSpeakerTimeline = [],
  asdSpeakerTimeline = [],
  fusedSpeakerTimeline = [],
  cameraPath: precomputedCameraPath = [],
  videoLetterbox = null,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const durationMs = Math.max(1, clipEndMs - clipStartMs);

  const normalizedAsdTimeline = useMemo(
    () =>
      normalizeSpeakerTimeline(asdSpeakerTimeline).filter(
        (s) =>
          (s.track_id != null || s.speaker_tag != null) &&
          (typeof s.confidence !== "number" || s.confidence >= MIN_ASD_CONFIDENCE),
      ),
    [asdSpeakerTimeline],
  );

  const normalizedFallbackTimeline = useMemo(
    () =>
      normalizeSpeakerTimeline(
        fusedSpeakerTimeline.length > 0
          ? fusedSpeakerTimeline
          : (wordSpeakerTimeline.length > 0 ? wordSpeakerTimeline : speakerTimeline),
    ),
    [fusedSpeakerTimeline, wordSpeakerTimeline, speakerTimeline],
  );

  const asdStats = useMemo(() => {
    if (durationMs <= 0 || normalizedAsdTimeline.length === 0) {
      return {
        coverage: 0,
        meanConfidence: 0,
        taggedRatio: 0,
        switchRate: 0,
      };
    }
    const covered = normalizedAsdTimeline.reduce(
      (sum, s) => sum + Math.max(0, s.end_ms - s.start_ms),
      0,
    );
    const confidenceValues = normalizedAsdTimeline
      .map((s) => s.confidence)
      .filter((c): c is number => typeof c === "number");
    const meanConfidence = confidenceValues.length > 0
      ? confidenceValues.reduce((sum, c) => sum + c, 0) / confidenceValues.length
      : 0;
    const tagged = normalizedAsdTimeline.filter((s) => s.speaker_tag != null).length;

    let switches = 0;
    for (let i = 1; i < normalizedAsdTimeline.length; i++) {
      const prevKey = normalizedAsdTimeline[i - 1].track_id ?? normalizedAsdTimeline[i - 1].speaker_tag ?? null;
      const currKey = normalizedAsdTimeline[i].track_id ?? normalizedAsdTimeline[i].speaker_tag ?? null;
      if (prevKey !== currKey) switches++;
    }

    return {
      coverage: covered / durationMs,
      meanConfidence,
      taggedRatio: tagged / normalizedAsdTimeline.length,
      switchRate: switches / Math.max(1, normalizedAsdTimeline.length - 1),
    };
  }, [durationMs, normalizedAsdTimeline]);

  const useAsdTimeline = (
    normalizedAsdTimeline.length >= ASD_MIN_SEGMENTS &&
    asdStats.coverage >= ASD_MIN_COVERAGE &&
    asdStats.meanConfidence >= ASD_MIN_MEAN_CONFIDENCE &&
    asdStats.taggedRatio >= ASD_MIN_TAGGED_RATIO &&
    asdStats.switchRate <= ASD_MAX_SWITCH_RATE
  );

  const fallbackUniqueSpeakerCount = useMemo(
    () =>
      new Set(
        normalizedFallbackTimeline
          .map((s) => s.speaker_tag)
          .filter((s): s is string => Boolean(s)),
      ).size,
    [normalizedFallbackTimeline],
  );
  const useFallbackSpeakerTimeline = fallbackUniqueSpeakerCount >= 2;

  const normalizedTimeline = useMemo(
    () => (
      useAsdTimeline
        ? normalizedAsdTimeline
        : (useFallbackSpeakerTimeline ? normalizedFallbackTimeline : [])
    ),
    [useAsdTimeline, useFallbackSpeakerTimeline, normalizedAsdTimeline, normalizedFallbackTimeline],
  );

  const cleanedTracking = useMemo(
    () =>
      tracking
        .filter((f) => Number.isFinite(f.time_ms))
        .filter((f) => {
          if (f.source !== "face") return true;
          const area = (f.bbox_w ?? 0) * (f.bbox_h ?? 0);
          return area >= MIN_FACE_AREA && (f.bbox_h ?? 0) >= MIN_FACE_HEIGHT;
        })
        .sort((a, b) => a.time_ms - b.time_ms),
    [tracking],
  );

  const stableTrackIds = useMemo(() => {
    const counts = new Map<string, number>();
    for (const f of cleanedTracking) {
      if (f.track_id == null) continue;
      const key = String(f.track_id);
      counts.set(key, (counts.get(key) || 0) + 1);
    }
    const stable = new Set<string>();
    for (const [key, count] of counts.entries()) {
      if (count >= MIN_STABLE_TRACK_FRAMES) stable.add(key);
    }
    return stable;
  }, [cleanedTracking]);

  const stableTracking = useMemo(
    () =>
      cleanedTracking.filter(
        (f) => f.track_id == null || stableTrackIds.has(String(f.track_id)),
      ),
    [cleanedTracking, stableTrackIds],
  );

  const speakerTracking = useMemo(
    () => stableTracking.filter((f) => f.speaker_tag != null),
    [stableTracking],
  );
  const faceTracking = useMemo(
    () => stableTracking.filter((f) => f.source === "face"),
    [stableTracking],
  );
  const bodyTracking = useMemo(
    () =>
      stableTracking.filter(
        (f) => f.source === "person" || f.source === "person_synth",
      ),
    [stableTracking],
  );

  const hasSpeakerData = normalizedTimeline.length > 0;
  const uniqueSpeakerCount = useMemo(
    () =>
      new Set(
        normalizedTimeline
          .map((s) => s.speaker_tag)
          .filter((s): s is string => Boolean(s)),
      ).size,
    [normalizedTimeline],
  );
  const useSpeakerTagFiltering = !useAsdTimeline && uniqueSpeakerCount >= 2;
  const totalFrames = Math.max(1, Math.ceil((durationMs / 1000) * fps));
  const normalizedPrecomputedCameraPath = useMemo(
    () =>
      precomputedCameraPath
        .map((p) => ({
          time_ms: Number(p.time_ms),
          x: Number(p.x),
          y: Number(p.y),
          zoom: Number(p.zoom),
        }))
        .filter(
          (p) =>
            Number.isFinite(p.time_ms) &&
            Number.isFinite(p.x) &&
            Number.isFinite(p.y) &&
            Number.isFinite(p.zoom),
        )
        .sort((a, b) => a.time_ms - b.time_ms),
    [precomputedCameraPath],
  );

  const cameraPath = useMemo(() => {
    const points: Array<{ x: number; y: number; boxW: number; zoom: number }> = [];
    let smoothX = 0.5;
    let smoothY = 0.5;
    let smoothBoxW = 0.12;
    let smoothZoom = 1.1;
    let previousTrackId: string | number | null = null;
    let lockedSpeaker: string | null = null;
    let lockedTrackId: string | number | null = null;
    let speakerLockUntilMs = -1;

    for (let f = 0; f < totalFrames; f++) {
      const ms = clipStartMs + (f / fps) * 1000;
      const activeSegment = hasSpeakerData
        ? getActiveSegment(normalizedTimeline, ms)
        : null;

      const activeSpeaker = activeSegment?.speaker_tag ?? null;
      const activeTrackId = activeSegment?.track_id ?? null;

      if (activeSpeaker || activeTrackId != null) {
        if (activeSpeaker) lockedSpeaker = activeSpeaker;
        if (activeTrackId != null) lockedTrackId = activeTrackId;
        speakerLockUntilMs = ms + SPEAKER_LOCK_HOLD_MS;
      }

      const targetSpeaker = activeSpeaker ??
        (
          hasSpeakerData &&
          lockedSpeaker &&
          ms <= speakerLockUntilMs
        ? lockedSpeaker
        : null
        );
      const targetTrackId = activeTrackId ??
        (
          hasSpeakerData &&
          lockedTrackId != null &&
          ms <= speakerLockUntilMs
            ? lockedTrackId
            : null
        );

      const nearFaces = getTracksInWindow(faceTracking, ms, DUAL_WINDOW_MS)
        .filter((c) => (c.bbox_w ?? 0) * (c.bbox_h ?? 0) >= DUAL_MIN_FACE_AREA)
        .sort((a, b) => ((b.bbox_w ?? 0) * (b.bbox_h ?? 0)) - ((a.bbox_w ?? 0) * (a.bbox_h ?? 0)));
      const dualSeparation = nearFaces.length >= 2
        ? Math.abs(nearFaces[0].center_x - nearFaces[1].center_x)
        : 0;
      const dualReady = nearFaces.length >= 2 && dualSeparation >= DUAL_MIN_SEPARATION_X;
      const strongAsdTarget = (
        useAsdTimeline &&
        targetTrackId != null &&
        (activeSegment?.confidence ?? 0) >= ASD_STRONG_TARGET_CONFIDENCE
      );

      let shotMode: "single" | "dual" | "fallback" = "fallback";
      if (strongAsdTarget || (targetSpeaker && useSpeakerTagFiltering)) {
        shotMode = "single";
      } else if (dualReady) {
        shotMode = "dual";
      }

      let candidateTracks = stableTracking;
      if (shotMode === "single") {
        if (targetTrackId != null) {
          const trackBodyFrames = filterByTrack(bodyTracking, targetTrackId);
          const trackFaceFrames = filterByTrack(faceTracking, targetTrackId);
          const trackAllFrames = filterByTrack(stableTracking, targetTrackId);
          if (trackFaceFrames.length > 0) {
            candidateTracks = trackFaceFrames;
          } else if (trackBodyFrames.length > 0) {
            candidateTracks = trackBodyFrames;
          } else if (trackAllFrames.length > 0) {
            candidateTracks = trackAllFrames;
          } else {
            candidateTracks = faceTracking.length > 0
              ? faceTracking
              : (bodyTracking.length > 0 ? bodyTracking : stableTracking);
          }
        } else if (targetSpeaker && useSpeakerTagFiltering) {
          const speakerFaceFrames = filterBySpeaker(faceTracking, targetSpeaker);
          const speakerFrames = filterBySpeaker(speakerTracking, targetSpeaker);
          if (speakerFaceFrames.length > 0) {
            candidateTracks = speakerFaceFrames;
          } else {
            candidateTracks = speakerFrames.length > 0 ? speakerFrames : speakerTracking;
          }
        } else {
          candidateTracks = faceTracking.length > 0
            ? faceTracking
            : (bodyTracking.length > 0 ? bodyTracking : stableTracking);
        }
      } else {
        candidateTracks = faceTracking.length > 0
          ? faceTracking
          : (bodyTracking.length > 0 ? bodyTracking : stableTracking);
      }

      if (candidateTracks.length === 0) {
        points.push({ x: smoothX, y: smoothY, boxW: smoothBoxW, zoom: smoothZoom });
        continue;
      }

      let target = findClosestTrack(
        candidateTracks,
        ms,
        { x: smoothX, y: smoothY },
        previousTrackId,
      );

      if (
        shotMode === "single" &&
        hasSpeakerData &&
        (targetSpeaker || targetTrackId != null) &&
        Math.abs(target.timeMs - ms) > MAX_SPEAKER_GAP_MS &&
        stableTracking.length > 0
      ) {
        const relaxedPool = targetTrackId != null
          ? (
            filterByTrack(stableTracking, targetTrackId).length > 0
              ? filterByTrack(stableTracking, targetTrackId)
              : (
                faceTracking.length > 0
                  ? faceTracking
                  : (bodyTracking.length > 0 ? bodyTracking : stableTracking)
              )
          )
          : (
            useSpeakerTagFiltering && speakerTracking.length > 0
              ? speakerTracking
              : (faceTracking.length > 0 ? faceTracking : stableTracking)
          );
        const relaxed = findClosestTrack(
          relaxedPool,
          ms,
          { x: smoothX, y: smoothY },
          previousTrackId,
        );
        if (Math.abs(relaxed.timeMs - ms) <= MAX_ANY_SPEAKER_GAP_MS) {
          target = relaxed;
        }
      }

      let desiredX = target.x;
      let desiredY = target.y;
      let desiredBoxW = target.bboxW;
      let desiredZoom = target.source === "face" ? 1.08 : 1.05;

      if (shotMode === "single") {
        if (target.source === "face") {
          // Keep more forehead/headroom for profile and leaning poses.
          desiredY = clamp(desiredY - 0.06, 0, 1);
          desiredBoxW = Math.max(desiredBoxW, 0.16);
        } else if (target.source === "person_synth" || target.source === "person") {
          const upBias = Math.max(0.08, target.bboxH * 0.2);
          desiredY = clamp(desiredY - upBias, 0, 1);
          desiredBoxW = Math.max(desiredBoxW, 0.2);
          desiredZoom = 1.05;
        }
      } else if (shotMode === "dual" && nearFaces.length >= 2) {
        const a = nearFaces[0];
        const b = nearFaces[1];
        desiredX = (a.center_x + b.center_x) / 2;
        desiredY = (a.center_y + b.center_y) / 2;
        desiredBoxW = Math.max(dualSeparation + 0.2, 0.36);
        desiredZoom = 1.0;
        previousTrackId = null;
      } else {
        // Fallback wide mode: avoid aggressive single-face lock when speaker
        // evidence is weak.
        if (nearFaces.length >= 2) {
          const a = nearFaces[0];
          const b = nearFaces[1];
          desiredX = (a.center_x + b.center_x) / 2;
          desiredY = (a.center_y + b.center_y) / 2;
          desiredBoxW = Math.max(Math.abs(a.center_x - b.center_x) + 0.2, 0.34);
          desiredZoom = 1.0;
          previousTrackId = null;
        } else {
          desiredBoxW = Math.max(desiredBoxW, 0.24);
          desiredZoom = Math.min(desiredZoom, 1.02);
        }
      }

      const area = Math.max(0.001, desiredBoxW * Math.max(target.bboxH, 0.12));
      const areaDrivenZoom = clamp(1.22 - area * 2.0, 1.04, 1.2);
      desiredZoom = Math.min(desiredZoom, areaDrivenZoom);

      const delta =
        Math.abs(desiredX - smoothX) +
        Math.abs(desiredY - smoothY);
      let alpha = delta > LARGE_MOVE_THRESHOLD
        ? FAST_SMOOTHING
        : BASE_SMOOTHING;
      if (shotMode !== "single") {
        alpha = Math.min(alpha, 0.12);
      }
      const stepScale = shotMode === "single" ? 1 : 0.72;

      const stepX = clamp(
        (desiredX - smoothX) * alpha,
        -MAX_STEP_X_PER_FRAME * stepScale,
        MAX_STEP_X_PER_FRAME * stepScale,
      );
      const stepY = clamp(
        (desiredY - smoothY) * alpha,
        -MAX_STEP_Y_PER_FRAME * stepScale,
        MAX_STEP_Y_PER_FRAME * stepScale,
      );
      const stepZoom = clamp(
        (desiredZoom - smoothZoom) * 0.18,
        -MAX_ZOOM_STEP_PER_FRAME,
        MAX_ZOOM_STEP_PER_FRAME,
      );

      smoothX += stepX;
      smoothY += stepY;
      smoothZoom += stepZoom;
      smoothBoxW += (desiredBoxW - smoothBoxW) * 0.2;
      previousTrackId = target.trackId;

      points.push({ x: smoothX, y: smoothY, boxW: smoothBoxW, zoom: smoothZoom });
    }

    return points;
  }, [
    bodyTracking,
    clipStartMs,
    faceTracking,
    fps,
    hasSpeakerData,
    normalizedTimeline,
    speakerTracking,
    stableTracking,
    totalFrames,
    useAsdTimeline,
    useSpeakerTagFiltering,
  ]);

  const fallbackPoint = cameraPath[Math.min(frame, cameraPath.length - 1)] || {
    x: 0.5,
    y: 0.5,
    boxW: 0.12,
    zoom: 1.06,
  };
  const precomputedPoint = normalizedPrecomputedCameraPath.length > 0
    ? normalizedPrecomputedCameraPath[
      Math.min(frame, normalizedPrecomputedCameraPath.length - 1)
    ]
    : null;
  const point = precomputedPoint
    ? {
      x: clamp(precomputedPoint.x, 0, 1),
      y: clamp(precomputedPoint.y, 0, 1),
      boxW: 0.16,
      zoom: clamp(precomputedPoint.zoom, 1, 1.35),
    }
    : fallbackPoint;

  const barCropZoom = useMemo(() => {
    if (!videoLetterbox) return 1;
    const w = clamp(Number(videoLetterbox.width) || 1, 0.05, 1);
    const h = clamp(Number(videoLetterbox.height) || 1, 0.05, 1);
    const zoom = 1 / Math.min(w, h);
    return clamp(zoom, 1, 1.9);
  }, [videoLetterbox]);
  const finalZoom = Math.max(1.01, point.zoom * barCropZoom);

  const startFromFrame = Math.round((clipStartMs / 1000) * fps);
  const endAtFrame = Math.round((clipEndMs / 1000) * fps);

  // Keep center within visible crop window while allowing meaningful pan.
  const halfVisibleAtZoom = HALF_VISIBLE_WIDTH / Math.max(1.0, finalZoom);
  const edgeMargin = 0.02;
  const safeMinCenter = halfVisibleAtZoom + edgeMargin;
  const safeMaxCenter = 1 - halfVisibleAtZoom - edgeMargin;

  let safeCenterX = safeMinCenter < safeMaxCenter
    ? clamp(point.x, safeMinCenter, safeMaxCenter)
    : 0.5;

  // Add slight inward bias near edges to protect profile head room.
  if (safeCenterX < 0.35) safeCenterX += 0.015;
  if (safeCenterX > 0.65) safeCenterX -= 0.015;
  safeCenterX = clamp(safeCenterX, safeMinCenter, safeMaxCenter);

  // objectFit cover guarantees full 9:16 fill without letterboxing.
  const objectPosX = clamp(safeCenterX * 100, MIN_OBJECT_POS_X, MAX_OBJECT_POS_X);
  const objectPosY = clamp(point.y * 100, MIN_OBJECT_POS_Y, MAX_OBJECT_POS_Y);

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
          objectPosition: `${objectPosX}% ${objectPosY}%`,
          transform: `scale(${finalZoom})`,
          transformOrigin: "50% 50%",
        }}
      />
    </AbsoluteFill>
  );
};
