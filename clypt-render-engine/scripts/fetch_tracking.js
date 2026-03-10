const { Storage } = require("@google-cloud/storage");
const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");

const ARRAY_PATH = path.resolve(__dirname, "../src/remotion_payloads_array.json");
const SINGLE_PATH = path.resolve(__dirname, "../src/remotion_payload.json");
const OUTPUT_PATH = path.resolve(__dirname, "../public/merged_tracking.json");
const VIDEO_PATH = path.resolve(__dirname, "../public/video.mp4");
const SPEAKER_GAP_MERGE_MS = 220;
const CAMERA_FPS = 24;
const CAMERA_WINDOW_MS = 140;
const CAMERA_SPEAKER_LOCK_MS = 1500;
const CAMERA_MIN_FACE_AREA = 0.01;
const CAMERA_DUAL_SEPARATION_X = 0.2;
const CAMERA_MAX_STEP_X = 0.015;
const CAMERA_MAX_STEP_Y = 0.011;
const CAMERA_MAX_STEP_Z = 0.009;
const PERSON_TO_FACE_TIME_WINDOW_MS = 120;
const PERSON_TO_FACE_MAX_DIST = 0.12;
const MIN_PERSON_AREA = 0.02;
const SYNTH_PERSON_SCALE_X = 3.0;
const SYNTH_PERSON_SCALE_Y = 4.2;
const SYNTH_PERSON_MIN_W = 0.16;
const SYNTH_PERSON_MIN_H = 0.28;

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

function detectVideoLetterbox(videoPath) {
  if (!fs.existsSync(videoPath)) return null;

  try {
    const dimsRaw = execSync(
      `ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0:s=x "${videoPath}"`,
      { encoding: "utf-8", maxBuffer: 10 * 1024 * 1024 },
    ).trim();
    const [srcWRaw, srcHRaw] = dimsRaw.split("x");
    const srcW = Number(srcWRaw);
    const srcH = Number(srcHRaw);
    if (!Number.isFinite(srcW) || !Number.isFinite(srcH) || srcW <= 0 || srcH <= 0) {
      return null;
    }

    const cropLog = execSync(
      `ffmpeg -v info -i "${videoPath}" -vf "cropdetect=24:16:0" -frames:v 180 -f null - 2>&1`,
      { encoding: "utf-8", maxBuffer: 20 * 1024 * 1024 },
    );
    const matches = [...cropLog.matchAll(/crop=(\d+):(\d+):(\d+):(\d+)/g)];
    if (matches.length === 0) return null;

    const counts = new Map();
    for (const m of matches) {
      const key = `${m[1]}:${m[2]}:${m[3]}:${m[4]}`;
      counts.set(key, (counts.get(key) || 0) + 1);
    }

    let bestKey = null;
    let bestCount = -1;
    for (const [key, count] of counts.entries()) {
      if (count > bestCount) {
        bestKey = key;
        bestCount = count;
      }
    }
    if (!bestKey) return null;

    const [cropWRaw, cropHRaw, cropXRaw, cropYRaw] = bestKey.split(":");
    const cropW = Number(cropWRaw);
    const cropH = Number(cropHRaw);
    const cropX = Number(cropXRaw);
    const cropY = Number(cropYRaw);
    if (
      !Number.isFinite(cropW) || !Number.isFinite(cropH) ||
      !Number.isFinite(cropX) || !Number.isFinite(cropY)
    ) {
      return null;
    }

    const left = clamp(cropX / srcW, 0, 1);
    const top = clamp(cropY / srcH, 0, 1);
    const width = clamp(cropW / srcW, 0, 1);
    const height = clamp(cropH / srcH, 0, 1);
    const likelyLetterbox =
      (width >= 0.95 && height <= 0.95) ||
      (height >= 0.95 && width <= 0.95);

    if (!likelyLetterbox) return null;
    return { left, top, width, height };
  } catch {
    return null;
  }
}

function normalizeSpeakerTimeline(segments) {
  const cleaned = (segments || [])
    .map((seg) => ({
      start_ms: Number(seg.start_ms),
      end_ms: Number(seg.end_ms),
      speaker_tag: seg.speaker_tag == null ? null : String(seg.speaker_tag),
      track_id: seg.track_id ?? null,
      confidence: seg.confidence == null ? undefined : Number(seg.confidence),
    }))
    .filter(
      (seg) =>
        Number.isFinite(seg.start_ms) &&
        Number.isFinite(seg.end_ms) &&
        seg.end_ms > seg.start_ms,
    )
    .sort((a, b) => a.start_ms - b.start_ms);

  const merged = [];
  for (const seg of cleaned) {
    if (merged.length === 0) {
      merged.push(seg);
      continue;
    }
    const prev = merged[merged.length - 1];
    const sameSpeaker = prev.speaker_tag === seg.speaker_tag;
    const sameTrack = (prev.track_id ?? null) === (seg.track_id ?? null);
    const smallGap = seg.start_ms <= prev.end_ms + SPEAKER_GAP_MERGE_MS;
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

function getActiveSegment(timeline, timeMs) {
  for (const seg of timeline) {
    if (timeMs >= seg.start_ms && timeMs < seg.end_ms) {
      return seg;
    }
  }
  return null;
}

function getTracksInWindow(tracking, targetMs, windowMs) {
  if (!tracking || tracking.length === 0) return [];
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

  const out = [];
  for (let i = lo; i < tracking.length; i++) {
    const t = tracking[i].time_ms;
    if (t > hiTarget) break;
    if (t >= loTarget) out.push(tracking[i]);
  }
  return out;
}

function frameArea(f) {
  return Math.max(0, Number(f.bbox_w) || 0) * Math.max(0, Number(f.bbox_h) || 0);
}

function pickStableTarget(candidates, previousTrackId, preferredPos) {
  if (!candidates || candidates.length === 0) return null;
  if (previousTrackId != null) {
    const sameTrack = candidates.find((c) => c.track_id === previousTrackId);
    if (sameTrack) return sameTrack;
  }

  let best = candidates[0];
  let bestScore = Number.POSITIVE_INFINITY;
  for (const c of candidates) {
    const continuity =
      Math.abs((Number(c.center_x) || 0.5) - preferredPos.x) +
      Math.abs((Number(c.center_y) || 0.5) - preferredPos.y);
    const areaBonus = frameArea(c) * 2.3;
    const confBonus = (Number(c.confidence) || 0) * 0.04;
    const score = continuity - areaBonus - confBonus;
    if (score < bestScore) {
      best = c;
      bestScore = score;
    }
  }
  return best;
}

function buildGlobalCameraPath({
  frames,
  clipStartMs,
  clipEndMs,
  speakerTimeline,
}) {
  const durationMs = Math.max(1, clipEndMs - clipStartMs);
  const totalFrames = Math.max(1, Math.ceil((durationMs / 1000) * CAMERA_FPS));
  const stable = (frames || [])
    .filter((f) => Number.isFinite(f.time_ms))
    .sort((a, b) => a.time_ms - b.time_ms);
  const faceFrames = stable.filter((f) => f.source === "face" && frameArea(f) >= CAMERA_MIN_FACE_AREA);
  const bodyFrames = stable.filter((f) => f.source === "person" || f.source === "person_synth");
  const speakerFrames = stable.filter((f) => f.speaker_tag != null);
  const timeline = normalizeSpeakerTimeline(speakerTimeline || []);

  const path = [];
  let smoothX = 0.5;
  let smoothY = 0.5;
  let smoothZoom = 1.06;
  let previousTrackId = null;
  let lockedSpeaker = null;
  let lockUntil = -1;

  for (let i = 0; i < totalFrames; i++) {
    const ms = clipStartMs + (i / CAMERA_FPS) * 1000;
    const active = getActiveSegment(timeline, ms);
    if (active && active.speaker_tag != null) {
      lockedSpeaker = String(active.speaker_tag);
      lockUntil = ms + CAMERA_SPEAKER_LOCK_MS;
    }
    const targetSpeaker = active?.speaker_tag != null
      ? String(active.speaker_tag)
      : (lockedSpeaker && ms <= lockUntil ? lockedSpeaker : null);

    const nearFaces = getTracksInWindow(faceFrames, ms, CAMERA_WINDOW_MS)
      .sort((a, b) => frameArea(b) - frameArea(a));
    const nearBodies = getTracksInWindow(bodyFrames, ms, CAMERA_WINDOW_MS);
    const nearSpeakerFrames = getTracksInWindow(speakerFrames, ms, CAMERA_WINDOW_MS);

    let mode = "wide-safe";
    let desiredX = smoothX;
    let desiredY = smoothY;
    let desiredZoom = smoothZoom;
    let chosen = null;

    if (targetSpeaker) {
      const speakerCandidates = nearSpeakerFrames.filter(
        (f) => String(f.speaker_tag) === targetSpeaker,
      );
      if (speakerCandidates.length > 0) {
        chosen = pickStableTarget(speakerCandidates, previousTrackId, { x: smoothX, y: smoothY });
        mode = "single";
      }
    }

    if (!chosen && nearFaces.length >= 2) {
      const a = nearFaces[0];
      const b = nearFaces[1];
      const sep = Math.abs((Number(a.center_x) || 0.5) - (Number(b.center_x) || 0.5));
      if (sep >= CAMERA_DUAL_SEPARATION_X) {
        mode = "dual-balanced";
        desiredX = ((Number(a.center_x) || 0.5) + (Number(b.center_x) || 0.5)) / 2;
        desiredY = (((Number(a.center_y) || 0.5) + (Number(b.center_y) || 0.5)) / 2) - 0.02;
        desiredZoom = 1.0;
        previousTrackId = null;
      }
    }

    if (!chosen && mode !== "dual-balanced") {
      const fallbackPool = nearFaces.length > 0 ? nearFaces : nearBodies;
      chosen = pickStableTarget(fallbackPool, previousTrackId, { x: smoothX, y: smoothY });
    }

    if (chosen) {
      desiredX = Number(chosen.center_x) || 0.5;
      desiredY = Number(chosen.center_y) || 0.5;
      const src = chosen.source || "unknown";
      if (mode === "single") {
        desiredY = clamp(desiredY - (src === "face" ? 0.06 : 0.09), 0, 1);
        desiredZoom = src === "face" ? 1.09 : 1.05;
      } else if (mode === "wide-safe") {
        desiredZoom = 1.01;
      }
      previousTrackId = chosen.track_id ?? null;
    }

    const alpha = mode === "single" ? 0.18 : 0.12;
    const stepX = clamp((desiredX - smoothX) * alpha, -CAMERA_MAX_STEP_X, CAMERA_MAX_STEP_X);
    const stepY = clamp((desiredY - smoothY) * alpha, -CAMERA_MAX_STEP_Y, CAMERA_MAX_STEP_Y);
    const stepZ = clamp((desiredZoom - smoothZoom) * 0.16, -CAMERA_MAX_STEP_Z, CAMERA_MAX_STEP_Z);
    smoothX += stepX;
    smoothY += stepY;
    smoothZoom += stepZ;

    path.push({
      time_ms: Math.round(ms),
      x: clamp(smoothX, 0.02, 0.98),
      y: clamp(smoothY, 0.1, 0.9),
      zoom: clamp(smoothZoom, 1.0, 1.3),
      mode,
      target_speaker: targetSpeaker,
      target_track_id: previousTrackId,
    });
  }

  return path;
}

function parseTrackingBundle(trackingData, letterbox = null) {
  const frames = [];
  const personFrames = [];
  const speakerWordTimeline = normalizeSpeakerTimeline(
    trackingData.speaker_word_timeline || [],
  );
  const asdSpeakerTimeline = normalizeSpeakerTimeline(
    trackingData.asd_active_speaker_timeline || [],
  );
  const faceDetections = trackingData.face_detections || [];
  const personDetections = trackingData.person_detections || [];

  for (const [faceIdx, detection] of faceDetections.entries()) {
    const speakerTag = detection.speaker_tag || null;
    const trackId =
      detection.face_track_index ??
      detection.track_index ??
      faceIdx;
    for (const tsObj of detection.timestamped_objects || []) {
      const bbox = tsObj.bounding_box || tsObj.normalized_bounding_box;
      const width = bbox ? bbox.right - bbox.left : 0;
      const height = bbox ? bbox.bottom - bbox.top : 0;
      frames.push({
        time_ms: tsObj.time_ms || tsObj.time_offset_ms,
        center_x: bbox ? (bbox.left + bbox.right) / 2 : 0.5,
        center_y: bbox ? (bbox.top + bbox.bottom) / 2 : 0.5,
        bbox_w: width,
        bbox_h: height,
        source: "face",
        speaker_tag: speakerTag,
        track_id: trackId,
        face_track_id: trackId,
        confidence: detection.confidence ?? null,
      });
    }
  }

  for (const [personIdx, detection] of personDetections.entries()) {
    for (const tsObj of detection.timestamped_objects || []) {
      const bbox = tsObj.bounding_box || tsObj.normalized_bounding_box;
      const width = bbox ? bbox.right - bbox.left : 0;
      const height = bbox ? bbox.bottom - bbox.top : 0;
      if (width * height < MIN_PERSON_AREA) {
        continue;
      }
      personFrames.push({
        time_ms: tsObj.time_ms || tsObj.time_offset_ms,
        center_x: bbox ? (bbox.left + bbox.right) / 2 : 0.5,
        center_y: bbox ? (bbox.top + bbox.bottom) / 2 : 0.5,
        bbox_w: width,
        bbox_h: height,
        source: "person",
        speaker_tag: null,
        track_id: detection.person_track_index ?? personIdx,
        face_track_id: null,
        confidence: detection.confidence ?? null,
      });
    }
  }

  const faceFrames = frames.filter((f) => f.source === "face");

  // Assign speaker tags + face-track links to person detections via nearest face.
  for (const p of personFrames) {
    let best = null;
    let bestDist = Number.POSITIVE_INFINITY;
    for (const f of faceFrames) {
      if (Math.abs(f.time_ms - p.time_ms) > PERSON_TO_FACE_TIME_WINDOW_MS) continue;
      const dist = Math.hypot(f.center_x - p.center_x, f.center_y - p.center_y);
      if (dist < bestDist) {
        best = f;
        bestDist = dist;
      }
    }
    if (best && bestDist <= PERSON_TO_FACE_MAX_DIST) {
      p.face_track_id = best.track_id ?? null;
      p.speaker_tag = best.speaker_tag;
    }
    frames.push(p);
  }

  // If person tracks are unavailable, synthesize torso-like boxes from faces.
  if (personFrames.length === 0) {
    for (const f of faceFrames) {
      const fw = Math.max(0.02, Number(f.bbox_w) || 0.08);
      const fh = Math.max(0.02, Number(f.bbox_h) || 0.08);
      const synthW = clamp(Math.max(SYNTH_PERSON_MIN_W, fw * SYNTH_PERSON_SCALE_X), 0.16, 0.72);
      const synthH = clamp(Math.max(SYNTH_PERSON_MIN_H, fh * SYNTH_PERSON_SCALE_Y), 0.28, 0.9);
      const cx = clamp(Number(f.center_x) || 0.5, synthW / 2, 1 - synthW / 2);
      const cy = clamp((Number(f.center_y) || 0.5) + fh * 0.85, synthH / 2, 1 - synthH / 2);
      frames.push({
        time_ms: f.time_ms,
        center_x: cx,
        center_y: cy,
        bbox_w: synthW,
        bbox_h: synthH,
        source: "person_synth",
        speaker_tag: f.speaker_tag ?? null,
        track_id: f.track_id ?? null,
        face_track_id: f.track_id ?? null,
        confidence: f.confidence ?? null,
      });
    }
  }

  if (letterbox) {
    const lbLeft = clamp(Number(letterbox.left) || 0, 0, 1);
    const lbTop = clamp(Number(letterbox.top) || 0, 0, 1);
    const lbWidth = Math.max(0.05, clamp(Number(letterbox.width) || 1, 0, 1));
    const lbHeight = Math.max(0.05, clamp(Number(letterbox.height) || 1, 0, 1));

    for (const f of frames) {
      const x = clamp((Number(f.center_x) - lbLeft) / lbWidth, 0, 1);
      const y = clamp((Number(f.center_y) - lbTop) / lbHeight, 0, 1);
      const w = clamp((Number(f.bbox_w) || 0) / lbWidth, 0.01, 1);
      const h = clamp((Number(f.bbox_h) || 0) / lbHeight, 0.01, 1);
      f.center_x = x;
      f.center_y = y;
      f.bbox_w = w;
      f.bbox_h = h;
    }
  }

  return {
    frames,
    speaker_word_timeline: speakerWordTimeline,
    asd_active_speaker_timeline: asdSpeakerTimeline,
  };
}

async function downloadUri(storage, uri, letterbox = null) {
  const match = uri.match(/^gs:\/\/([^/]+)\/(.+)$/);
  if (!match) {
    console.warn(`Skipping invalid GCS URI: ${uri}`);
    return { frames: [], speaker_word_timeline: [], asd_active_speaker_timeline: [] };
  }
  const [, bucketName, objectPath] = match;
  console.log(`Downloading: ${uri}`);
  const [contents] = await storage.bucket(bucketName).file(objectPath).download();
  return parseTrackingBundle(JSON.parse(contents.toString("utf-8")), letterbox);
}

async function main() {
  let payloads;
  if (fs.existsSync(ARRAY_PATH)) {
    const raw = JSON.parse(fs.readFileSync(ARRAY_PATH, "utf-8"));
    payloads = Array.isArray(raw) ? raw : [raw];
    console.log(`Loaded array payload with ${payloads.length} clip(s)`);
  } else if (fs.existsSync(SINGLE_PATH)) {
    payloads = [JSON.parse(fs.readFileSync(SINGLE_PATH, "utf-8"))];
    console.log("Loaded single payload (legacy mode)");
  } else {
    throw new Error(
      "No Remotion payload file found. Run phase_4_auto_curate.py first " +
      "to generate outputs/remotion_payloads_array.json.",
    );
  }

  const storage = new Storage();
  const videoLetterbox = detectVideoLetterbox(VIDEO_PATH);
  if (videoLetterbox) {
    console.log(
      `Detected letterbox: left=${videoLetterbox.left.toFixed(4)} ` +
      `top=${videoLetterbox.top.toFixed(4)} ` +
      `width=${videoLetterbox.width.toFixed(4)} ` +
      `height=${videoLetterbox.height.toFixed(4)}`,
    );
  }
  const result = {};

  for (let i = 0; i < payloads.length; i++) {
    const trackingUris = payloads[i].tracking_uris || [];
    const clipFrames = [];
    const clipSpeakerWords = [];
    const clipAsdTimeline = [];

    for (const uri of trackingUris) {
      const bundle = await downloadUri(storage, uri, videoLetterbox);
      clipFrames.push(...bundle.frames);
      clipSpeakerWords.push(...bundle.speaker_word_timeline);
      clipAsdTimeline.push(...bundle.asd_active_speaker_timeline);
    }

    clipFrames.sort((a, b) => a.time_ms - b.time_ms);
    const mergedSpeakerWords = normalizeSpeakerTimeline(clipSpeakerWords);
    const mergedAsdTimeline = normalizeSpeakerTimeline(clipAsdTimeline);
    const payloadSpeakerTimeline = normalizeSpeakerTimeline(payloads[i].active_speaker_timeline || []);
    const effectiveSpeakerTimeline = mergedAsdTimeline.length > 0
      ? mergedAsdTimeline
      : (payloadSpeakerTimeline.length > 0 ? payloadSpeakerTimeline : mergedSpeakerWords);
    const clipStartMs = Number(payloads[i].clip_start_ms) || 0;
    const clipEndMs = Number(payloads[i].clip_end_ms) || clipStartMs + 1000;
    const cameraPath = buildGlobalCameraPath({
      frames: clipFrames,
      clipStartMs,
      clipEndMs,
      speakerTimeline: effectiveSpeakerTimeline,
    });

    result[String(i)] = {
      frames: clipFrames,
      speaker_word_timeline: mergedSpeakerWords,
      asd_active_speaker_timeline: mergedAsdTimeline,
      fused_active_speaker_timeline: effectiveSpeakerTimeline,
      camera_path: cameraPath,
      video_letterbox: videoLetterbox,
    };
    console.log(
      `Clip ${i + 1}: ${clipFrames.length} tracking frames, ` +
      `${mergedSpeakerWords.length} word-level speaker segments, ` +
      `${mergedAsdTimeline.length} ASD segments, ` +
      `${cameraPath.length} camera keyframes`,
    );
  }

  fs.writeFileSync(OUTPUT_PATH, JSON.stringify(result, null, 2));
  console.log(`Merged tracking for ${payloads.length} clip(s) → ${OUTPUT_PATH}`);
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
