const { Storage } = require("@google-cloud/storage");
const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");

const ARRAY_PATH = path.resolve(__dirname, "../src/remotion_payloads_array.json");
const SINGLE_PATH = path.resolve(__dirname, "../src/remotion_payload.json");
const OUTPUT_PATH = path.resolve(__dirname, "../public/merged_tracking.json");
const VIDEO_PATH = path.resolve(__dirname, "../public/video.mp4");

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

function normalizeFrame(frame) {
  const time = Number(frame.time_ms ?? frame.timeMs);
  const cx = Number(frame.center_x ?? frame.centerX);
  const cy = Number(frame.center_y ?? frame.centerY);
  if (!Number.isFinite(time) || !Number.isFinite(cx) || !Number.isFinite(cy)) {
    return null;
  }

  return {
    time_ms: time,
    center_x: cx,
    center_y: cy,
    source: frame.source || "track",
    speaker_tag: frame.speaker_tag ?? null,
    track_id: frame.track_id ?? null,
    face_track_id: frame.face_track_id ?? null,
    bbox_w: Number.isFinite(Number(frame.bbox_w)) ? Number(frame.bbox_w) : undefined,
    bbox_h: Number.isFinite(Number(frame.bbox_h)) ? Number(frame.bbox_h) : undefined,
    confidence: Number.isFinite(Number(frame.confidence)) ? Number(frame.confidence) : null,
  };
}

function extractLegacyFrames(trackingData) {
  const frames = [];

  for (const [faceIdx, detection] of (trackingData.face_detections || []).entries()) {
    const trackId = detection.face_track_index ?? detection.track_index ?? faceIdx;
    for (const tsObj of detection.timestamped_objects || []) {
      const bbox = tsObj.bounding_box || tsObj.normalized_bounding_box;
      if (!bbox) continue;
      frames.push({
        time_ms: tsObj.time_ms || tsObj.time_offset_ms,
        center_x: (bbox.left + bbox.right) / 2,
        center_y: (bbox.top + bbox.bottom) / 2,
        source: "face",
        speaker_tag: detection.speaker_tag ?? null,
        track_id: trackId,
        face_track_id: trackId,
        bbox_w: bbox.right - bbox.left,
        bbox_h: bbox.bottom - bbox.top,
        confidence: detection.confidence ?? null,
      });
    }
  }

  for (const [personIdx, detection] of (trackingData.person_detections || []).entries()) {
    const trackId = detection.person_track_index ?? personIdx;
    for (const tsObj of detection.timestamped_objects || []) {
      const bbox = tsObj.bounding_box || tsObj.normalized_bounding_box;
      if (!bbox) continue;
      frames.push({
        time_ms: tsObj.time_ms || tsObj.time_offset_ms,
        center_x: (bbox.left + bbox.right) / 2,
        center_y: (bbox.top + bbox.bottom) / 2,
        source: "person",
        speaker_tag: null,
        track_id: trackId,
        face_track_id: null,
        bbox_w: bbox.right - bbox.left,
        bbox_h: bbox.bottom - bbox.top,
        confidence: detection.confidence ?? null,
      });
    }
  }

  return frames;
}

function applyLetterbox(frames, letterbox) {
  if (!letterbox) return frames;

  const lbLeft = clamp(Number(letterbox.left) || 0, 0, 1);
  const lbTop = clamp(Number(letterbox.top) || 0, 0, 1);
  const lbWidth = Math.max(0.05, clamp(Number(letterbox.width) || 1, 0, 1));
  const lbHeight = Math.max(0.05, clamp(Number(letterbox.height) || 1, 0, 1));

  for (const f of frames) {
    f.center_x = clamp((Number(f.center_x) - lbLeft) / lbWidth, 0, 1);
    f.center_y = clamp((Number(f.center_y) - lbTop) / lbHeight, 0, 1);
    if (Number.isFinite(Number(f.bbox_w))) {
      f.bbox_w = clamp(Number(f.bbox_w) / lbWidth, 0.01, 1);
    }
    if (Number.isFinite(Number(f.bbox_h))) {
      f.bbox_h = clamp(Number(f.bbox_h) / lbHeight, 0.01, 1);
    }
  }

  return frames;
}

function parseTrackingBundle(trackingData, letterbox = null) {
  let frames = [];

  if (Array.isArray(trackingData.frames)) {
    frames = trackingData.frames
      .map(normalizeFrame)
      .filter(Boolean);
  } else {
    frames = extractLegacyFrames(trackingData)
      .map(normalizeFrame)
      .filter(Boolean);
  }

  frames.sort((a, b) => a.time_ms - b.time_ms);
  applyLetterbox(frames, letterbox);

  return { frames };
}

async function downloadUri(storage, uri, letterbox = null) {
  const match = uri.match(/^gs:\/\/([^/]+)\/(.+)$/);
  if (!match) {
    console.warn(`Skipping invalid GCS URI: ${uri}`);
    return { frames: [] };
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
      "No Remotion payload file found. Run backend/pipeline/phase_5_auto_curate.py first " +
      "to generate backend/outputs/remotion_payloads_array.json.",
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

    for (const uri of trackingUris) {
      const bundle = await downloadUri(storage, uri, videoLetterbox);
      clipFrames.push(...bundle.frames);
    }

    clipFrames.sort((a, b) => a.time_ms - b.time_ms);

    result[String(i)] = {
      frames: clipFrames,
      video_letterbox: videoLetterbox,
    };

    console.log(`Clip ${i + 1}: ${clipFrames.length} tracking frames`);
  }

  fs.writeFileSync(OUTPUT_PATH, JSON.stringify(result, null, 2));
  console.log(`Merged tracking for ${payloads.length} clip(s) -> ${OUTPUT_PATH}`);
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
