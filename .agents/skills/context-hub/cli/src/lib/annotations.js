import { readFileSync, writeFileSync, mkdirSync, unlinkSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { getChubDir } from './config.js';

function getAnnotationsDir() {
  return join(getChubDir(), 'annotations');
}

function annotationPath(entryId) {
  const safe = entryId.replace(/\//g, '--');
  return join(getAnnotationsDir(), `${safe}.json`);
}

export function readAnnotation(entryId) {
  try {
    return JSON.parse(readFileSync(annotationPath(entryId), 'utf8'));
  } catch {
    return null;
  }
}

export function writeAnnotation(entryId, note) {
  const dir = getAnnotationsDir();
  mkdirSync(dir, { recursive: true });
  const data = {
    id: entryId,
    note,
    updatedAt: new Date().toISOString(),
  };
  writeFileSync(annotationPath(entryId), JSON.stringify(data, null, 2));
  return data;
}

export function clearAnnotation(entryId) {
  try {
    unlinkSync(annotationPath(entryId));
    return true;
  } catch {
    return false;
  }
}

export function listAnnotations() {
  const dir = getAnnotationsDir();
  try {
    const files = readdirSync(dir).filter((f) => f.endsWith('.json'));
    return files.map((f) => {
      try {
        return JSON.parse(readFileSync(join(dir, f), 'utf8'));
      } catch {
        return null;
      }
    }).filter(Boolean);
  } catch {
    return [];
  }
}
