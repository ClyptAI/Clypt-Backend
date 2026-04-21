import { describe, it, expect } from 'vitest';
import { normalizeLanguage, displayLanguage } from '../../src/lib/normalize.js';

describe('normalizeLanguage', () => {
  it('normalizes short aliases to full names', () => {
    expect(normalizeLanguage('py')).toBe('python');
    expect(normalizeLanguage('js')).toBe('javascript');
    expect(normalizeLanguage('ts')).toBe('typescript');
    expect(normalizeLanguage('rb')).toBe('ruby');
    expect(normalizeLanguage('cs')).toBe('csharp');
  });

  it('passes through full language names', () => {
    expect(normalizeLanguage('python')).toBe('python');
    expect(normalizeLanguage('javascript')).toBe('javascript');
    expect(normalizeLanguage('typescript')).toBe('typescript');
  });

  it('is case-insensitive', () => {
    expect(normalizeLanguage('PY')).toBe('python');
    expect(normalizeLanguage('JS')).toBe('javascript');
    expect(normalizeLanguage('Python')).toBe('python');
  });

  it('returns null for null/undefined', () => {
    expect(normalizeLanguage(null)).toBeNull();
    expect(normalizeLanguage(undefined)).toBeNull();
  });

  it('passes through unknown languages as lowercase', () => {
    expect(normalizeLanguage('rust')).toBe('rust');
    expect(normalizeLanguage('Go')).toBe('go');
  });
});

describe('displayLanguage', () => {
  it('converts full names to short display forms', () => {
    expect(displayLanguage('javascript')).toBe('js');
    expect(displayLanguage('typescript')).toBe('ts');
    expect(displayLanguage('python')).toBe('py');
  });

  it('passes through unknown languages', () => {
    expect(displayLanguage('rust')).toBe('rust');
    expect(displayLanguage('go')).toBe('go');
  });
});
