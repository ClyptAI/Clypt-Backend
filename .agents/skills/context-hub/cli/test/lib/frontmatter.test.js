import { describe, it, expect } from 'vitest';
import { parseFrontmatter } from '../../src/lib/frontmatter.js';

describe('parseFrontmatter', () => {
  it('parses valid YAML frontmatter', () => {
    const content = `---
name: test
description: A test entry
metadata:
  languages: "python"
  versions: "1.0.0"
---

# Test Content

Body text here.`;

    const { attributes, body } = parseFrontmatter(content);
    expect(attributes.name).toBe('test');
    expect(attributes.description).toBe('A test entry');
    expect(attributes.metadata.languages).toBe('python');
    expect(attributes.metadata.versions).toBe('1.0.0');
    expect(body).toContain('# Test Content');
    expect(body).toContain('Body text here.');
  });

  it('returns empty attributes when no frontmatter', () => {
    const content = '# Just a heading\n\nSome content.';
    const { attributes, body } = parseFrontmatter(content);
    expect(attributes).toEqual({});
    expect(body).toBe(content);
  });

  it('handles frontmatter with no metadata field', () => {
    const content = `---
name: simple
description: No metadata
---

Content.`;

    const { attributes } = parseFrontmatter(content);
    expect(attributes.name).toBe('simple');
    expect(attributes.metadata).toBeUndefined();
  });

  it('handles empty frontmatter block', () => {
    const content = `---
---

Content.`;

    const { attributes, body } = parseFrontmatter(content);
    expect(attributes).toEqual({});
    expect(body).toContain('Content.');
  });
});
