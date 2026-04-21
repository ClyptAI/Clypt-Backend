import { existsSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import chalk from 'chalk';
import { getChubDir } from './config.js';

const WELCOME_MARKER = '.welcome_shown';

/**
 * Show the first-run welcome notice if it hasn't been shown yet.
 * Creates a marker file so it only displays once.
 */
export function showWelcomeIfNeeded(opts = {}) {
  if (opts.json) return;
  if (!process.stdout.isTTY || !process.stderr.isTTY) return;

  const chubDir = getChubDir();
  const markerPath = join(chubDir, WELCOME_MARKER);
  const configPath = join(chubDir, 'config.yaml');

  if (existsSync(markerPath)) return;

  // Print to stderr so it doesn't interfere with JSON/piped output
  console.error(`
${chalk.bold('Welcome to Context Hub (chub)!')} Chub helps your AI coding agents make API calls correctly, by providing \
the latest documentation.

By using chub, you agree to the Terms of Service at ${chalk.underline('https://www.aichub.org/tos.html')}

Chub asks agents to provide feedback on documentation, and this feedback is used to improve docs for the developer \
community. If you wish to disable this feedback, add ${chalk.bold('"feedback: false"')} to ${chalk.bold(configPath)}. See \
${chalk.underline('https://github.com/andrewyng/context-hub')} for details.
`);

  try {
    if (!existsSync(chubDir)) {
      mkdirSync(chubDir, { recursive: true });
    }
    writeFileSync(markerPath, new Date().toISOString(), 'utf8');
  } catch {
    // Best-effort — don't block CLI if marker can't be written
  }
}
