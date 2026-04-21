/**
 * Dual-mode output: human-friendly (default) or JSON (--json flag).
 *
 * Every command calls `output(data, humanFormatter, opts)`.
 * - In JSON mode: prints JSON to stdout, nothing else.
 * - In human mode: calls humanFormatter(data) which prints with chalk.
 */
export function output(data, humanFormatter, opts) {
  if (opts?.json) {
    console.log(JSON.stringify(data, null, 2));
  } else {
    humanFormatter(data);
  }
}

/**
 * Print a message to stderr (for confirmations when -o is used).
 */
export function info(msg) {
  process.stderr.write(msg + '\n');
}

/**
 * Print an error and exit.
 */
export function error(msg, opts) {
  if (opts?.json) {
    console.log(JSON.stringify({ error: msg }));
  } else {
    process.stderr.write(`Error: ${msg}\n`);
  }
  process.exit(1);
}
