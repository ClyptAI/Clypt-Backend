import chalk from 'chalk';
import { readAnnotation, writeAnnotation, clearAnnotation, listAnnotations } from '../lib/annotations.js';
import { output, error, info } from '../lib/output.js';

export function registerAnnotateCommand(program) {
  program
    .command('annotate [id] [note]')
    .description('Attach agent notes to a doc or skill')
    .option('--clear', 'Remove annotation for this entry')
    .option('--list', 'List all annotations')
    .action((id, note, opts) => {
      const globalOpts = program.optsWithGlobals();

      if (opts.list) {
        const annotations = listAnnotations();
        output(
          annotations,
          (data) => {
            if (data.length === 0) {
              console.log('No annotations.');
              return;
            }
            for (const a of data) {
              console.log(`${chalk.bold(a.id)} ${chalk.dim(`(${a.updatedAt})`)}`);
              console.log(`  ${a.note}`);
              console.log();
            }
          },
          globalOpts
        );
        return;
      }

      if (!id) {
        error('Missing required argument: <id>. Run: chub annotate <id> <note> | chub annotate <id> --clear | chub annotate --list', globalOpts);
      }

      if (opts.clear) {
        const removed = clearAnnotation(id);
        output(
          { id, cleared: removed },
          (data) => {
            if (data.cleared) {
              console.log(`Annotation cleared for ${chalk.bold(id)}.`);
            } else {
              console.log(`No annotation found for ${chalk.bold(id)}.`);
            }
          },
          globalOpts
        );
        return;
      }

      if (!note) {
        // Show existing annotation
        const existing = readAnnotation(id);
        if (existing) {
          output(
            existing,
            (data) => {
              console.log(`${chalk.bold(data.id)} ${chalk.dim(`(${data.updatedAt})`)}`);
              console.log(data.note);
            },
            globalOpts
          );
        } else {
          output(
            { id, note: null },
            () => console.log(`No annotation for ${chalk.bold(id)}.`),
            globalOpts
          );
        }
        return;
      }

      const data = writeAnnotation(id, note);
      output(
        data,
        (d) => console.log(`Annotation saved for ${chalk.bold(d.id)}.`),
        globalOpts
      );
    });
}
