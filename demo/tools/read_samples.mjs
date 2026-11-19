// Read the texts the ad's footage shows with the engine, and keep the numbers the ad draws around them (the strip's
// position and band, the verdict word, the token count), so what the hook and the end card show is the engine's own.
//
//   node tools/read_samples.mjs --litmus <litmus.exe> --out data/ad/readings.json [--manifest <release manifest.json>]
//
// The texts are data/ad/texts/ai_essay.txt (pasted into the app) and data/ad/texts/notes.txt (the editor document the
// quick check reads from), whose blank-line paragraphs are: two written by a language model for this demo (selected
// together), one short sentence, and one paragraph of a nineteenth-century novel (public domain).

import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc), []));
for (const k of ['litmus', 'out']) if (!args[k]) throw new Error(`give --${k}`);

const essay = readFileSync(join(here, '../data/ad/texts/ai_essay.txt'), 'utf8').trim();
const notes = readFileSync(join(here, '../data/ad/texts/notes.txt'), 'utf8').trim().split(/\n\n+/);
if (notes.length !== 4) throw new Error(`notes.txt should have four paragraphs, has ${notes.length}`);
const samples = {
  app: essay,
  quick_ai: `${notes[0]}\n\n${notes[1]}`,
  quick_short: notes[2],
  quick_human: notes[3],
};

const dir = mkdtempSync(join(tmpdir(), 'litmus-samples-'));
const out = {};
for (const [name, text] of Object.entries(samples)) {
  const file = join(dir, `${name}.txt`);
  writeFileSync(file, `${text}\n`);
  const row = JSON.parse(execFileSync(resolve(args.litmus), ['score', '--json', '--no-cache', file], { encoding: 'utf8', maxBuffer: 1 << 28 }).trim().split('\n')[0]);
  const v = row.reading.verdict;
  const set = v.sets['0.1'];
  out[name] = {
    label: row.label,
    set,
    tokens: row.reading.n_tokens,
    probability: Number(v.probability.toFixed(4)),
    strip: { position: Number(v.strip.position.toFixed(4)), band_lo: Number(v.strip.band_lo.toFixed(4)), band_hi: Number(v.strip.band_hi.toFixed(4)), unlike_reference: v.strip.unlike_reference },
    out_of_reference: v.out_of_reference,
  };
}
// The ad's captions say what these readings show ("AI", "It says when it can't tell", a human reading); a release that
// reads a text differently must be given another text, not another caption by accident.
const expected = { app: 'AI', quick_ai: 'AI', quick_short: 'uncertain', quick_human: 'human' };
for (const [name, label] of Object.entries(expected)) {
  if (out[name].label !== label) throw new Error(`${name} reads "${out[name].label}", not "${label}": choose another text for it in data/ad/texts and record the footage again`);
}
const manifest = args.manifest ? JSON.parse(readFileSync(resolve(args.manifest), 'utf8')) : null;
writeFileSync(
  resolve(args.out),
  JSON.stringify({ about: 'Measured by tools/read_samples.mjs: the engine\'s reading of each text the footage shows. Replaced by re-measuring with a new release.', model: manifest ? { version: manifest.model.version, size: manifest.model.size, checkpoint_step: manifest.model.checkpoint_step } : null, readings: out }, null, 1),
);
console.log(JSON.stringify(out, null, 1));
