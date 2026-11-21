// One command from a release to the finished ad: re-read the sample texts, re-run the pinned speed measurement,
// recapture the in-browser checker, prepare the footage, and render the ad, its teaser, and the wide cut (with its web encodes).
//
//   node make.mjs --work <dir> --litmus <litmus.exe> --site <built site folder> --manifest <release manifest.json>
//                 --clips <dir with app.mp4, app.json, quick.mp4, quick.json, memory.json from capture/run_linux.sh> [--core 6]
//                 [--cuts wide] [--workers <n>]
//
// The desktop footage and the app's memory come from the Linux app on a virtual display (capture/run_linux.sh), recorded
// on a machine with Docker; everything else runs here. Each step is a script in tools/ or capture/ and can be run alone.
// The numbers the ad draws are written to data/ad/ (readings.json, hype_run.json and app_memory.json), replacing the
// earlier release's. The two sets of documents are fetched once into <work>/sets and reused; delete the folder to fetch
// a fresh draw.

import { spawnSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc), []));
for (const k of ['work', 'litmus', 'site', 'manifest', 'clips']) if (!args[k]) throw new Error(`give --${k}`);
// `--cuts wide` renders the one cut a release ships (the site's film and the README's preview); the default is all three.
const cuts = args.cuts ?? 'all';
const work = resolve(args.work);
const clips = resolve(args.clips);
for (const f of ['app.mp4', 'app.json', 'quick.mp4', 'quick.json', 'memory.json']) if (!existsSync(join(clips, f))) throw new Error(`${join(clips, f)} is missing: record the desktop footage first (capture/linux_capture.py)`);
mkdirSync(work, { recursive: true });

const step = (label, script, argv) => {
  console.log(`\n== ${label}`);
  const r = spawnSync(process.execPath, [join(here, script), ...argv], { stdio: 'inherit' });
  if (r.status !== 0) throw new Error(`${label} failed`);
};

step('install the dependencies', 'render.mjs', ['--work', work, '--install']);
step('read the sample texts', 'tools/read_samples.mjs', ['--litmus', args.litmus, '--manifest', args.manifest, '--out', join(here, 'data/ad/readings.json')]);
const sets = join(work, 'sets');
if (!existsSync(join(sets, 'chosen.json'))) step('fetch the two sets of documents', 'tools/fetch_sets.mjs', ['--out', sets]);
step('measure the pinned run', 'tools/measure_run.mjs', ['--litmus', args.litmus, '--sets', sets, '--work', join(work, 'measure'), '--out', join(here, 'data/ad/hype_run.json'), '--core', args.core ?? '6', '--manifest', args.manifest]);
step('summarize the app memory', 'tools/read_app_memory.mjs', ['--in', join(clips, 'memory.json'), '--out', join(here, 'data/ad/app_memory.json'), '--manifest', args.manifest]);
const checker = join(work, 'checker');
step('capture the browser checker', 'capture/site_capture.mjs', ['--site', args.site, '--text', join(here, 'data/ad/texts/ai_essay.txt'), '--out', checker, '--work', work]);
const staged = join(work, 'clips');
mkdirSync(staged, { recursive: true });
cpSync(clips, staged, { recursive: true });
cpSync(checker, join(staged, 'site'), { recursive: true });
step('prepare the footage', 'tools/prepare_footage.mjs', ['--clips', staged, '--out', join(work, 'footage'), '--work', work]);
step(`render ${cuts === 'all' ? 'the wide cut, the ad, and the teaser' : cuts}`, 'render.mjs', [cuts, '--work', work, '--footage', join(work, 'footage'), '--out', join(work, 'out'), '--stills', ...(args.workers ? ['--workers', args.workers] : [])]);
console.log(`\ndone: ${join(work, 'out')}`);
