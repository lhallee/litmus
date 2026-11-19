// Summarize the desktop app's memory, as measured by `capture/linux_capture.py memory` on the Linux build: the app's
// resident memory with the model loaded and idle, with a long text pasted, and while it checks that text, from the
// operating system's own counters (/proc/<pid>/status), and the processor time it used while checking.
//
//   node tools/read_app_memory.mjs --in <memory.json from the capture> --out data/ad/app_memory.json [--manifest <release manifest.json>]

import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc), []));
for (const k of ['in', 'out']) if (!args[k]) throw new Error(`give --${k}`);
const capture = JSON.parse(readFileSync(resolve(args.in), 'utf8'));
const events = capture.events;
const start = events.find((e) => e.what === 'start');
const phase = (what) => events.filter((e) => e.what === what && e.rss_kb !== undefined);
const mb = (kb) => Number((kb / 1024).toFixed(1));
const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const idle = phase('idle');
const pasted = phase('pasted');
const checking = phase('checking');
if (idle.length < 5 || checking.length < 20) throw new Error('the capture has too few samples');
const characters = events.find((e) => e.what === 'paste').characters;

// Processor seconds the app used in the checking phase (the reading itself is a fraction of a second, and the rest is idle).
const wall = checking[checking.length - 1].t - checking[0].t;
const cpuSeconds = (checking[checking.length - 1].cpu_ticks - checking[0].cpu_ticks) / start.ticks_per_second;
const manifest = args.manifest ? JSON.parse(readFileSync(resolve(args.manifest), 'utf8')) : null;
const result = {
  about: 'Measured by capture/linux_capture.py (the memory scene) and summarized by tools/read_app_memory.mjs: the Linux desktop app with the model loaded, then checking one long text.',
  basis: {
    measured_at: new Date().toISOString(),
    where: 'the Linux build of the app in a container on a virtual display, software rendering, no GPU, six processor cores available to the container; the app is not pinned to one core, so the processor figure is what it used, not a limit',
    counters: '/proc/<pid>/status VmRSS (resident now), VmHWM (resident at its peak), RssAnon (the private part), sampled every 100 ms; processor time from /proc/<pid>/stat',
    text_characters: characters,
    model: manifest ? { version: manifest.model.version, size: manifest.model.size, checkpoint_step: manifest.model.checkpoint_step } : null,
  },
  memory_mb: {
    idle_resident: mb(median(idle.map((e) => e.rss_kb))),
    pasted_resident: mb(median(pasted.map((e) => e.rss_kb))),
    checking_peak_resident: mb(Math.max(...checking.map((e) => e.peak_rss_kb))),
    checking_peak_private: mb(Math.max(...checking.map((e) => e.private_rss_kb))),
  },
  processor: { window_wall_seconds: Number(wall.toFixed(1)), window_cpu_seconds: Number(cpuSeconds.toFixed(2)), window: 'the 12 s from pressing Check, which includes the seconds after the reading, when the app is idle' },
};
writeFileSync(resolve(args.out), JSON.stringify(result, null, 1));
console.log(JSON.stringify(result, null, 1));
