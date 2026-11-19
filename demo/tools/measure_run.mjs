// The speed run, measured: read the two sets of long documents (written by people, written by AI) with the `litmus` command
// line pinned to one core, and record when each document starts and finishes, what the process costs in memory and
// processor time (from the operating system's own counters), and how busy the machine is while it runs.
//
//   node tools/measure_run.mjs --litmus <litmus.exe> --sets <fetch_sets output dir> --work <scratch dir> --out <json>
//                              [--core 6] [--manifest <release manifest.json>]
//
// Two passes over the same files, both pinned by the operating system at launch (`start /affinity`), so the process never
// runs on another core:
//   A. one `litmus score` process per document, in turn: gives each document's time and reading (the command line prints
//      its readings only when it has read them all, so a per-document time needs a per-document process), and the replay.
//   B. one process reading every document: gives the speed with the model loaded once, and the memory of a process that
//      has read them all.
// The memory and the processor time of every `litmus` process are sampled by tools/sample_process.ps1 every 50 ms or so
// (Windows' working set, peak working set and private bytes; processor seconds). The starting of a process is timed on a
// one-word text beforehand and reported, never hidden inside a speed. The file written is everything the ad draws in its
// speed segment; it names no document, only the hash of the list.

import { spawn, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc), []));
for (const k of ['litmus', 'sets', 'work', 'out']) if (!args[k]) throw new Error(`give --${k}`);
const litmus = resolve(args.litmus);
const sets = resolve(args.sets);
const work = resolve(args.work);
const core = Number(args.core ?? 6);
const cores = os.cpus().length;
if (core >= cores) throw new Error(`core ${core} does not exist on a machine with ${cores}`);
if (process.platform !== 'win32') throw new Error('this measurement pins with `start /affinity` and reads Windows counters, so it runs on Windows');
const MB = 1024 * 1024;
const mask = (1n << BigInt(core)).toString(16);

const chosen = JSON.parse(readFileSync(join(sets, 'chosen.json'), 'utf8'));
const kinds = ['human', 'ai'];
const per = Object.fromEntries(kinds.map((k) => [k, chosen[k].documents.map((d) => ({ kind: k, ...d, text: readFileSync(join(sets, 'text', k, d.file), 'utf8') }))]));
const n = Math.min(...kinds.map((k) => per[k].length));
const order = [];
for (let i = 0; i < n; i++) for (const k of kinds) order.push(per[k][i]);
const listHash = createHash('sha256').update(order.map((d) => d.sha256).join('\n')).digest('hex');

rmSync(join(work, 'run'), { recursive: true, force: true });
mkdirSync(join(work, 'run'), { recursive: true });
mkdirSync(join(work, 'tiny'), { recursive: true });
order.forEach((d, i) => writeFileSync(join(work, 'run', `${String(i).padStart(3, '0')}.txt`), d.text));
writeFileSync(join(work, 'tiny', 'tiny.txt'), 'Hello.\n');

/** Run a pinned `litmus` with these arguments, resolving with the seconds it took and its stdout lines. */
const runPinned = (litmusArgs, outFile) =>
  new Promise((ok, fail) => {
    rmSync(outFile, { force: true });
    const cmdline = `start "" /b /wait /affinity ${mask} "${litmus}" ${litmusArgs} > "${outFile}"`;
    const t0 = process.hrtime.bigint();
    const child = spawn('cmd.exe', ['/d', '/s', '/c', `"${cmdline}"`], { windowsVerbatimArguments: true, windowsHide: true, stdio: 'ignore' });
    child.on('exit', (code) => {
      const seconds = Number(process.hrtime.bigint() - t0) / 1e9;
      if (code !== 0 || !existsSync(outFile)) fail(new Error(`litmus exited ${code} on ${litmusArgs}`));
      else ok({ seconds, lines: readFileSync(outFile, 'utf8').trim().split('\n') });
    });
  });
const scoreArgs = (target) => `score --json --no-cache "${target}"`;

// The memory and processor sampler, started first and stopped last.
const samples = [];
const sampler = spawn('powershell.exe', ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', join(here, 'sample_process.ps1'), '-Name', 'litmus', '-IntervalMs', '50'], { stdio: ['ignore', 'pipe', 'ignore'], windowsHide: true });
let partial = '';
sampler.stdout.on('data', (chunk) => {
  const lines = (partial + chunk.toString()).split('\n');
  partial = lines.pop() ?? '';
  for (const line of lines) {
    const [ms, pid, ws, peak, priv, cpu] = line.trim().split(',').map(Number);
    if (Number.isFinite(cpu)) samples.push({ ms, pid, ws, peak, priv, cpu });
  }
});
await new Promise((r) => setTimeout(r, 1500));

const mark = () => Date.now();
// The cost of starting the program, on a one-word text, pinned the same way; also the memory of a loaded, idle model.
const tinyFrom = mark();
const startups = [];
for (let i = 0; i < 7; i++) startups.push((await runPinned(scoreArgs(join(work, 'tiny', 'tiny.txt')), join(work, 'one.json'))).seconds);
const tinyTo = mark();
startups.sort((a, b) => a - b);
const startup = startups[Math.floor(startups.length / 2)];

// Pass A: one process per document, with the machine's trace.
const clockMs = Date.now();
const clock = process.hrtime.bigint();
const now = () => Number(process.hrtime.bigint() - clock) / 1e9;
const cpuTimes = () => os.cpus().map((c) => c.times);
const trace = [];
let last = cpuTimes();
const watcher = setInterval(() => {
  const cur = cpuTimes();
  trace.push({
    t: now(),
    busy: cur.map((c, i) => {
      const sum = (x) => x.user + x.nice + x.sys + x.idle + x.irq;
      const total = sum(c) - sum(last[i]);
      return total > 0 ? 1 - (c.idle - last[i].idle) / total : 0;
    }),
    used_mb: (os.totalmem() - os.freemem()) / MB,
  });
  last = cur;
}, 100);

const rows = [];
const aFrom = mark();
for (const [i, d] of order.entries()) {
  const t_start = now();
  const { lines } = await runPinned(scoreArgs(join(work, 'run', `${String(i).padStart(3, '0')}.txt`)), join(work, 'one.json'));
  const t_end = now();
  const row = JSON.parse(lines[0]);
  const v = row.reading.verdict;
  const set = v.sets['0.1'];
  const verdict = set.length === 0 ? 'unlike' : set.length === 1 ? set[0] : 'uncertain';
  rows.push({ set: d.kind, source: d.kind === 'ai' ? d.model : 'Project Gutenberg', markdown: /^(#{1,6} |\*\*|---)/m.test(d.text), tokens: row.reading.n_tokens, verdict, p_not_human: Number(v.probability.toFixed(4)), t_start, t_end });
  if (i % 20 === 0) console.log(`${i + 1} of ${order.length}`);
}
const aTo = mark();
clearInterval(watcher);

// Pass B: one process over every document.
const bFrom = mark();
const batch = await runPinned(scoreArgs(join(work, 'run')), join(work, 'all.json'));
const bTo = mark();
const batchRows = batch.lines.map((l) => JSON.parse(l));
if (batchRows.length !== order.length) throw new Error(`the batch read ${batchRows.length} documents, not ${order.length}`);

// The engine's own account of its peak memory and speed, pinned the same way, for comparison with the operating system's.
const benchFrom = mark();
const bench = JSON.parse((await runPinned('bench --tokens 8192 --repeat 2 --json', join(work, 'bench.json'))).lines.pop());
const benchTo = mark();

await new Promise((r) => setTimeout(r, 500));
sampler.kill();

// ---- the numbers ----

const inWindow = (from, to) => samples.filter((s) => s.ms >= from && s.ms <= to);
const peakWs = (list) => Math.max(...list.map((s) => Math.max(s.ws, s.peak))) / MB;
const peakPriv = (list) => Math.max(...list.map((s) => s.priv)) / MB;
const medianOf = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const pids = (list) => [...new Set(list.map((s) => s.pid))];
const cpuSecondsOf = (list) => pids(list).reduce((acc, pid) => acc + Math.max(...list.filter((s) => s.pid === pid).map((s) => s.cpu)), 0);

const aSamples = inWindow(aFrom, aTo + 200);
const bSamples = inWindow(bFrom, bTo + 200);
const tinySamples = inWindow(tinyFrom, tinyTo + 200);
if (aSamples.length < 10 || bSamples.length < 3) throw new Error(`the sampler saw too little (${aSamples.length} and ${bSamples.length} samples): is PowerShell blocked?`);

const begin = rows[0].t_start;
const finish = rows[rows.length - 1].t_end;
const tokens = rows.reduce((a, d) => a + d.tokens, 0);
const batchSeconds = (bTo - bFrom) / 1000;
const within = trace.filter((s) => s.t >= begin && s.t <= finish);
const mean = (xs) => xs.reduce((a, v) => a + v, 0) / Math.max(1, xs.length);
const aSeconds = finish - begin;
const aCpu = cpuSecondsOf(aSamples);

// The replay trace: every 100 ms, the processor seconds `litmus` used in it (from the samples of each process),
// its working set, and the machine's utilization and memory in use.
const bins = within.map((s) => {
  const from = clockMs + s.t * 1000 - 100;
  const to = clockMs + s.t * 1000;
  let cpuSeconds = 0;
  let ws = 0;
  for (const pid of pids(aSamples)) {
    const mine = aSamples.filter((x) => x.pid === pid).sort((a, b) => a.ms - b.ms);
    for (let k = 0; k < mine.length; k++) {
      if (mine[k].ms > from && mine[k].ms <= to) {
        cpuSeconds += mine[k].cpu - (k > 0 ? mine[k - 1].cpu : 0);
        ws = Math.max(ws, mine[k].ws / MB);
      }
    }
  }
  return { t: s.t - begin, cpuSeconds, ws, busy: s.busy, used_mb: s.used_mb };
});

// Document tally, overall and by set and by generating model.
const countBy = (list) => {
  const count = (v) => list.filter((d) => d.verdict === v).length;
  return { documents: list.length, tokens: list.reduce((a, d) => a + d.tokens, 0), human: count('human'), uncertain: count('uncertain'), mixed: count('mixed'), ai: count('ai'), unlike: count('unlike') };
};
const tally = Object.fromEntries(kinds.map((k) => [k, countBy(rows.filter((d) => d.set === k))]));
const byModel = Object.fromEntries([...new Set(rows.filter((d) => d.set === 'ai').map((d) => d.source))].map((m) => [m, countBy(rows.filter((d) => d.source === m))]));

// The generated stories keep the formatting they were written with (some have markdown headings or bold lines, the
// people-written documents have none): the tally is also given with and without it, so a reading cannot be put down to it unseen.
const tallyByFormatting = Object.fromEntries([true, false].map((md) => [md ? 'ai_with_markdown' : 'ai_plain', countBy(rows.filter((d) => d.set === 'ai' && d.markdown === md))]));

// The batch must read every document as the per-document passes did.
const byFile = new Map(batchRows.map((r) => [r.file.split(/[\\/]/).pop(), r]));
order.forEach((d, i) => {
  const r = byFile.get(`${String(i).padStart(3, '0')}.txt`);
  const set = r.reading.verdict.sets['0.1'];
  const verdict = set.length === 0 ? 'unlike' : set.length === 1 ? set[0] : 'uncertain';
  if (verdict !== rows[i].verdict || r.reading.n_tokens !== rows[i].tokens) throw new Error(`document ${i} reads differently in the batch than alone`);
});

const lengths = (k) => per[k].map((d) => d.chars);
const info = spawnSync(litmus, ['info'], { encoding: 'utf8' }).stdout.split('\n')[0];
const manifest = args.manifest ? JSON.parse(readFileSync(resolve(args.manifest), 'utf8')) : null;
const result = {
  about: 'Measured by tools/measure_run.mjs; the ad draws this file and nothing else for its speed segment. Re-measure with a new release to replace it.',
  basis: {
    measured_at: new Date().toISOString(),
    machine: { cpu: os.cpus()[0].model.trim(), logical_cores: cores, memory_mb: Math.round(os.totalmem() / MB), os: `${os.type()} ${os.release()}`, node: process.version },
    pinned_core: core,
    affinity_mask: `0x${mask}`,
    commands: {
      per_document: `start /affinity ${mask} litmus score --json --no-cache <one text file>, once per document, in turn (pass A)`,
      batch: `start /affinity ${mask} litmus score --json --no-cache <a folder of all the documents>, one process (pass B)`,
      engine_bench: `start /affinity ${mask} litmus bench --tokens 8192 --repeat 2 --json`,
    },
    memory_basis: 'Windows process counters read by tools/sample_process.ps1 about every 50 ms for each litmus process: working set (physical memory held), peak working set, and private bytes (committed memory the process alone holds). The peaks are the largest of those readings and of the system-kept peak working set.',
    cpu_basis: 'Processor seconds the litmus processes used (their TotalProcessorTime), divided by the wall seconds and by the machine\'s logical cores; the machine\'s own utilization is from os.cpus() times every 100 ms, so everything else is the machine minus litmus.',
    engine: info.trim(),
    model: manifest ? { version: manifest.model.version, size: manifest.model.size, checkpoint_step: manifest.model.checkpoint_step } : null,
    document_sets: {
      list_sha256: listHash,
      per_set: n,
      rule: chosen.rule,
      human: { source: chosen.human.source, chars: { min: Math.min(...lengths('human')), median: medianOf(lengths('human')), max: Math.max(...lengths('human')) } },
      ai: { source: chosen.ai.source, chars: { min: Math.min(...lengths('ai')), median: medianOf(lengths('ai')), max: Math.max(...lengths('ai')) }, models: Object.fromEntries(Object.entries(byModel).map(([m, t]) => [m, t.documents])) },
    },
    startup_seconds: Number(startup.toFixed(3)),
  },
  totals: {
    documents: rows.length,
    tokens,
    per_document: { seconds: Number(aSeconds.toFixed(3)), tokens_per_second: Number((tokens / aSeconds).toFixed(0)) },
    batch: { seconds: Number(batchSeconds.toFixed(3)), tokens_per_second: Number((tokens / batchSeconds).toFixed(0)) },
    engine_bench: { tokens: bench.tokens, text_tokens_per_second: Math.round(bench.text_tokens_per_second), process_peak_mb: Number((bench.process_peak_bytes / MB).toFixed(1)) },
    cpu: {
      litmus_share_of_machine: Number((aCpu / (aSeconds * cores)).toFixed(4)),
      litmus_cores_busy: Number((aCpu / aSeconds).toFixed(3)),
      machine_busy_mean: Number(mean(within.map((s) => mean(s.busy))).toFixed(4)),
      pinned_core_busy_mean: Number(mean(within.map((s) => s.busy[core])).toFixed(3)),
      other_cores_busy_mean: Number(mean(within.map((s) => s.busy.reduce((b, v, i) => (i === core ? b : b + v), 0) / (cores - 1))).toFixed(3)),
    },
    memory_mb: {
      peak_working_set: Number(peakWs(aSamples.concat(bSamples)).toFixed(1)),
      peak_private: Number(peakPriv(aSamples.concat(bSamples)).toFixed(1)),
      batch_peak_working_set: Number(peakWs(bSamples).toFixed(1)),
      batch_peak_private: Number(peakPriv(bSamples).toFixed(1)),
      per_document_median_peak_working_set: Number(medianOf(pids(aSamples).map((pid) => peakWs(aSamples.filter((s) => s.pid === pid)))).toFixed(1)),
      idle_model_peak_working_set: Number(peakWs(tinySamples).toFixed(1)),
      machine_in_use_mean: Number(mean(within.map((s) => s.used_mb)).toFixed(0)),
    },
    samples: { per_document_pass: aSamples.length, batch_pass: bSamples.length },
  },
  tally,
  tally_by_model: byModel,
  tally_by_formatting: tallyByFormatting,
  documents: rows.map((d) => ({ ...d, t_start: Number((d.t_start - begin).toFixed(3)), t_end: Number((d.t_end - begin).toFixed(3)) })),
  trace: {
    interval_s: 0.1,
    cores,
    samples: bins.map((b) => [Number(b.t.toFixed(2)), Number(b.cpuSeconds.toFixed(4)), Number(b.ws.toFixed(1)), Math.round(mean(b.busy) * 1000) / 10, Math.round(b.used_mb), ...b.busy.map((x) => Math.round(x * 100))]),
    columns: 't, litmus processor seconds in the interval, litmus working set MB, machine busy %, machine memory in use MB, then each core busy %',
  },
};
writeFileSync(resolve(args.out), JSON.stringify(result));
const t = result.totals;
console.log(`${t.documents} documents, ${t.tokens} tokens: per-document ${t.per_document.seconds} s (${t.per_document.tokens_per_second} tokens/s with startup ${startup.toFixed(3)} s each), one process ${t.batch.seconds} s (${t.batch.tokens_per_second} tokens/s)`);
console.log(`litmus used ${(t.cpu.litmus_share_of_machine * 100).toFixed(1)}% of the machine (${t.cpu.litmus_cores_busy} cores), peak working set ${t.memory_mb.peak_working_set} MB, private ${t.memory_mb.peak_private} MB; engine says ${t.engine_bench.process_peak_mb} MB`);
console.log(JSON.stringify(tally));
console.log(JSON.stringify(Object.fromEntries(Object.entries(byModel).map(([m, x]) => [m, { ai: x.ai, human: x.human, uncertain: x.uncertain, mixed: x.mixed }]))));
console.log(JSON.stringify(tallyByFormatting));
