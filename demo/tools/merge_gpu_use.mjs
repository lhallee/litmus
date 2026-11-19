// Merge the GPU measurements of one or more machines (the JSON that `python -m litmus_benchmarks gpu-use` writes) into
// data/ad/gpu_use.json, the file the speed shot's GPU panel reads. A detector that does not fit one machine's GPU was
// measured on another, so every row carries the machine it was measured on, and the file's basis says so plainly.
//
//   node tools/merge_gpu_use.mjs --in <a10.json> <gh200.json> --out data/ad/gpu_use.json [--note "text" ...]
//
// The inputs must be finished measurements (not pilots) of the same documents and each must have a working per-process
// query. Exactly one machine's file may hold the Litmus engine's row. Each --note is carried into `basis.notes` word for word
// (what was tried and did not work, with its evidence).

import { readFileSync, writeFileSync } from 'node:fs';
import { basename, resolve } from 'node:path';

const argv = process.argv.slice(2);
const take = (flag, many = false) => {
  const out = [];
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] !== flag) continue;
    for (let j = i + 1; j < argv.length && !argv[j].startsWith('--'); j++) {
      out.push(argv[j]);
      if (!many) break;
    }
  }
  return out;
};
const inputs = take('--in', true);
const [out] = take('--out');
if (inputs.length === 0 || !out) throw new Error('give --in <json>... and --out <json>');
const notes = take('--note');

const files = inputs.map((path) => ({ path: resolve(path), json: JSON.parse(readFileSync(resolve(path), 'utf8')) }));
const documents = JSON.stringify(files[0].json.documents);
for (const { path, json } of files) {
  if (json.schema !== 1) throw new Error(`${path}: schema ${json.schema}, expected 1`);
  if (json.pilot) throw new Error(`${path} is a pilot run`);
  if (!json.per_process_query_works) throw new Error(`${path}: the per-process query did not work on that machine`);
  if (JSON.stringify(json.documents) !== documents) throw new Error(`${path} read other documents than ${files[0].path}`);
}

const detectors = files.flatMap(({ json }) => json.detectors.map((row) => ({ ...row, machine: json.machine, idle_baseline: json.idle_baseline })));
const names = detectors.map((row) => row.detector);
if (new Set(names).size !== names.length) throw new Error(`a detector appears twice: ${names.join(', ')}`);
const litmus = detectors.filter((row) => row.device === 'cpu');
if (litmus.length !== 1) throw new Error(`expected exactly one Litmus row, found ${litmus.length}`);

const tokens = (row) => row.tokens_read.toLocaleString('en-US');
const others = detectors.filter((row) => row.device !== 'cpu');
const basis = {
  summary:
    `GPU busy time (nvidia-smi utilization.gpu, the share of time a kernel was running, averaged over the reading, polled every ${files[0].json.sampling.interval_achieved_seconds} s) and the peak GPU memory of each detector's own process, ` +
    `each reading the same ${files[0].json.documents.documents} documents (${files[0].json.documents.words.toLocaleString('en-US')} words), measured by python -m litmus_benchmarks gpu-use.`,
  reading:
    `The other detectors truncate each document to their input limit and read ${others.map((row) => `${row.detector} ${tokens(row)}`).join(', ')} tokens in all; ` +
    `Litmus's engine read every token of every document, ${tokens(litmus[0])} tokens. The figures say what each detector's GPU did while it read its own share, not that they read the same text.`,
  windows:
    `Reading took (${others.map((row) => `${row.detector} ${row.wall_seconds.toFixed(1)} s, ${row.polls_during_reading} polls`).join('; ')}), so a mean busy time rests on the polls named.`,
  machines: Object.fromEntries(files.map(({ json }) => [json.machine.name, json.detectors.map((row) => row.detector)])),
  notes,
};

const merged = {
  schema: 1,
  pilot: false,
  measured_at: files.map(({ json }) => json.measured_at),
  sources: files.map(({ path }) => basename(path)), // the files' names: a full path names the machine they were merged on
  basis,
  documents: files[0].json.documents,
  per_process_query_works: true,
  detectors,
};
writeFileSync(resolve(out), `${JSON.stringify(merged, null, 1)}\n`);
console.log(`${out}: ${detectors.length} rows from ${files.length} machines`);
