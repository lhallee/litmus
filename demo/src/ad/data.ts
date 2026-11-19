// The ad's inputs, all measured and stored in data/ad: the engine's readings of the texts the footage shows, and the
// pinned speed run. The scenes draw these and type no number of their own.

import hypeFile from '../../data/ad/hype_run.json';
import readingsFile from '../../data/ad/readings.json';

export type Origin = 'human' | 'mixed' | 'ai';

export interface SampleReading {
  label: string;
  set: Origin[];
  tokens: number;
  probability: number;
  strip: { position: number; band_lo: number; band_hi: number; unlike_reference: boolean };
}

export const sample = (key: 'app' | 'quick_ai' | 'quick_short' | 'quick_human'): SampleReading => {
  const r = (readingsFile as unknown as { readings: Record<string, SampleReading> }).readings[key];
  if (!r) throw new Error(`data/ad/readings.json has no reading "${key}"`);
  return r;
};

const NAMES: Record<Origin, string> = { human: 'human', mixed: 'mixed', ai: 'AI' };
const ORDER: Origin[] = ['human', 'mixed', 'ai'];

/** The app's headline for a set of origins it cannot rule out (ported from app/src/wording.rs: one origin names itself, several read "Uncertain" and list themselves in strip order). */
export const headline = (set: Origin[]): string => {
  const named = ORDER.filter((o) => set.includes(o));
  if (named.length === 0) return 'Unlike reference';
  if (named.length === 1) return named[0] === 'ai' ? 'AI' : named[0][0].toUpperCase() + named[0].slice(1);
  const words = named.map((o) => NAMES[o]);
  return `Uncertain: ${words.length === 2 ? words.join(' or ') : `${words.slice(0, -1).join(', ')}, or ${words[words.length - 1]}`}`;
};

export type Verdict = 'human' | 'uncertain' | 'mixed' | 'ai' | 'unlike';
export type SetName = 'human' | 'ai';

export interface HypeDoc {
  set: SetName;
  source: string;
  tokens: number;
  verdict: Verdict;
  p_not_human: number;
  t_start: number;
  t_end: number;
}

export interface Tally {
  documents: number;
  tokens: number;
  human: number;
  uncertain: number;
  mixed: number;
  ai: number;
  unlike: number;
}

export interface HypeRun {
  basis: { pinned_core: number; startup_seconds: number; machine: { cpu: string; logical_cores: number; memory_mb: number }; document_sets: { per_set: number } };
  totals: {
    documents: number;
    tokens: number;
    per_document: { seconds: number; tokens_per_second: number };
    batch: { seconds: number; tokens_per_second: number };
    engine_bench: { tokens: number; text_tokens_per_second: number; process_peak_mb: number };
    cpu: { litmus_share_of_machine: number; litmus_cores_busy: number; machine_busy_mean: number; pinned_core_busy_mean: number; other_cores_busy_mean: number };
    memory_mb: { peak_working_set: number; peak_private: number; batch_peak_working_set: number; machine_in_use_mean: number };
  };
  tally: Record<SetName, Tally>;
  documents: HypeDoc[];
  /** One row per interval: time, litmus processor seconds in it, litmus working set (MB), machine busy %, machine memory in use (MB), then each core's busy %. */
  trace: { interval_s: number; cores: number; samples: number[][] };
}

export const hype: HypeRun = hypeFile as unknown as HypeRun;

const COL = { cpuSeconds: 1, ws: 2, busy: 3, used: 4 };

/** The trace's index at run time `r` seconds. */
const indexAt = (r: number): number => Math.min(hype.trace.samples.length - 1, Math.max(0, Math.floor(r / hype.trace.interval_s)));

/** What `litmus` and the machine were doing at run time `r`, from the recorded trace, each averaged over the trailing second so a short pause between two documents does not show. */
export interface Moment {
  /** `litmus`'s share of the whole machine's processing, percent. */
  litmus: number;
  /** The machine's total utilization, percent; never below `litmus`. */
  machine: number;
  /** `litmus`'s working set, MB (the largest in the second). */
  memory: number;
  /** Memory the machine has in use, MB. */
  used: number;
}

export const momentAt = (r: number, window = 1): Moment => {
  const { samples, interval_s, cores } = hype.trace;
  const end = indexAt(r);
  const n = Math.max(1, Math.round(window / interval_s));
  const from = Math.max(0, end - n + 1);
  const rows = samples.slice(from, end + 1);
  const seconds = rows.length * interval_s;
  const litmus = (rows.reduce((a, s) => a + s[COL.cpuSeconds], 0) / (seconds * cores)) * 100;
  const machine = Math.max(litmus, rows.reduce((a, s) => a + s[COL.busy], 0) / rows.length);
  return { litmus, machine, memory: Math.max(...rows.map((s) => s[COL.ws])), used: samples[end][COL.used] };
};

/** The whole run's figures, for the end of the segment: averages for processing, peaks for memory. */
export const runMoment = (): Moment => ({
  litmus: hype.totals.cpu.litmus_share_of_machine * 100,
  machine: Math.max(hype.totals.cpu.litmus_share_of_machine * 100, hype.totals.cpu.machine_busy_mean * 100),
  memory: hype.totals.memory_mb.peak_working_set,
  used: hype.totals.memory_mb.machine_in_use_mean,
});

/** The documents finished by run time `r`, in the order they finished. */
export const doneBy = (r: number): HypeDoc[] => hype.documents.filter((d) => d.t_end <= r);
