// The GPU comparison of the speed shot: for each detector, the share of time the GPU was busy while it read and the GPU memory
// of its own process, beside Litmus, which uses none, each row naming the GPU it was measured on. Every figure is read from
// data/ad/gpu_use.json (made by tools/merge_gpu_use.mjs from the output of `python -m litmus_benchmarks gpu-use`, measured
// with nvidia-smi); when that file is absent the page has no `window.__gpuUse` and `gpuUse()` is null, so the shot draws without
// this element and nothing is invented. What each detector read (the others truncate, Litmus reads every token) is in the
// file's basis, and nothing here says they read the whole documents.

import { brand, fonts } from '../palette';
import { clamp, easeOut, span } from '../motion';
import { esc } from '../draw/type';
import { G } from './kit';

export interface GpuDetector {
  detector: string;
  device: string;
  machine: { name: string };
  gpu_utilization_mean_percent: number | null;
  gpu_utilization_added_over_idle_percent: number | null;
  gpu_memory_process_peak_mib: number | null;
}

export interface GpuUse {
  schema: number;
  pilot: boolean;
  /** Set only by the layout fixture, never by a measurement: the element then stamps itself as fixture data. */
  fixture?: boolean;
  per_process_query_works: boolean;
  detectors: GpuDetector[];
}

declare global {
  interface Window {
    __gpuUse?: GpuUse | null;
  }
}

/** The detectors' names as the benchmark figures write them (`SHORT_NAMES` in benchmarks/src/litmus_benchmarks/palette.py). */
const NAMES: Record<string, string> = {
  'ModernBERT RAID+MAGE (Drayson)': 'ModernBERT (Drayson)',
  'OpenAI RoBERTa-base (GPT-2)': 'RoBERTa-base (OpenAI)',
  'RoBERTa ChatGPT (HC3)': 'RoBERTa (HC3)',
  'Litmus engine': 'Litmus',
};
const name = (detector: string): string => NAMES[detector] ?? detector;
/** "NVIDIA GH200 480GB" as "GH200": the vendor and the capacity are not what tells the machines apart. */
const gpuName = (machine: string): string => machine.replace(/^NVIDIA\s+/, '').replace(/\s+\d+\s*GB$/, '');

/** The measured GPU figures, or null when the measurement has not been made (no file). A file that is not a finished
 *  measurement is an error, never a quiet fallback: a pilot, a missing Litmus row, or a detector without its figures. */
export const gpuUse = (): GpuUse | null => {
  const g = window.__gpuUse;
  if (!g) return null;
  if (g.schema !== 1) throw new Error(`gpu_use.json has schema ${g.schema}, the scene reads 1`);
  if (g.pilot) throw new Error('gpu_use.json is a pilot run: measure with the full document set before rendering');
  const litmus = g.detectors.filter((d) => d.device === 'cpu');
  if (litmus.length !== 1) throw new Error(`gpu_use.json needs exactly one Litmus row on the CPU, found ${litmus.length}`);
  for (const d of g.detectors) {
    if (!d.machine?.name) throw new Error(`gpu_use.json names no machine for ${d.detector}`);
    if (d.device !== 'cpu' && (d.gpu_utilization_mean_percent === null || d.gpu_memory_process_peak_mib === null)) throw new Error(`gpu_use.json has no figures for ${d.detector}`);
  }
  if (g.detectors.length < 2) throw new Error('gpu_use.json holds no GPU detector to compare');
  return g;
};

const percent = (v: number): string => (v === 0 ? '0%' : `${v < 10 ? v.toFixed(1) : Math.round(v)}%`);
/** MiB from nvidia-smi as MB and GB on the ad's own convention (the process list's "MB" is MiB too: 1,024 to the GB). */
const memory = (mib: number): string => (mib >= 1024 ? `${(mib / 1024).toFixed(1)} GB` : `${Math.round(mib)} MB`);

export interface GpuBox {
  x: number;
  y: number;
  w: number;
  cols: number;
  rowH: number;
  /** Text size of a name; the figures under it and beside it follow. */
  size: number;
}

/** The panel's height for a box: a title and its rows (the detectors split over `cols` columns). */
export const gpuHeight = (g: GpuUse, box: GpuBox): number => 46 + Math.ceil(g.detectors.length / box.cols) * box.rowH + 10;

/** The panel: Litmus first, then the detectors by GPU busy time. A row is the name over the GPU it was measured on, a bar of the
 *  busy share, and that share over the process's peak GPU memory. The bars fill over `reveal` (0 to 1; the figures are the
 *  measured means, only their drawing is animated). */
export const gpuPanel = (g: GpuUse, box: GpuBox, reveal: number): string => {
  const litmus = g.detectors.find((d) => d.device === 'cpu') as GpuDetector;
  const others = g.detectors.filter((d) => d.device !== 'cpu').sort((a, b) => (b.gpu_utilization_mean_percent as number) - (a.gpu_utilization_mean_percent as number));
  const ordered = [litmus, ...others];
  const rows = Math.ceil(ordered.length / box.cols);
  const gap = 14;
  const cell = (box.w - 2 * gap - (box.cols - 1) * gap) / box.cols;
  const sub = Math.round(box.size * 0.74);
  const nameW = Math.round(box.size * 10.6);
  const figureW = Math.round(box.size * 4.1);
  const barW = cell - 16 - nameW - figureW - 2 * 12;
  const fill = easeOut(clamp(reveal));
  const cells = ordered.map((d, i) => {
    const col = Math.floor(i / rows);
    const row = i % rows;
    const x = gap + col * (cell + gap);
    const y = 46 + row * box.rowH;
    const isLitmus = d.device === 'cpu';
    const util = d.gpu_utilization_mean_percent ?? 0;
    const none = isLitmus && d.gpu_memory_process_peak_mib === 0 && Math.abs(d.gpu_utilization_added_over_idle_percent ?? 1) < 0.5;
    // Litmus shows what it added over the idle baseline when that is nothing; every other row shows its measured mean.
    const pct = percent(none ? 0 : util);
    const mem = none ? 'none' : memory(d.gpu_memory_process_peak_mib as number);
    const where = isLitmus ? 'CPU, one core' : gpuName(d.machine.name);
    const fillW = isLitmus ? 0 : (util / 100) * barW * fill;
    const hot = isLitmus ? `background:rgba(217,72,59,0.16);outline:2px solid ${brand.band};outline-offset:-1px;` : '';
    const h = box.rowH - 4;
    const ink = isLitmus ? G.ink : G.soft;
    const barTop = h / 2 - 6;
    return `<div style="position:absolute;left:${x}px;top:${y}px;width:${cell}px;height:${h}px;border-radius:12px;${hot}">
<div style="position:absolute;left:10px;top:4px;width:${nameW}px;font-family:${fonts.body};font-weight:${isLitmus ? 700 : 600};font-size:${box.size}px;line-height:1.15;color:${ink};white-space:nowrap">${esc(name(d.detector))}</div>
<div style="position:absolute;left:10px;top:${Math.round(4 + box.size * 1.1)}px;width:${nameW}px;font-family:${fonts.mono};font-size:${sub}px;line-height:1.15;color:${G.faint};white-space:nowrap">${esc(where)}</div>
<div style="position:absolute;left:${16 + nameW + 12}px;top:${barTop}px;width:${barW}px;height:12px;border-radius:6px;background:${G.ground}"></div>
<div style="position:absolute;left:${16 + nameW + 12}px;top:${barTop}px;width:${fillW.toFixed(1)}px;height:12px;border-radius:6px;background:${brand.mixed}"></div>
<div style="position:absolute;right:10px;top:4px;width:${figureW}px;text-align:right;font-family:${fonts.mono};font-weight:600;font-size:${box.size}px;line-height:1.15;color:${isLitmus ? G.ink : G.soft}">${pct}</div>
<div style="position:absolute;right:10px;top:${Math.round(4 + box.size * 1.1)}px;width:${figureW}px;text-align:right;font-family:${fonts.mono};font-size:${sub}px;line-height:1.15;color:${G.soft};white-space:nowrap">${mem}</div>
</div>`;
  });
  const stamp = g.fixture
    ? `<div style="position:absolute;left:0;top:0;right:0;bottom:0;border-radius:22px;border:3px dashed rgba(255,64,64,0.85);pointer-events:none"></div><div style="position:absolute;right:20px;top:10px;font-family:${fonts.mono};font-weight:700;font-size:20px;letter-spacing:0.08em;color:rgba(255,80,80,0.95)">FIXTURE, NOT MEASURED</div>`
    : '';
  return `<div class="abs" style="left:${box.x}px;top:${box.y}px;width:${box.w}px;height:${gpuHeight(g, box)}px;border-radius:22px;background:${G.surface};box-shadow:0 0 0 2px rgba(255,255,255,0.06),0 24px 60px rgba(0,0,0,0.35)">
<div style="position:absolute;left:22px;top:12px;font-family:${fonts.mono};font-size:20px;color:${G.faint}">GPU busy and GPU memory, by detector</div>
${cells.join('\n')}${stamp}
</div>`;
};

/** How far the bars have filled at scene time `t` for a panel that appears at `from`. */
export const gpuReveal = (t: number, from: number): number => span(t, from + 0.2, from + 1.1);
