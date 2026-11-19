// The speed segment: many long documents, half written by people and half by a language model, read one after another by
// one process pinned to one core, with a process list that shows what that costs (the processing it takes as a share of the
// whole machine, and its memory) drawn from the recorded trace, and a running tally of what each set reads as. Everything
// here is replayed from data/ad/hype_run.json at a speed that file's own times fix; at the end the figures are the run's own
// average and peak, not the last instant's.

import { brand, fonts } from '../palette';
import { clamp, shown } from '../motion';
import { esc } from '../draw/type';
import { PACE, cueIn } from './clock';
import { doneBy, hype, momentAt, runMoment, type SetName, type Verdict } from './data';
import { gpuHeight, gpuPanel, gpuReveal, gpuUse, type GpuBox } from './gpu';
import { G, flash, ground, line, safe, wrap } from './kit';
import type { Layout, Scene } from '../stage';

const VERDICT_COLOR: Record<Verdict, string> = { human: brand.human, uncertain: brand.band, mixed: brand.mixed, ai: brand.ai, unlike: '#8A8290' };
const WINDOW_S = 12;
const COLS = 15;
const GAP = 6;

const clock = (s: number): string => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
const thousands = (n: number): string => Math.round(n).toLocaleString('en-US');
const percent = (v: number): string => `${v < 10 ? v.toFixed(1) : Math.round(v)}%`;

/** One set's column: its name, a tile for every document, and what the documents read so far read as. */
const column = (set: SetName, x: number, y: number, width: number, done: ReturnType<typeof doneBy>): string => {
  // The tiles fill the column's width: 15 across.
  const TILE = Math.floor((width - 14 * GAP) / COLS);
  const total = hype.tally[set].documents;
  const mine = done.filter((d) => d.set === set);
  const count = (v: Verdict[]): number => mine.filter((d) => v.includes(d.verdict)).length;
  const share = (v: Verdict[]): number => (mine.length === 0 ? 0 : Math.round((count(v) / mine.length) * 100));
  const rows = Math.ceil(total / COLS);
  const tiles = Array.from({ length: total }, (_, i) => {
    const d = mine[i];
    return `<div style="position:absolute;left:${(i % COLS) * (TILE + GAP)}px;top:${Math.floor(i / COLS) * (TILE + GAP)}px;width:${TILE}px;height:${TILE}px;border-radius:7px;background:${d ? VERDICT_COLOR[d.verdict] : '#2A2630'}"></div>`;
  }).join('');
  const chips: { value: number; label: string; color: string }[] = [
    { value: share(['human']), label: 'read as human', color: G.humanInk },
    { value: share(['ai']), label: 'read as AI', color: G.aiInk },
    { value: share(['uncertain', 'mixed']), label: 'uncertain or mixed', color: G.mixedInk },
  ];
  if (hype.tally[set].unlike > 0) chips.push({ value: share(['unlike']), label: 'unlike reference', color: G.faint });
  const chip = (c: (typeof chips)[number]): string =>
    `<div style="flex:1"><div style="font-family:${fonts.head};font-weight:620;font-size:50px;line-height:1;color:${c.color}">${c.value}<span style="font-size:28px">%</span></div><div style="font-family:${fonts.body};font-weight:560;font-size:19px;color:${G.ink};margin-top:2px">${esc(c.label)}</div></div>`;
  const name = set === 'human' ? 'Written by people' : 'Written by AI';
  return `<div class="abs" style="left:${x}px;top:${y}px;width:${width}px">
<div style="font-family:${fonts.mono};font-size:25px;letter-spacing:0.1em;text-transform:uppercase;color:${G.ink}">${esc(name)}</div>
<div style="position:relative;margin-top:14px;height:${rows * (TILE + GAP) - GAP}px">${tiles}</div>
<div style="display:flex;gap:8px;margin-top:20px">${chips.map(chip).join('')}</div>
</div>`;
};

/** The process list: `litmus` highlighted, everything else as one row, and below them the processing and memory drawn from the trace. */
const monitor = (r: number, final: boolean, lock: number, box: { x: number; y: number; w: number }, compact = false): string => {
  // Compact (the tall cut with the GPU panel): the rows sit closer and the two traces are shorter, 58 px less in all.
  const pos = compact ? { r1: 44, r2: 124, gLabel: 216, gTop: 240, gh: 52, mLabel: 304, mTrack: 330, mFill: 326, h: 356 } : { r1: 48, r2: 138, gLabel: 232, gTop: 258, gh: 70, mLabel: 340, mTrack: 368, mFill: 364, h: 414 };
  const moment = final ? runMoment() : momentAt(r);
  const other = Math.max(0, moment.machine - moment.litmus);
  const memOther = Math.max(0, moment.used - moment.memory);
  const { x: x0, y: y0, w } = box;
  const glow = lock > 0 ? `text-shadow:0 0 ${(36 * lock).toFixed(0)}px rgba(238,232,220,${(0.5 * lock).toFixed(3)});` : '';
  const cpuLabel = final ? 'CPU, run average' : 'CPU';
  const memLabel = final ? 'Memory, run peak' : 'Memory';
  const head = `<div style="position:absolute;left:28px;top:14px;font-family:${fonts.mono};font-size:20px;color:${G.faint}">Processes</div>
<div style="position:absolute;right:${w - 640}px;top:14px;font-family:${fonts.mono};font-size:20px;color:${G.faint};white-space:nowrap">${cpuLabel}</div>
<div style="position:absolute;right:28px;top:14px;font-family:${fonts.mono};font-size:20px;color:${G.faint};white-space:nowrap">${memLabel}</div>`;
  const row = (y: number, hot: boolean, name: string, sub: string, cpu: number, mem: number): string =>
    `<div style="position:absolute;left:12px;top:${y}px;width:${w - 24}px;height:80px;border-radius:16px;${hot ? `background:rgba(217,72,59,0.16);outline:3px solid ${brand.band};outline-offset:-1px` : ''}">
<div style="position:absolute;left:16px;top:${sub ? 10 : 20}px;font-family:${fonts.body};font-weight:${hot ? 700 : 560};font-size:${hot ? 38 : 33}px;color:${hot ? G.ink : G.soft};line-height:1.1">${esc(name)}</div>
${sub ? `<div style="position:absolute;left:16px;top:52px;font-family:${fonts.mono};font-size:19px;color:${G.soft}">${esc(sub)}</div>` : ''}
<div style="position:absolute;right:${w - 24 - 628}px;top:12px;font-family:${fonts.head};font-weight:620;font-size:${hot ? 54 : 44}px;color:${hot ? G.aiInk : G.soft};line-height:1.1;${hot ? glow : ''}">${percent(cpu)}</div>
<div style="position:absolute;right:16px;top:12px;font-family:${fonts.head};font-weight:620;font-size:${hot ? 54 : 44}px;color:${hot ? G.ink : G.soft};line-height:1.1;${hot ? glow : ''}">${thousands(mem)}<span style="font-size:${hot ? 28 : 24}px"> MB</span></div>
</div>`;

  // The processing over the last WINDOW_S seconds of the run: everything else above, `litmus` below, the whole machine at 100%.
  const gw = w - 56;
  const gh = pos.gh;
  const { samples, interval_s } = hype.trace;
  const end = Math.min(samples.length - 1, Math.floor(r / interval_s));
  const span = Math.round(WINDOW_S / interval_s);
  const top: string[] = [];
  const low: string[] = [];
  for (let s = Math.max(0, end - span); s <= end; s++) {
    const m = momentAt(s * interval_s);
    const px = ((s - Math.max(0, end - span)) / span) * gw;
    top.push(`${px.toFixed(1)},${(gh - (Math.max(m.machine, m.litmus) / 100) * gh).toFixed(1)}`);
    low.push(`${px.toFixed(1)},${(gh - (m.litmus / 100) * gh).toFixed(1)}`);
  }
  const lastX = top.length > 0 ? top[top.length - 1].split(',')[0] : '0';
  const area = (pts: string[], fill: string, stroke: string): string => (pts.length > 1 ? `<polygon points="0,${gh} ${pts.join(' ')} ${lastX},${gh}" fill="${fill}"/><polyline points="${pts.join(' ')}" fill="none" stroke="${stroke}" stroke-width="2.5" stroke-linejoin="round"/>` : '');
  const graph = `<div style="position:absolute;left:28px;top:${pos.gLabel}px;font-family:${fonts.mono};font-size:18px;color:${G.faint}">processing, whole machine</div>
<svg style="position:absolute;left:28px;top:${pos.gTop}px;overflow:hidden;border-radius:10px;background:${G.ground}" width="${gw}" height="${gh}" viewBox="0 0 ${gw} ${gh}">
<line x1="0" x2="${gw}" y1="${gh / 2}" y2="${gh / 2}" stroke="rgba(255,255,255,0.07)"/>${area(top, 'rgba(142,91,201,0.30)', brand.mixed)}${area(low, 'rgba(217,72,59,0.55)', brand.ai)}</svg>`;

  // The memory the machine has in use, as one bar; `litmus`'s share of it is the red mark at the left.
  const total = hype.basis.machine.memory_mb;
  const usedW = clamp(moment.used / total) * gw;
  const litW = Math.max(4, clamp(moment.memory / total) * gw);
  const bar = `<div style="position:absolute;left:28px;top:${pos.mLabel}px;font-family:${fonts.mono};font-size:18px;color:${G.faint}">memory, whole machine</div>
<div style="position:absolute;left:28px;top:${pos.mTrack}px;width:${gw}px;height:14px;border-radius:7px;background:${G.ground}"></div>
<div style="position:absolute;left:28px;top:${pos.mTrack}px;width:${usedW.toFixed(1)}px;height:14px;border-radius:7px;background:rgba(142,91,201,0.55)"></div>
<div style="position:absolute;left:28px;top:${pos.mFill}px;width:${litW.toFixed(1)}px;height:22px;border-radius:3px;background:${brand.ai}"></div>`;

  return `<div class="abs" style="left:${x0}px;top:${y0}px;width:${w}px;height:${pos.h}px;border-radius:26px;background:${G.surface};box-shadow:0 0 0 2px rgba(255,255,255,0.06),0 24px 60px rgba(0,0,0,0.35)">
${head}
${row(pos.r1, true, 'Litmus', 'pinned to one core', moment.litmus, moment.memory)}
${row(pos.r2, false, 'Other tools', '', other, memOther)}
${graph}${bar}
</div>`;
};

export const hypeScene: Scene = (t, dur, L: Layout, ctx) => {
  const playFrom = cueIn(ctx.cut, 'hype', 'hype_run');
  const lockAt = cueIn(ctx.cut, 'hype', 'hype_lock');
  const total = hype.documents[hype.documents.length - 1].t_end;
  const k = clamp((t - playFrom) / (lockAt - playFrom));
  const r = k * total;
  // The recorded run shown in (lockAt - playFrom) scene seconds, each PACE real seconds long: the factor on screen is real time.
  const speedUp = total / ((lockAt - playFrom) * PACE);
  const done = doneBy(r);
  const final = t >= lockAt;
  const lock = final ? Math.max(0, 1 - (t - lockAt) / 1.0) : 0;
  const tokens = done.reduce((a, d) => a + d.tokens, 0);

  // Tall: the facts above, the process list, then the two sets side by side. Wide: the facts and the counts at the left,
  // the process list and the two sets at the right. With the measured GPU figures (data/ad/gpu_use.json) the GPU panel
  // joins: under the two sets in the tall cut, whose facts and process list then sit a little tighter, and under the facts
  // at the left in the wide cut, whose counts move under the sets. Without the file the shot is drawn as it was.
  const wide = L.wide;
  const gpu = gpuUse();
  const s0 = safe(L);
  const left = wide ? { x: s0.x0, w: 800, align: 'left' as const } : {};
  const y = wide ? (gpu ? [120, 232, 344] : [240, 360, 480]) : gpu ? [252, 340, 428] : [268, 366, 464];
  const size = wide ? [96, 96, 70] : gpu ? [84, 84, 60] : [90, 90, 66];
  const title =
    line(t, { ...left, text: 'One CPU core.', y: y[0], size: size[0], from: 0.1, to: 99, font: 'head', gap: 0.05 }) +
    line(t, { ...left, text: `${Math.ceil(hype.totals.memory_mb.peak_working_set)} MB of memory.`, y: y[1], size: size[1], from: 0.45, to: 99, font: 'head', gap: 0.05 }) +
    line(t, { ...left, text: `${thousands(hype.totals.batch.tokens_per_second)} tokens a second.`, y: y[2], size: size[2], from: 0.8, to: 99, font: 'head', gap: 0.05, color: G.soft });
  const ready = shown(t, 0.9, 99, 0.35, 0.01);
  const sets = wide
    ? { monitor: { x: 930, y: 140, w: 894 }, human: [930, 610], ai: [1392, 610], col: 432 }
    : gpu
      ? { monitor: { x: 70, y: 520, w: 940 }, human: [70, 900], ai: [566, 900], col: 450 }
      : { monitor: { x: 70, y: 606, w: 940 }, human: [70, 1052], ai: [566, 1052], col: 450 };
  const panel = `<div style="opacity:${ready.toFixed(3)}">${monitor(r, final, lock, sets.monitor, gpu !== null && !wide)}${column('human', sets.human[0], sets.human[1], sets.col, done)}${column('ai', sets.ai[0], sets.ai[1], sets.col, done)}</div>`;

  let gpuHtml = '';
  let footY = wide ? 660 : 1424;
  if (gpu) {
    const box: GpuBox = wide ? { x: 96, y: 470, w: 784, cols: 1, rowH: 66, size: 27 } : { x: 70, y: 1190, w: 940, cols: 2, rowH: 62, size: 22 };
    const appears = playFrom + 0.6;
    const a = shown(t, appears, 99, 0.35, 0.01);
    gpuHtml = `<div style="opacity:${a.toFixed(3)}">${gpuPanel(gpu, box, gpuReveal(t, appears))}</div>`;
    footY = wide ? 960 : box.y + gpuHeight(gpu, box) + 16;
  }
  const foot = wide && !gpu
    ? 'left:96px;top:660px;width:800px;display:flex;flex-direction:column;gap:12px;font-size:30px'
    : wide
      ? `left:930px;top:${footY}px;width:894px;display:flex;justify-content:space-between;font-size:24px`
      : `left:70px;top:${footY}px;width:940px;display:flex;justify-content:space-between;font-size:22px`;
  const footer = `<div class="abs" style="${foot};font-family:${fonts.mono};color:${G.soft};opacity:${ready.toFixed(3)}"><span>${thousands(tokens)} tokens read</span><span>${clock(r)} of ${clock(total)} &nbsp;&middot;&nbsp; shown at ×${Math.round(speedUp)}</span></div>`;
  return wrap(L, t, dur, `${ground(L, ctx.abs + t, 1.1)}${title}${panel}${gpuHtml}${footer}${flash(L, t, 0.2)}`);
};
