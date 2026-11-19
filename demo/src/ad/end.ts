// The end card: the mark draws itself, the name, the address of the site (the most prominent line after the name), the
// first line (verbatim), free, private, and fast, the three doors, and two small lines. In the wide cut the mark, the
// name, the address, and the first line stand at the left and the pills, doors, and small lines at the right. No cut dips
// out at the end: the last frame is the whole card, address included, so a player that stops there shows it.

import { easeOutQuint, shown, span } from '../motion';
import { fonts } from '../palette';
import { drawMark, markDims } from '../draw/mark';
import { esc } from '../draw/type';
import { cueIn, isFull } from './clock';
import { G, flash, ground, line, wrap } from './kit';
import type { Scene } from '../stage';

const FIRST_LINE = 'Litmus is a free detector of AI written text that says how sure it is';

/** The site's address as people type it: no scheme, no trailing slash (served at https://lhallee.github.io/litmus/, release/RELEASING.md step 5). */
const ADDRESS = 'lhallee.github.io/litmus';

const DOORS: [string, string][] = [
  ['The desktop app', 'Select text anywhere, press Ctrl+Alt+L'],
  ['In your browser', 'The text never leaves the page'],
  ['The contribute site', 'Share a text to help improve Litmus'],
];

export const end: Scene = (t, dur, L, ctx) => {
  const nameAt = cueIn(ctx.cut, 'end', 'end_name');
  const doorsAt = isFull(ctx.cut) ? cueIn(ctx.cut, 'end', 'end_doors') : Number.POSITIVE_INFINITY;
  const wide = L.wide;
  const markH = 190;
  const dims = markDims(markH);
  const mark = drawMark({ x: wide ? 96 : (1080 - dims.w) / 2, y: wide ? 100 : 235, h: markH, k: easeOutQuint(span(t, 0.05, nameAt + 0.2)), glow: shown(t, 0.5, dur, 0.5, 0.01) });
  const left = wide ? { x: 96, w: 900, align: 'left' as const } : {};
  const name = line(t, { ...left, text: 'Litmus', y: wide ? 300 : 430, size: wide ? 170 : 160, from: nameAt - 0.1, to: 99, font: 'head', weight: 600 });
  // The address comes right after the name and stays to the last frame; it is the largest line after the name.
  const address = line(t, { ...left, text: ADDRESS, y: wide ? 495 : 636, size: wide ? 78 : 76, from: nameAt + 0.15, to: 99, weight: 650, color: G.ink, gap: 0.02 });
  const first = line(t, { ...left, text: FIRST_LINE, y: wide ? 650 : 778, size: wide ? 44 : 46, from: nameAt + 0.45, to: 99, weight: 520, color: G.soft, gap: 0.035, lineHeight: 1.2 });
  // Three pills in the scale's order, blue, purple, red: 240 px each with 24 px between, centered in the tall frame and at the right column's edge in the wide one.
  const inks = [G.humanInk, G.mixedInk, G.aiInk];
  const tags = ['Free', 'Private', 'Fast'].map((w, i) => {
    const a = shown(t, nameAt + 0.9 + i * 0.15, 99, 0.2, 0.01);
    const x = (wide ? 1020 : 156) + i * 264;
    const y = wide ? 190 : 925;
    return `<div class="abs" style="left:${x}px;top:${y}px;width:240px;text-align:center;opacity:${a.toFixed(3)};font-family:${fonts.body};font-weight:650;font-size:50px;color:${inks[i]};border:3px solid ${inks[i]};border-radius:999px;padding:12px 0">${w}</div>`;
  });
  const doors = DOORS.map(([head, sub], i) => {
    const from = doorsAt + i * 0.25;
    const a = shown(t, from, 99, 0.25, 0.01);
    const k = easeOutQuint(span(t, from, from + 0.5));
    const box = wide
      ? `left:1020px;top:${360 + i * 140}px;width:804px;display:block`
      : `left:90px;top:${1055 + i * 100}px;width:900px;display:flex;align-items:baseline;gap:22px`;
    return `<div class="abs" style="${box};opacity:${a.toFixed(3)};transform:translateY(${((1 - k) * 24).toFixed(1)}px);border-bottom:2px solid ${G.rule};padding-bottom:12px"><div style="font-family:${fonts.body};font-weight:650;font-size:${wide ? 46 : 42}px;color:${G.ink};white-space:nowrap;display:inline-block">${esc(head)}</div><span style="${wide ? 'display:block;margin-top:4px;' : ''}font-family:${fonts.body};font-size:${wide ? 30 : 28}px;color:${G.soft}">${esc(sub)}</span></div>`;
  });
  // Two lines, each on its own line.
  const small = ['Open source', 'Free for noncommercial use']
    .map((text, i) => line(t, { ...(wide ? { x: 1020, w: 804, align: 'left' as const } : {}), text, y: (wide ? 830 : 1360) + i * (wide ? 42 : 44), size: wide ? 28 : 30, from: doorsAt + 0.9, to: 99, weight: 420, color: G.soft, gap: 0.02 }))
    .join('');
  return wrap(L, t, dur, `${ground(L, ctx.abs + t, 1.1)}${mark}${name}${address}${first}${tags.join('')}${doors.join('')}${small}${flash(L, t, 0.25)}`, { out: false });
};
