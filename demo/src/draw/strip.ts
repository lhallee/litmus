// The litmus strip as the site draws it (blue human through purple to red AI, a mark at the chance of AI, a dashed
// pale band over that chance's range), plus the film's one addition: the color wicks into unreacted paper from the
// left, as it does into a real strip of litmus paper.

import { clamp, lerp } from '../motion';
import { brand, fonts, type Ground } from '../palette';
import { esc } from './type';

export interface StripState {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  g: Ground;
  /** How far the color has wicked in, 0 none to 1 the whole strip. */
  wick: number;
  position: number;
  /** The mark's arrival, 0 above and invisible to 1 in place (a spring value, so it may pass 1). */
  markIn: number;
  band: [number, number] | null;
  /** The band opening from the mark out to its ends, 0 to 1. */
  bandIn: number;
  /** Opacity of the labels under the strip's ends, Human and AI. */
  labels: number;
  labelSize?: number;
  /** The words under the left end, the middle, and the right end; the middle is empty as in the app. */
  labelText?: [string, string, string];
}

export const strip = (s: StripState): string => {
  const { w, h, g } = s;
  const pad = h * 1.2;
  const r = h / 2;
  const labelSize = s.labelSize ?? Math.max(16, h * 0.62);
  const height = h + pad * 2 + labelSize * 1.6;
  const wickW = clamp(s.wick, 0, 1.05) * (w + 60) - 30;

  const band = (() => {
    if (!s.band || s.bandIn <= 0) return '';
    const e = clamp(s.bandIn);
    const lo = lerp(s.position, s.band[0], e);
    const hi = lerp(s.position, s.band[1], e);
    const ext = h * 0.22;
    const bw = Math.max(h * 0.9, (hi - lo) * w + ext * 2);
    const bx = ((lo + hi) / 2) * w - bw / 2;
    const rx = (h + ext * 2) * 0.36;
    const sw = Math.max(2, h * 0.075);
    return `<rect x="${bx.toFixed(2)}" y="${(pad - ext).toFixed(2)}" width="${bw.toFixed(2)}" height="${(h + ext * 2).toFixed(2)}" rx="${rx.toFixed(2)}" fill="${g.bandWash}" stroke="${g.bandEdge}" stroke-width="${sw.toFixed(2)}" stroke-dasharray="${(sw * 4).toFixed(1)} ${(sw * 3).toFixed(1)}" opacity="${Math.min(1, e * 1.6).toFixed(3)}"/>`;
  })();

  const mark = (() => {
    if (s.markIn <= 0) return '';
    const mw = Math.max(5, h * 0.24);
    const over = h * 0.42;
    const drop = (1 - s.markIn) * -h * 2.6;
    const mx = s.position * w - mw / 2;
    const alpha = clamp(s.markIn * 3);
    return `<g transform="translate(0 ${drop.toFixed(2)})" opacity="${alpha.toFixed(3)}"><rect x="${mx.toFixed(2)}" y="${(pad - over).toFixed(2)}" width="${mw.toFixed(2)}" height="${(h + over * 2).toFixed(2)}" rx="${(mw / 2).toFixed(2)}" fill="${g.ink}" stroke="${g.markRing}" stroke-width="${Math.max(2, h * 0.09).toFixed(2)}" paint-order="stroke"/></g>`;
  })();

  const labelText = s.labelText ?? ['Human', '', 'AI'];
  const labels =
    s.labels > 0
      ? `<g font-family="${fonts.body}" font-size="${labelSize}" font-weight="500" opacity="${s.labels.toFixed(3)}">` +
        `<text x="0" y="${pad * 2 + h + labelSize * 0.45}" fill="${g.humanInk}">${esc(labelText[0])}</text>` +
        `<text x="${w / 2}" y="${pad * 2 + h + labelSize * 0.45}" fill="${g.mixedInk}" text-anchor="middle">${esc(labelText[1])}</text>` +
        `<text x="${w}" y="${pad * 2 + h + labelSize * 0.45}" fill="${g.aiInk}" text-anchor="end">${esc(labelText[2])}</text></g>`
      : '';

  const id = s.id;
  return `<svg class="abs" style="left:${s.x}px;top:${s.y - pad}px;overflow:visible" width="${w}" height="${height}" viewBox="0 0 ${w} ${height}">
<defs>
  <linearGradient id="grad-${id}" x1="0" x2="1" y1="0" y2="0">
    <stop offset="0" stop-color="${brand.human}"/><stop offset="0.5" stop-color="${brand.mixed}"/><stop offset="1" stop-color="${brand.ai}"/>
  </linearGradient>
  <filter id="wet-${id}" x="-10%" y="-60%" width="120%" height="220%">
    <feTurbulence type="fractalNoise" baseFrequency="0.012 0.09" numOctaves="2" seed="7" result="n"/>
    <feDisplacementMap in="SourceGraphic" in2="n" scale="${(h * 1.1).toFixed(1)}" xChannelSelector="R" yChannelSelector="G" result="d"/>
    <feGaussianBlur in="d" stdDeviation="${(h * 0.06).toFixed(2)}"/>
  </filter>
  <mask id="mask-${id}" maskUnits="userSpaceOnUse" x="-60" y="${pad - h}" width="${w + 120}" height="${h * 3}">
    <rect x="-60" y="${pad - h}" width="${Math.max(0, wickW + 60).toFixed(2)}" height="${h * 3}" fill="#fff" filter="url(#wet-${id})"/>
  </mask>
  <clipPath id="clip-${id}"><rect x="0" y="${pad}" width="${w}" height="${h}" rx="${r}"/></clipPath>
</defs>
<g clip-path="url(#clip-${id})">
  <rect x="0" y="${pad}" width="${w}" height="${h}" fill="${g.unreacted}"/>
  <rect x="0" y="${pad}" width="${w}" height="${h}" fill="url(#grad-${id})" mask="url(#mask-${id})"/>
</g>
${band}
${mark}
${labels}
</svg>`;
};

/** The vertical offset from a strip's `y` to the top of its labels, for laying out what comes under it. */
export const stripDepth = (h: number, labelSize?: number): number => h + h * 1.2 + (labelSize ?? Math.max(16, h * 0.62)) * 1.6;
