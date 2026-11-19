// The cursive L drawing itself as one strip: blue through the loop and stem, purple through the lower loop,
// red along the tail, then the pale band. A soft light rides the pen tip while it draws.

import { mark } from '../content';
import { clamp, easeInOut, span } from '../motion';

/** Where each stroke draws within the mark's 0..1 progress. */
const spans: [number, number][] = [
  [0.0, 0.44],
  [0.4, 0.7],
  [0.66, 0.9],
  [0.86, 1.0],
];

let samples: { x: number; y: number }[][] | null = null;

/** Points along each path, measured once in the page (the film runs in a browser, so the DOM can measure). */
const pathSamples = (): { x: number; y: number }[][] => {
  if (samples) return samples;
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('style', 'position:absolute;left:-9999px;top:-9999px');
  document.body.appendChild(svg);
  samples = mark.paths.map((p) => {
    const el = document.createElementNS(ns, 'path');
    el.setAttribute('d', p.d);
    svg.appendChild(el);
    const len = el.getTotalLength();
    return Array.from({ length: 101 }, (_, i) => {
      const pt = el.getPointAtLength((len * i) / 100);
      return { x: pt.x, y: pt.y };
    });
  });
  svg.remove();
  return samples;
};

export interface MarkOpts {
  x: number;
  y: number;
  /** Height in pixels; the width follows the viewBox. */
  h: number;
  /** Drawing progress, 0 nothing to 1 complete. */
  k: number;
  /** Strength of the glow behind the finished mark, 0 to 1. */
  glow?: number;
  opacity?: number;
}

export const markDims = (h: number): { w: number; h: number } => {
  const [, , vw, vh] = mark.viewBox.split(/\s+/).map(Number);
  return { w: (h * vw) / vh, h };
};

export const drawMark = (o: MarkOpts): string => {
  const [vx, vy, vw, vh] = mark.viewBox.split(/\s+/).map(Number);
  const w = (o.h * vw) / vh;
  const pts = pathSamples();
  let tip = '';
  const strokes = mark.paths
    .map((p, i) => {
      const [a, b] = spans[i];
      const k = easeInOut(span(o.k, a, b));
      if (k <= 0) return '';
      if (p.band) {
        return `<path d="${p.d}" stroke="${p.stroke}" stroke-linecap="butt" opacity="${k.toFixed(3)}"/>`;
      }
      if (k < 1) {
        const pt = pts[i][Math.round(k * 100)];
        tip = `<circle cx="${pt.x.toFixed(2)}" cy="${pt.y.toFixed(2)}" r="16" fill="${p.stroke}" opacity="0.55" filter="url(#tipglow)"/>`;
      }
      return `<path d="${p.d}" stroke="${p.stroke}" pathLength="1" stroke-dasharray="1 2" stroke-dashoffset="${(1 - k).toFixed(4)}"/>`;
    })
    .join('');
  const glow = clamp(o.glow ?? 0);
  const glowLayer =
    glow > 0
      ? `<g fill="none" stroke-width="16" stroke-linecap="round" stroke-linejoin="round" filter="url(#markglow)" opacity="${(glow * 0.55).toFixed(3)}">${mark.paths
          .filter((p) => !p.band)
          .map((p) => `<path d="${p.d}" stroke="${p.stroke}"/>`)
          .join('')}</g>`
      : '';
  // The pen touches down softly rather than a full dot appearing on the first frame.
  const opacity = Math.min(o.opacity ?? 1, clamp(o.k * 30));
  return `<svg class="abs" style="left:${o.x}px;top:${o.y}px;overflow:visible;opacity:${opacity.toFixed(3)}" width="${w.toFixed(1)}" height="${o.h}" viewBox="${vx} ${vy} ${vw} ${vh}">
<defs>
  <filter id="tipglow" x="-200%" y="-200%" width="500%" height="500%"><feGaussianBlur stdDeviation="9"/></filter>
  <filter id="markglow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="14"/></filter>
</defs>
${glowLayer}
<g fill="none" stroke-width="16" stroke-linecap="round" stroke-linejoin="round">${strokes}</g>
${tip}
</svg>`;
};
