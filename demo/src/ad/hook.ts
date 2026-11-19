// The first two seconds: the question, slammed in on the first beat, then the answer that is Litmus's difference, a strip
// whose pale band opens when it cannot tell. The strip is drawn from the engine's reading of a short sentence.

import { easeInOut, easeOutQuint, spring, span } from '../motion';
import { fonts } from '../palette';
import { strip } from '../draw/strip';
import { cueIn, isFull } from './clock';
import { headline, sample } from './data';
import { G, flash, ground, line, scaleText, safe, wrap } from './kit';
import type { Scene } from '../stage';

export const hook: Scene = (t, dur, L, ctx) => {
  const second = cueIn(ctx.cut, 'hook', 'hook_line');
  const bandAt = isFull(ctx.cut) ? cueIn(ctx.cut, 'hook', 'hook_band') : second + 0.5;
  const reading = sample('quick_short');
  const s0 = safe(L);
  const width = s0.x1 - s0.x0;

  const slam = easeOutQuint(span(t, 0, 0.3));
  const toTop = easeInOut(span(t, second - 0.1, second + 0.4));
  const titleScale = (1.14 - 0.14 * slam) * (1 - 0.42 * toTop);
  const titleY = L.wide ? 290 - 260 * toTop : 600 - 300 * toTop;
  const titleAlpha = 1 - 0.4 * toTop;
  const titleSize = L.wide ? 230 : 210;
  const title = `<div class="abs" style="left:0;top:${titleY}px;width:${L.W}px;text-align:center;font-family:${fonts.head};font-weight:640;font-size:${titleSize}px;line-height:1;color:${G.ink};opacity:${titleAlpha.toFixed(3)};transform:scale(${titleScale.toFixed(4)});transform-origin:50% 50%;white-space:nowrap">Is this ${scaleText('AI?')}</div>`;

  const wick = easeInOut(span(t, 0.1, 1.1));
  const markIn = spring(span(t, second + 0.3, second + 0.95), 6);
  const bandIn = easeInOut(span(t, bandAt, bandAt + 0.6));
  const labels = span(t, bandAt, bandAt + 0.4);
  const bar = L.wide ? { x: 340, y: 640, w: 1240, h: 72, labelSize: 36 } : { x: 110, y: 1230, w: 860, h: 70, labelSize: 34 };
  const s = strip({
    id: 'hook',
    ...bar,
    g: G,
    wick,
    position: reading.strip.position,
    markIn,
    band: [reading.strip.band_lo, reading.strip.band_hi],
    bandIn,
    labels,
  });
  const verdict = line(t, { text: headline(reading.set), x: s0.x0, w: width, y: L.wide ? 900 : 1420, size: L.wide ? 60 : 54, from: bandAt + 0.3, to: 99, color: G.mixedInk, weight: 560 });
  const answer = line(t, { text: 'Litmus tells you how sure it is.', x: s0.x0, w: width, y: L.wide ? 235 : 560, size: L.wide ? 108 : 118, from: second + 0.28, to: 99, font: 'head', weight: 600, gap: 0.05 });
  // The first frame is fully drawn (no dip in); the shot dips out like the others.
  return wrap(L, t, dur, `${ground(L, ctx.abs + t, 1 + 0.2 * slam)}${title}${answer}${s}${verdict}${flash(L, t, 0.1)}`, { in: false });
};
