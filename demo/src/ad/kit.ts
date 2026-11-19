// Shared pieces of the ad's scenes: the ground, captions in the safe area, and small labels. The frame is 1080 by
// 1920; text stays inside x 70 to 1010 and, for anything that must be read, y 250 to 1480, clear of the interface the
// platforms draw over the top and bottom of a vertical video.

import { clamp, easeOut, easeOutQuint, shown, span } from '../motion';
import { brand, dark, fonts } from '../palette';
import { esc, words } from '../draw/type';
import type { Layout } from '../stage';
import { pulse } from './clock';

export const G = dark;
export const SAFE = { x0: 70, x1: 1010, y0: 250, y1: 1480 };

/** The wide cut has no platform interface over it, so it keeps an even margin: 96 px at the sides, 60 px above and below. */
export const SAFE_WIDE = { x0: 96, x1: 1824, y0: 60, y1: 1020 };
export const safe = (L: Layout): typeof SAFE => (L.wide ? SAFE_WIDE : SAFE);

/** The ground: three soft washes in the scale's colors, breathing a little with the kick. */
export const ground = (L: Layout, abs: number, tint = 1): string => {
  const p = pulse(abs, 6);
  const washes = L.wide
    ? ['1300px 760px at 32% 18%', '1300px 800px at 6% 100%', '1200px 760px at 96% 90%']
    : ['900px 760px at 50% 24%', '1000px 900px at 14% 92%', '900px 800px at 92% 78%'];
  return `<div class="abs" style="left:0;top:0;width:${L.W}px;height:${L.H}px;background:
    radial-gradient(${washes[0]}, rgba(142,91,201,${(0.2 * tint + 0.04 * p).toFixed(3)}), transparent 70%),
    radial-gradient(${washes[1]}, rgba(47,111,219,${(0.16 * tint).toFixed(3)}), transparent 70%),
    radial-gradient(${washes[2]}, rgba(217,72,59,${(0.11 * tint).toFixed(3)}), transparent 70%)"></div>`;
};

export interface Line {
  text: string;
  y: number;
  size: number;
  from: number;
  to: number;
  font?: 'head' | 'body';
  weight?: number;
  color?: string;
  gap?: number;
  align?: 'center' | 'left';
  x?: number;
  w?: number;
  lineHeight?: number;
}

/** A caption that arrives word by word at `from` and leaves at `to`, in the safe area. */
export const line = (t: number, o: Line): string => {
  const alpha = shown(t, o.from, o.to, 0.01, 0.3);
  if (alpha <= 0) return '';
  const family = o.font === 'head' ? fonts.head : fonts.body;
  const style = [
    `left:${o.x ?? SAFE.x0}px`,
    `top:${o.y}px`,
    `width:${o.w ?? SAFE.x1 - SAFE.x0}px`,
    `font-family:${family}`,
    `font-size:${o.size}px`,
    `font-weight:${o.weight ?? (o.font === 'head' ? 600 : 560)}`,
    `line-height:${o.lineHeight ?? 1.14}`,
    `color:${o.color ?? G.ink}`,
    `text-align:${o.align ?? 'center'}`,
    `opacity:${alpha.toFixed(3)}`,
    'text-wrap:balance',
  ].join(';');
  return `<div class="abs" style="${style}">${words(o.text, t, { start: o.from, gap: o.gap ?? 0.04, dur: 0.55, rise: 0.22, blur: 5 })}</div>`;
};

/** Text filled with the litmus scale, blue through purple to red. */
export const scaleText = (text: string): string =>
  `<span style="background:linear-gradient(90deg,${brand.human},${brand.mixed} 55%,${brand.ai});-webkit-background-clip:text;background-clip:text;color:transparent">${esc(text)}</span>`;

/** A small pill, for the labels laid over footage and for the end card's doors. */
export const pill = (text: string, x: number, y: number, t: number, from: number, to: number, color = G.ink, size = 34, anchor: 'left' | 'center' = 'left'): string => {
  const alpha = shown(t, from, to, 0.28, 0.3);
  if (alpha <= 0) return '';
  const k = easeOutQuint(span(t, from, from + 0.4));
  const shift = anchor === 'center' ? 'translateX(-50%)' : '';
  return `<div class="abs" style="left:${x}px;top:${y}px;transform:${shift} translateY(${((1 - k) * 14).toFixed(1)}px);opacity:${alpha.toFixed(3)};font-family:${fonts.body};font-weight:600;font-size:${size}px;color:${color};background:rgba(32,29,37,0.92);border:2px solid ${color};border-radius:999px;padding:${size * 0.28}px ${size * 0.7}px;white-space:nowrap;box-shadow:0 10px 30px rgba(0,0,0,0.4)">${esc(text)}</div>`;
};

/** A white flash on the beat a shot cuts in, fading over ${FLASH_S} scene seconds. */
const FLASH_S = 0.3;
export const flash = (L: Layout, t: number, strength = 0.18): string => {
  const k = 1 - easeOut(span(t, 0, FLASH_S));
  return k <= 0 ? '' : `<div class="abs" style="left:0;top:0;width:${L.W}px;height:${L.H}px;background:#fff;opacity:${(k * strength).toFixed(3)};pointer-events:none"></div>`;
};

/** The dip through dark at a cut: a shot's content fades in over its first ${DIP_S} scene seconds and out over its last, so a cut does not pop. */
const DIP_S = 0.24;
export const wrap = (L: Layout, t: number, dur: number, inner: string, dip: { in?: boolean; out?: boolean } = {}): string => {
  const a = clamp(Math.min(dip.in === false ? 1 : t / DIP_S, dip.out === false ? 1 : (dur - t) / DIP_S));
  return `<div class="abs" style="left:0;top:0;width:${L.W}px;height:${L.H}px;opacity:${(0.2 + 0.8 * a).toFixed(3)}">${inner}</div>`;
};
