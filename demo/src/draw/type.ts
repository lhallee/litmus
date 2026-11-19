// Typography: words that arrive one after another, captions, section kickers, and labels.

import { easeOut, easeOutQuint, shown, span } from '../motion';
import { fonts, type Ground } from '../palette';

export const esc = (s: string): string =>
  s.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');

let measureCtx: CanvasRenderingContext2D | null = null;

/** Width in pixels of `text` set in a CSS font shorthand; fonts are loaded before the first frame. */
export const textWidth = (text: string, font: string): number => {
  measureCtx ??= document.createElement('canvas').getContext('2d');
  if (!measureCtx) throw new Error('no 2D canvas to measure text');
  measureCtx.font = font;
  return measureCtx.measureText(text).width;
};

export interface Reveal {
  start: number;
  /** Seconds between one word's start and the next. */
  gap?: number;
  /** Seconds a word takes to arrive. */
  dur?: number;
  /** Rise in em as a word arrives. */
  rise?: number;
  blur?: number;
}

/** Words of `text` arriving one by one from `start`, each rising and sharpening into place. */
export const words = (text: string, t: number, r: Reveal): string => {
  const gap = r.gap ?? 0.05;
  const dur = r.dur ?? 0.6;
  const rise = r.rise ?? 0.32;
  const blur = r.blur ?? 6;
  return text
    .split(' ')
    .map((w, i) => {
      const k = easeOutQuint(span(t, r.start + i * gap, r.start + i * gap + dur));
      if (k >= 1) return `<span class="w">${esc(w)}</span>`;
      const style = `opacity:${k.toFixed(3)};transform:translateY(${((1 - k) * rise).toFixed(3)}em);filter:blur(${((1 - k) * blur).toFixed(2)}px)`;
      return `<span class="w" style="${style}">${esc(w)}</span>`;
    })
    .join(' ');
};

/** Seconds `words` needs to bring in every word of `text`. */
export const wordsTime = (text: string, r: Reveal): number => text.split(' ').length * (r.gap ?? 0.05) + (r.dur ?? 0.6);

export interface CaptionOpts {
  x: number;
  y: number;
  w: number;
  size: number;
  align?: 'left' | 'center';
  font?: 'head' | 'body';
  weight?: number;
  color: string;
  from: number;
  to: number;
  gap?: number;
  lineHeight?: number;
}

/** A line of words that arrives at `from` and fades by `to`. */
export const caption = (text: string, t: number, o: CaptionOpts): string => {
  const alpha = shown(t, o.from, o.to, 0.01, 0.45);
  if (alpha <= 0) return '';
  const family = o.font === 'head' ? fonts.head : fonts.body;
  const style = [
    `left:${o.x}px`,
    `top:${o.y}px`,
    `width:${o.w}px`,
    `font-family:${family}`,
    `font-size:${o.size}px`,
    `font-weight:${o.weight ?? (o.font === 'head' ? 560 : 420)}`,
    `line-height:${o.lineHeight ?? 1.25}`,
    `color:${o.color}`,
    `text-align:${o.align ?? 'center'}`,
    `opacity:${alpha.toFixed(3)}`,
  ].join(';');
  return `<div class="abs cap" style="${style}">${words(text, t, { start: o.from, gap: o.gap ?? 0.045 })}</div>`;
};

/** The numbered section label in a scene's top left: a colored rule, a number, and a word. */
export const kicker = (n: string, label: string, t: number, g: Ground, x: number, y: number, size = 24): string => {
  const k = easeOut(span(t, 0.1, 0.9));
  const alpha = Math.min(k, 1);
  const rule = `<span style="display:inline-block;width:${(56 * k).toFixed(1)}px;height:3px;border-radius:2px;background:linear-gradient(90deg,#2F6FDB,#8E5BC9,#D9483B);vertical-align:middle;margin-right:18px"></span>`;
  return `<div class="abs" style="left:${x}px;top:${y}px;font-family:${fonts.mono};font-size:${size}px;letter-spacing:0.14em;text-transform:uppercase;color:${g.soft};opacity:${alpha.toFixed(3)};white-space:nowrap">${rule}<span style="color:${g.ink}">${esc(n)}</span>&nbsp;&nbsp;${esc(label)}</div>`;
};

/** The small dashed label on every illustrated reading. */
export const illustrationTag = (text: string, t: number, g: Ground, x: number, y: number, alpha: number, size = 20, anchor: 'left' | 'right' = 'right'): string => {
  if (alpha <= 0) return '';
  const pos = anchor === 'right' ? `right:${x}px` : `left:${x}px`;
  return `<div class="abs" style="${pos};top:${y}px;font-family:${fonts.mono};font-size:${size}px;color:${g.soft};border:1.5px dashed ${g.bandEdge};border-radius:999px;padding:${size * 0.35}px ${size * 0.8}px;opacity:${(alpha * 0.92).toFixed(3)};white-space:nowrap">${esc(text)}</div>`;
};

/** A value waiting on the first release, drawn as the site draws one: hatched, dashed, said in words. */
export const pendingChip = (text: string, g: Ground, size: number): string =>
  `<span style="display:inline-block;font-family:${fonts.mono};font-size:${size}px;color:${g.pendingInk};background-color:${g.pendingGround};background-image:repeating-linear-gradient(135deg, rgba(227,201,138,0.10) 0 6px, transparent 6px 12px);border:1.5px dashed ${g.pendingInk};border-radius:${size * 0.3}px;padding:${size * 0.15}px ${size * 0.55}px">${esc(text)}</span>`;
