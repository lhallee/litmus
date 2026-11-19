// The in-browser checker on a phone: real frames of the built page at a phone's size, with the pointer redrawn from the
// capture's log, laid on the beats. In the wide cut the phone stands at the right of the frame, captions at the left.

import { card, at, remap } from './footage';
import { clamp, easeInOut, span } from '../motion';
import { cueIn, inScene, live } from './clock';
import { G, flash, ground, line, wrap, safe } from './kit';
import type { Scene } from '../stage';

const RATIO = 2532 / 1170;

export const web: Scene = (t, dur, L, ctx) => {
  const verdictL = cueIn(ctx.cut, 'web', 'web_verdict');
  const paste = at('checker', 'paste');
  const press = at('checker', 'click_check');
  const read = at('checker', 'result');
  const clickL = verdictL - inScene(read - press);
  const jumpL = clickL - inScene(1.0);
  const keys: [number, number][] = [
    [0, paste - 0.2],
    [jumpL, paste - 0.2 + live(jumpL)],
    [jumpL, press - 1.0],
    [clickL, press],
    [dur, press + live(dur - clickL)],
  ];
  const time = remap(keys, t);

  // The phone is drawn at FIT of its full size so the Check button (the press the shot is about) sits inside the
  // band the platforms leave clear of their own controls (tall); in the wide cut it is sized to the frame's height.
  const SCREEN_W = L.wide ? 440 : 648;
  const SCREEN_H = Math.round(SCREEN_W * RATIO);
  const FIT = L.wide ? 1 : 0.82;
  const rise = easeInOut(span(t, 0, 0.5));
  const zoom = 1 + 0.025 * clamp(t / dur);
  const bodyW = SCREEN_W + 32;
  const bodyH = SCREEN_H + 32;
  const x = L.wide ? 1204 : (1080 - bodyW) / 2;
  const y = (L.wide ? (1080 - bodyH) / 2 : 451) + (1 - rise) * 60;
  const phone = `<div class="abs" style="left:${x}px;top:${y}px;width:${bodyW}px;height:${bodyH}px;border-radius:${L.wide ? 66 : 84}px;background:#0c0b0e;box-shadow:0 40px 100px rgba(0,0,0,0.6),0 0 0 3px #3a3540,inset 0 0 0 2px #26222b;transform:scale(${zoom.toFixed(4)});transform-origin:50% 30%;opacity:${rise.toFixed(3)}"></div>`;
  const screen = card({
    x: x + 16,
    y: y + 16,
    w: SCREEN_W,
    h: SCREEN_H,
    radius: L.wide ? 52 : 68,
    clipName: 'checker',
    time,
    region: { cx: 585, cy: 1266, w: 1170 },
    pointer: { clicks: [at('checker', 'click_box'), press] },
    opacity: rise,
    scale: zoom,
    shadow: false,
  });
  const s0 = safe(L);
  const box = L.wide ? { x: s0.x0, w: 900, align: 'left' as const } : { x: s0.x0, w: s0.x1 - s0.x0, align: 'center' as const };
  const size = (tall: number, wide: number): number => (L.wide ? wide : tall);
  const top = (tall: number): number => (L.wide ? 380 : tall);
  const captions =
    line(t, { ...box, text: 'Or in your browser.', y: top(276), size: size(90, 116), from: 0.1, to: verdictL - 0.05, font: 'head' }) +
    line(t, { ...box, text: 'Nothing leaves the page.', y: top(282), size: size(76, 100), from: verdictL + 0.1, to: dur, font: 'head', color: G.humanInk });
  const device = `<div class="abs" style="left:0;top:0;width:${L.W}px;height:${L.H}px;transform:scale(${FIT});transform-origin:${L.wide ? '0 0' : '540px 0'}">${phone}${screen}</div>`;
  return wrap(L, t, dur, `${ground(L, ctx.abs + t)}${device}${captions}${flash(L, t, 0.2)}`);
};
