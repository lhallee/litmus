// Quick check, the feature that works in any program: real footage of text selected in an editor and the hotkey
// pressed, with the small window the app opens by the pointer. The footage's moments are laid on the beats; outside
// the stretches the cues squeeze (the idle wait before the hotkey), it plays at its recorded speed.

import { card, at, remap } from './footage';
import { clamp, lerp } from '../motion';
import { cueIn, inScene, isFull, live } from './clock';
import { G, flash, ground, line, wrap, safe } from './kit';
import type { Scene } from '../stage';

export const quick: Scene = (t, dur, L, ctx) => {
  const long = isFull(ctx.cut);
  const selectL = cueIn(ctx.cut, 'quick', 'quick_select_1');
  const popupL = cueIn(ctx.cut, 'quick', 'quick_popup_1');
  const sa = at('quick', 'select_ai');
  const ra = at('quick', 'result_ai');
  const keys: [number, number][] = [
    [0, sa - live(selectL)],
    [selectL, sa],
    [popupL, ra],
  ];
  let secondL = Number.POSITIVE_INFINITY;
  if (long) {
    const ss = at('quick', 'select_short');
    const rs = at('quick', 'result_short');
    secondL = cueIn(ctx.cut, 'quick', 'quick_select_2');
    const popup2L = cueIn(ctx.cut, 'quick', 'quick_popup_2');
    const cutL = secondL - inScene(0.3);
    keys.push([cutL, ra + live(cutL - popupL)], [cutL, ss - 0.3], [secondL, ss], [popup2L, rs], [dur, rs + live(dur - popup2L)]);
  } else {
    keys.push([dur, ra + live(dur - popupL)]);
  }
  const time = remap(keys, t);

  const k = clamp(t / dur);
  // The left edge stays on the text's first letters (x = 295 in the capture) while the view pushes in.
  const w = lerp(1100, 1020, k);
  const region = { cx: 295 + w / 2, cy: lerp(640, 620, k), w };
  const clicks = [sa, long ? at('quick', 'select_short') : sa];
  const geometry = L.wide ? { x: 880, y: 70, w: 930, h: 940 } : { x: 60, y: 440, w: 960, h: 1000 };
  const frame = card({ ...geometry, clipName: 'quick', time, region, pointer: { clicks } });

  const hotkeyAt = selectL + (at('quick', 'hotkey_ai') - sa) / ((ra - sa) / (popupL - selectL));
  const s0 = safe(L);
  const box = L.wide ? { x: s0.x0, w: 740, align: 'left' as const } : { x: s0.x0, w: s0.x1 - s0.x0, align: 'center' as const };
  const size = (tall: number, wide: number): number => (L.wide ? wide : tall);
  const top = (tall: number): number => (L.wide ? 380 : tall);
  const captions =
    line(t, { ...box, text: 'Select text in any app.', y: top(278), size: size(84, 108), from: 0.1, to: hotkeyAt - 0.05, font: 'head' }) +
    line(t, { ...box, text: 'Press Ctrl+Alt+L.', y: top(272), size: size(100, 120), from: hotkeyAt, to: long ? secondL - 0.1 : dur, font: 'head' }) +
    (long ? line(t, { ...box, text: 'It says when it can’t tell.', y: top(280), size: size(78, 100), from: secondL, to: dur, font: 'head', color: G.mixedInk }) : '');
  return wrap(L, t, dur, `${ground(L, ctx.abs + t)}${frame}${captions}${flash(L, t, 0.2)}`);
};
