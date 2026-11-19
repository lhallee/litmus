// The desktop app, as it runs: real footage of a text pasted and checked, with the pointer redrawn from the capture's
// log. The footage's own moments (paste, press, reading) are laid on the music's beats by remapping its time; between
// the moments it plays at its recorded speed (live and inScene in clock.ts), whatever the tempo.

import { card, at, remap, type Region } from './footage';
import { brand } from '../palette';
import { clamp, lerp } from '../motion';
import { cueIn, inScene, live } from './clock';
import { G, flash, ground, line, pill, wrap, safe } from './kit';
import type { Scene } from '../stage';

// Where the parts of a reading sit in the footage, in the capture's pixels (the app's own layout at this window size).
const UI = { verdict: [177, 192], strip: [1190, 416], text: [640, 1000] } as const;

export const app: Scene = (t, dur, L, ctx) => {
  const pasteL = cueIn(ctx.cut, 'app', 'app_paste');
  const verdictL = cueIn(ctx.cut, 'app', 'app_verdict');
  const paste = at('app', 'paste');
  const press = at('app', 'click_check');
  const read = at('app', 'result');
  const clickL = verdictL - inScene(read - press);
  const jumpL = clickL - inScene(1.2);
  const keys: [number, number][] = [
    [0, paste - live(pasteL)],
    [jumpL, paste + live(jumpL - pasteL)],
    [jumpL, press - 1.2],
    [clickL, press],
    [dur, press + live(dur - clickL)],
  ];
  const time = remap(keys, t);

  const geometry = L.wide ? { x: 880, y: 70, w: 930, h: 940 } : { x: 60, y: 440, w: 960, h: 1020 };
  // The wide card is shorter than the tall one, so the view sits 34 px higher to keep the window's title row in it.
  const lift = L.wide ? 34 : 0;
  const from: Region = { cx: 650, cy: 705 - lift, w: 1300 };
  const k = clamp(t / dur);
  const region: Region = { cx: lerp(from.cx, 650, k), cy: lerp(from.cy, 690 - lift, k), w: lerp(from.w, 1230, k) };
  const scale = geometry.w / region.w;
  const x0 = region.cx - region.w / 2;
  const y0 = region.cy - geometry.h / scale / 2;
  const toFrame = (p: readonly [number, number]): [number, number] => [geometry.x + (p[0] - x0) * scale, geometry.y + (p[1] - y0) * scale];

  const glow = t >= verdictL ? Math.max(0, 1 - (t - verdictL) / 0.9) : 0;
  const clicks = [at('app', 'click_box'), press];
  const frame = card({ ...geometry, clipName: 'app', time, region, pointer: { clicks }, border: `3px solid rgba(217,72,59,${(0.15 + 0.7 * glow).toFixed(3)})` });

  const [vx, vy] = toFrame(UI.verdict);
  const [sx, sy] = toFrame(UI.strip);
  const [tx, ty] = toFrame(UI.text);
  const labels =
    pill('the verdict', vx + 110, vy - 36, t, verdictL + 0.25, verdictL + 1.5, G.aiInk, 34) +
    pill('how sure', sx - 330, sy + 92, t, verdictL + 0.7, verdictL + 2.0, brand.band, 34) +
    pill('where it reads AI', tx - 190, ty - 120, t, verdictL + 1.4, dur, G.aiInk, 34);

  const s0 = safe(L);
  const box = L.wide ? { x: s0.x0, w: 740, align: 'left' as const } : { x: s0.x0, w: s0.x1 - s0.x0, align: 'center' as const };
  const size = (tall: number, wide: number): number => (L.wide ? wide : tall);
  const top = (tall: number): number => (L.wide ? 380 : tall);
  const captions =
    line(t, { ...box, text: 'Paste. Check.', y: top(272), size: size(100, 120), from: 0.15, to: verdictL - 0.1, font: 'head' }) +
    line(t, { ...box, text: 'A verdict, and how sure.', y: top(280), size: size(78, 100), from: verdictL + 0.1, to: verdictL + 1.5, font: 'head' }) +
    line(t, { ...box, text: 'Tinted where it reads.', y: top(280), size: size(82, 100), from: verdictL + 1.55, to: dur, font: 'head' });
  return wrap(L, t, dur, `${ground(L, ctx.abs + t)}${frame}${labels}${captions}${flash(L, t, 0.2)}`);
};
