// Real footage in the frame: which captured frame to show at a time, a camera that zooms and pans across it, the
// pointer redrawn from the capture's log, and the rounded card it sits in.

import { clamp, easeInOut, lerp, span } from '../motion';

export interface Glide {
  t: number;
  from: [number, number];
  to: [number, number];
  seconds: number;
}

export interface Clip {
  frames: number;
  fps?: number;
  /** For the checker's frames, which were taken at irregular moments. */
  times?: number[];
  width: number;
  height: number;
  scale: number;
  events: Record<string, number>;
  glides: Glide[];
}

export interface FootageManifest {
  clips: Record<string, Clip>;
}

declare global {
  interface Window {
    __footage?: { base: string; manifest: FootageManifest };
  }
}

const footage = (): { base: string; manifest: FootageManifest } => {
  if (!window.__footage) throw new Error('no footage: render with --footage <dir> (the output of tools/prepare_footage.mjs)');
  return window.__footage;
};

export const clip = (name: string): Clip => {
  const c = footage().manifest.clips[name];
  if (!c) throw new Error(`the footage has no clip "${name}"`);
  return c;
};

/** A moment in a clip, from the clip's event list. */
export const at = (name: string, event: string): number => {
  const t = clip(name).events[event];
  if (t === undefined) throw new Error(`the clip "${name}" has no event "${event}"`);
  return t;
};

/** The url of the frame to show at footage time `t`. */
export const frameUrl = (name: string, t: number): string => {
  const c = clip(name);
  let index: number;
  if (c.times) {
    index = 0;
    for (let i = 0; i < c.times.length && c.times[i] <= t; i++) index = i;
  } else {
    index = Math.min(c.frames - 1, Math.max(0, Math.round(t * (c.fps ?? 30))));
  }
  return `${footage().base}/${name === 'checker' ? 'checker' : name}/f${String(index + 1).padStart(5, '0')}.jpg`;
};

/** Piecewise-linear footage time from keyframes of [scene time, footage time]; equal scene times make a cut. */
export const remap = (keys: [number, number][], t: number): number => {
  if (t <= keys[0][0]) return keys[0][1];
  for (let i = 1; i < keys.length; i++) {
    const [t0, f0] = keys[i - 1];
    const [t1, f1] = keys[i];
    if (t < t1) return t1 === t0 ? f1 : lerp(f0, f1, (t - t0) / (t1 - t0));
  }
  return keys[keys.length - 1][1];
};

/** Where the pointer was at footage time `t`, in the capture's pixels, redrawn from its log. */
export const pointerAt = (name: string, t: number): [number, number] => {
  const glides = clip(name).glides;
  let pos: [number, number] = glides[0].from;
  for (const g of glides) {
    if (t < g.t) break;
    const k = clamp((t - g.t) / g.seconds);
    const e = k * k * (3 - 2 * k);
    pos = [lerp(g.from[0], g.to[0], e), lerp(g.from[1], g.to[1], e)];
  }
  return pos;
};

export interface Region {
  /** Center and width, in the capture's pixels; the height follows the card's shape. */
  cx: number;
  cy: number;
  w: number;
}

/** Ease from one region to another between scene times `a` and `b`. */
export const pan = (from: Region, to: Region, t: number, a: number, b: number): Region => {
  const k = easeInOut(span(t, a, b));
  return { cx: lerp(from.cx, to.cx, k), cy: lerp(from.cy, to.cy, k), w: lerp(from.w, to.w, k) };
};

export interface CardOpts {
  x: number;
  y: number;
  w: number;
  h: number;
  radius?: number;
  clipName: string;
  /** Footage time to show. */
  time: number;
  region: Region;
  /** Draw the redrawn pointer, and click rings at these footage times. */
  pointer?: { clicks: number[] };
  opacity?: number;
  scale?: number;
  border?: string;
  shadow?: boolean;
}

/** An arrow of the kind every desktop draws, white with a dark edge. */
const arrow = (size: number): string =>
  `<svg width="${size}" height="${size * 1.4}" viewBox="0 0 20 28" style="position:absolute;left:0;top:0;filter:drop-shadow(0 3px 5px rgba(0,0,0,0.55))"><path d="M2 1.5 L2 22 L7.4 17.2 L10.8 25.2 L14.2 23.7 L10.9 15.9 L18 15.4 Z" fill="#fff" stroke="#16141A" stroke-width="1.6" stroke-linejoin="round"/></svg>`;

/** The frame at `time` inside a rounded card, seen through `region`. The image is decoded before the frame is taken (main.ts). */
export const card = (o: CardOpts): string => {
  const c = clip(o.clipName);
  const k = o.w / o.region.w;
  const imgW = c.width * k;
  const imgH = c.height * k;
  const left = o.w / 2 - o.region.cx * k;
  const top = o.h / 2 - o.region.cy * k;
  const radius = o.radius ?? 36;
  let pointer = '';
  // The checker's log is in the page's CSS pixels and its frames are at the device pixel ratio; the desktop clips log display pixels.
  const ps = o.clipName === 'checker' ? c.scale : 1;
  if (o.pointer) {
    const [rx, ry] = pointerAt(o.clipName, o.time);
    const px = rx * ps;
    const py = ry * ps;
    const rings = o.pointer.clicks
      .map((ct) => {
        const e = o.time - ct;
        if (e < 0 || e > 0.5) return '';
        const [cx, cy] = pointerAt(o.clipName, ct);
        const r = 14 + e * 110;
        return `<div style="position:absolute;left:${(cx * ps * k - r).toFixed(1)}px;top:${(cy * ps * k - r).toFixed(1)}px;width:${(r * 2).toFixed(1)}px;height:${(r * 2).toFixed(1)}px;border-radius:50%;border:3px solid rgba(238,232,220,${(0.8 * (1 - e / 0.5)).toFixed(2)})"></div>`;
      })
      .join('');
    pointer = `${rings}<div style="position:absolute;left:${(px * k).toFixed(1)}px;top:${(py * k).toFixed(1)}px;width:0;height:0">${arrow(30)}</div>`;
  }
  const shadow = o.shadow === false ? '' : 'box-shadow:0 30px 80px rgba(0,0,0,0.55),0 0 0 1.5px rgba(238,232,220,0.10);';
  return `<div class="abs" style="left:${o.x}px;top:${o.y}px;width:${o.w}px;height:${o.h}px;border-radius:${radius}px;overflow:hidden;background:#16141A;${shadow}opacity:${(o.opacity ?? 1).toFixed(3)};${o.scale ? `transform:scale(${o.scale});` : ''}${o.border ? `border:${o.border};` : ''}">
<div style="position:absolute;left:${left.toFixed(1)}px;top:${top.toFixed(1)}px;width:${imgW.toFixed(1)}px;height:${imgH.toFixed(1)}px">
<img data-src="${frameUrl(o.clipName, o.time)}" width="${imgW.toFixed(1)}" height="${imgH.toFixed(1)}" style="display:block">${pointer}
</div></div>`;
};
