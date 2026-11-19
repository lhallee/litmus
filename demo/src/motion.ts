// Time and easing. Every frame is a pure function of time in seconds, so nothing here keeps state.

export const clamp = (v: number, lo = 0, hi = 1): number => Math.min(hi, Math.max(lo, v));
export const lerp = (a: number, b: number, k: number): number => a + (b - a) * k;

/** Progress 0..1 of `t` through the span [a, b]. */
export const span = (t: number, a: number, b: number): number => (b <= a ? (t >= b ? 1 : 0) : clamp((t - a) / (b - a)));

export const easeInOut = (k: number): number => (k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2);
export const easeOut = (k: number): number => 1 - Math.pow(1 - k, 3);
export const easeOutQuint = (k: number): number => 1 - Math.pow(1 - k, 5);
export const easeIn = (k: number): number => k * k * k;

/** An ease-out curve with a small overshoot, for things that land (the mark, cells). */
export const easeBack = (k: number, s = 1.3): number => {
  const c3 = s + 1;
  return 1 + c3 * Math.pow(k - 1, 3) + s * Math.pow(k - 1, 2);
};

/** A damped spring from 0 to 1, settling by k = 1. */
export const spring = (k: number, damping = 6): number => {
  if (k <= 0) return 0;
  if (k >= 1) return 1;
  return 1 - Math.exp(-damping * k) * Math.cos(2 * Math.PI * 1.2 * k);
};

/** Opacity of something shown from `a` to `b`, fading in over `fin` and out over `fout` seconds. */
export const shown = (t: number, a: number, b: number, fin = 0.5, fout = 0.5): number =>
  Math.min(easeOut(span(t, a, a + fin)), 1 - easeIn(span(t, b - fout, b)));

/** Deterministic noise in [0, 1) for a seed, used for grain offsets and jitter. */
export const hash = (n: number): number => {
  const s = Math.sin(n * 127.1 + 311.7) * 43758.5453;
  return s - Math.floor(s);
};
