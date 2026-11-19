// The mark, read from the app's own SVG at build time, so a redrawn mark reaches the ad on the next render.

import markSource from '../../app/assets/litmus_mark.svg';

export interface MarkPath {
  d: string;
  stroke: string;
  band: boolean;
}

/** The mark's paths in drawing order (blue, purple, red, then the pale band), parsed from the app's SVG. */
export const mark: { viewBox: string; paths: MarkPath[] } = (() => {
  const viewBox = /viewBox="([^"]+)"/.exec(markSource)?.[1];
  if (!viewBox) throw new Error('litmus_mark.svg has no viewBox');
  const paths = [...markSource.matchAll(/<path d="([^"]+)" stroke="([^"]+)"/g)].map((m) => ({
    d: m[1],
    stroke: m[2].toUpperCase(),
    band: m[2].toUpperCase() === '#C9B8E6',
  }));
  const order = ['#2F6FDB', '#8E5BC9', '#D9483B', '#C9B8E6'];
  paths.sort((a, b) => order.indexOf(a.stroke) - order.indexOf(b.stroke));
  if (paths.length !== 4 || paths.some((p) => !order.includes(p.stroke)))
    throw new Error('litmus_mark.svg no longer has the four strokes the ad draws; update content.ts');
  return { viewBox, paths };
})();
