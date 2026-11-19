// The cuts: the vertical ad (1080 by 1920), its teaser, and the wide cut (1920 by 1080, the ad's story recomposed for a
// monitor's ratio), all at 30 fps, built from the shots and lengths in data/ad/timeline.json (bars of the music's grid).
// The wide cut is not a second set of scenes: every scene reads its layout (stage.ts) and draws the same shot for the frame
// it is given. STORYBOARD.md mirrors this.

import { app } from './ad/app';
import { shotsOf } from './ad/clock';
import { end } from './ad/end';
import { hook } from './ad/hook';
import { hypeScene } from './ad/hype';
import { quick } from './ad/quick';
import { web } from './ad/web';
import type { Cut, Layout, Scene, Shot } from './stage';

const SCENES: Record<string, Scene> = { hook, app, quick, web, hype: hypeScene, end };

const TALL: Layout = { W: 1080, H: 1920, wide: false };
const WIDE: Layout = { W: 1920, H: 1080, wide: true };

/** Local scene times worth a still in each shot, for review: the arrival, the moves, the settled frame, and the frame before the cut. */
const STILLS: Record<string, number[]> = {
  hook: [0, 0.3, 1.2, 2.2, 3.6, 3.95],
  app: [0.05, 0.4, 1.8, 3.4, 4.3, 5.0, 5.8, 5.95],
  quick: [0.05, 0.3, 1.2, 2.3, 3.4, 4.6, 5.0, 5.95],
  web: [0.05, 0.5, 1.4, 2.4, 3.6, 3.95],
  hype: [0.05, 0.5, 1.5, 3.0, 5.0, 7.0, 7.9, 8.3, 9.6, 9.95],
  end: [0.05, 0.4, 1.6, 3.4, 5.6, 5.95],
};

const build = (name: string, layout: Layout): Cut => ({
  name,
  W: layout.W,
  H: layout.H,
  fps: 30,
  shots: shotsOf(name).map<Shot>((s) => {
    const scene = SCENES[s.name];
    if (!scene) throw new Error(`no scene for the shot "${s.name}"`);
    return { name: s.name, scene, dur: s.dur, stills: (STILLS[s.name] ?? [0.5]).filter((x) => x < s.dur) };
  }),
});

export const layouts: Record<string, Layout> = { ad: TALL, teaser: TALL, wide: WIDE };
export const cuts: Record<string, Cut> = { ad: build('ad', TALL), teaser: build('teaser', TALL), wide: build('wide', WIDE) };
