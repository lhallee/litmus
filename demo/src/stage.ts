// What a scene is: a pure function from its own clock to the HTML of one frame.

/** The frame a scene draws into: the vertical ad is 1080 by 1920, the wide cut 1920 by 1080; `wide` picks the composition. */
export interface Layout {
  W: number;
  H: number;
  wide: boolean;
}

/** What a scene may need to know beyond its own clock: where it sits in the cut. */
export interface Context {
  /** Seconds from the start of the cut to the start of this shot. */
  abs: number;
  cut: string;
}

/** `t` is scene seconds since the scene started and `dur` its length (see clock.ts for scene seconds against real ones). */
export type Scene = (t: number, dur: number, L: Layout, ctx: Context) => string;

export interface Shot {
  name: string;
  scene: Scene;
  dur: number;
  /** Local scene times worth a still, for review. */
  stills: number[];
}

export interface Cut {
  name: string;
  W: number;
  H: number;
  fps: number;
  shots: Shot[];
}

/** Absolute start times of a cut's shots and its total length, in scene seconds; shots follow one another with no overlap, on the bars. */
export const schedule = (cut: Cut): { starts: number[]; duration: number } => {
  const starts: number[] = [];
  let at = 0;
  for (const s of cut.shots) {
    starts.push(at);
    at += s.dur;
  }
  return { starts, duration: at };
};
