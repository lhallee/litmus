// The ad's clock: bars and beats from data/ad/timeline.json, the same file the music is synthesized from.
//
// Scenes are drawn in "scene seconds": the clock of a film at the tempo the scenes' own durations were authored at
// (authored_bpm, 120 beats a minute, so a beat is half a second and a bar two). The film plays at bpm; one scene second
// lasts PACE real seconds, so every cut, cue, fade, word and camera move slows together when bpm falls (main.ts does the
// conversion). Footage is the exception: a recording is shown at its own speed, so footage code converts with `live`
// and `inScene` below.

import timelineFile from '../../data/ad/timeline.json';
import type { Cut } from '../stage';

interface Cue {
  name: string;
  beat: number;
  sound: string;
}

interface TimelineFile {
  bpm: number;
  authored_bpm: number;
  beats_per_bar: number;
  aliases: Record<string, string>;
  cuts: Record<string, { shots: { name: string; bars: number }[] }>;
  cues: Record<string, Cue[]>;
}

const timeline = timelineFile as unknown as TimelineFile;

/** The cut whose shots, cues and sections a cut uses: the wide cut is the ad's story in another frame. */
export const baseCut = (cut: string): string => timeline.aliases[cut] ?? cut;

/** True for the ad and its wide cut, the long versions of the shots; false for the teaser. */
export const isFull = (cut: string): boolean => baseCut(cut) === 'ad';

/** Real seconds per scene second: authored_bpm over bpm. */
export const PACE = timeline.authored_bpm / timeline.bpm;

/** A beat and a bar in scene seconds. */
export const BEAT = 60 / timeline.authored_bpm;
export const BAR = BEAT * timeline.beats_per_bar;

/** Footage seconds that pass while `sceneSeconds` of scene time do, when the footage plays at its recorded speed. */
export const live = (sceneSeconds: number): number => sceneSeconds * PACE;

/** Scene seconds that `footageSeconds` of footage played at its recorded speed take. */
export const inScene = (footageSeconds: number): number => footageSeconds / PACE;

/** The cue's time in scene seconds from the start of the cut, which is a whole number of beats. */
export const cueTime = (cut: string, name: string): number => {
  const cue = timeline.cues[baseCut(cut)]?.find((c) => c.name === name);
  if (!cue) throw new Error(`timeline.json has no cue "${name}" for the cut "${cut}"`);
  return cue.beat * BEAT;
};

/** Where a shot starts, in scene seconds from the start of its cut. */
export const shotStart = (cut: string, shot: string): number => {
  let at = 0;
  for (const s of timeline.cuts[baseCut(cut)].shots) {
    if (s.name === shot) return at;
    at += s.bars * BAR;
  }
  throw new Error(`the cut "${cut}" has no shot "${shot}"`);
};

/** A cue's time within its shot (the shot's own clock), for scenes that anchor footage to the music. */
export const cueIn = (cut: string, shot: string, name: string): number => cueTime(cut, name) - shotStart(cut, shot);

/** 1 on a beat, falling away quickly, for things that breathe with the kick (`abs` in scene seconds). */
export const pulse = (abs: number, decay = 5): number => Math.exp(-((abs / BEAT) % 1) * decay);

/** A cut's shots with their lengths in scene seconds. */
export const shotsOf = (cut: string): { name: string; dur: number }[] => timeline.cuts[baseCut(cut)].shots.map((s) => ({ name: s.name, dur: s.bars * BAR }));

export type Maker = (name: string, dur: number) => Cut['shots'][number];
