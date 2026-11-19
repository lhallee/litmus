// Turn the captured clips into what the ad draws: numbered frames, and a list of when each thing happened, in the
// footage's own time.
//
//   node tools/prepare_footage.mjs --clips <dir with app.mp4, quick.mp4 and their .json logs, and site/> --out <dir> --work <dir with node_modules>
//
// The capture logs events on the scene's clock; the recording's clock starts a little later and by an amount that
// varies, so each recording carries a white flash in a corner at a logged moment, which is found here and gives the
// offset. The moment a reading is complete is found from the footage itself: the first frame after the action that
// matches the frame just before the scene lets go (the settled reading), so the ad's cuts follow the real app, whatever
// speed a new release reads at. Writes <out>/footage.json and <out>/<clip>/f00001.jpg and so on.

import { spawnSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { join, resolve } from 'node:path';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc), []));
for (const k of ['clips', 'out', 'work']) if (!args[k]) throw new Error(`give --${k}`);
const clips = resolve(args.clips);
const out = resolve(args.out);
const ffmpeg = createRequire(join(resolve(args.work), 'package.json'))('ffmpeg-static');
mkdirSync(out, { recursive: true });

const TW = 128;
const TH = 80;

/** Per-frame gray thumbnails (TW x TH) of a clip, from ffmpeg's raw output. */
const thumbs = (mp4) => {
  const run = spawnSync(ffmpeg, ['-loglevel', 'error', '-i', mp4, '-vf', `scale=${TW}:${TH}:flags=area,format=gray`, '-f', 'rawvideo', '-'], { maxBuffer: 1 << 30 });
  if (run.status !== 0) throw new Error(`ffmpeg failed on ${mp4}`);
  const frames = run.stdout.length / (TW * TH);
  return Array.from({ length: frames }, (_, i) => run.stdout.subarray(i * TW * TH, (i + 1) * TW * TH));
};
const diff = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
  return s / a.length;
};
/** Mean brightness of the bottom-right corner, where the capture flashes a white window. */
const corner = (f) => {
  let s = 0;
  for (let y = TH - 4; y < TH; y++) for (let x = TW - 4; x < TW; x++) s += f[y * TW + x];
  return s / 16;
};

const manifest = { about: 'Made by tools/prepare_footage.mjs from the captured clips; times are seconds in each clip.', clips: {} };

for (const name of ['app', 'quick']) {
  const mp4 = join(clips, `${name}.mp4`);
  const log = JSON.parse(readFileSync(join(clips, `${name}.json`), 'utf8'));
  const fps = log.fps;
  const dir = join(out, name);
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });
  const ex = spawnSync(ffmpeg, ['-loglevel', 'error', '-i', mp4, '-q:v', '3', '-start_number', '1', join(dir, 'f%05d.jpg')]);
  if (ex.status !== 0) throw new Error(`frame extraction failed for ${name}`);
  const t = thumbs(mp4);
  const base = corner(t[0]);
  const flash = t.findIndex((f) => corner(f) > base + 40);
  if (flash < 0) throw new Error(`${name}: the flash was not found in the footage`);
  const logged = log.events.find((e) => e.what === 'pulse_on').t;
  const offset = flash / fps - logged;
  const at = (what) => {
    const e = log.events.find((x) => x.what === what);
    if (!e) throw new Error(`${name}: the log has no "${what}"`);
    return e.t + offset;
  };
  const frameAt = (s) => Math.min(t.length - 1, Math.max(0, Math.round(s * fps)));
  /** First time after `from` whose frame matches the frame at `ref`, the settled state. */
  const settled = (from, ref) => {
    const target = t[frameAt(ref)];
    for (let i = frameAt(from); i <= frameAt(ref); i++) if (diff(t[i], target) < 0.35) return i / fps;
    throw new Error(`${name}: no settled frame between ${from} and ${ref}`);
  };
  const events = {};
  for (const e of log.events) if (e.what !== 'glide') events[e.what] = Number((e.t + offset).toFixed(3));
  if (name === 'app') events.result = settled(at('click_check') + 0.2, at('end') - 0.2);
  else for (const k of ['ai', 'short', 'human']) events[`result_${k}`] = settled(at(`hotkey_${k}`) + 0.1, at(`closed_${k}`) - 0.3);
  manifest.clips[name] = {
    frames: t.length,
    fps,
    width: log.display[0],
    height: log.display[1],
    scale: log.scale,
    offset_s: Number(offset.toFixed(3)),
    events,
    glides: log.events.filter((e) => e.what === 'glide').map((e) => ({ t: Number((e.t + offset).toFixed(3)), from: [e.x0, e.y0], to: [e.x1, e.y1], seconds: e.seconds })),
  };
  console.log(`${name}: ${t.length} frames, flash at frame ${flash}, offset ${offset.toFixed(3)} s`);
}

// The browser checker: frames with their times, and the pointer's glides, as captured.
{
  const site = JSON.parse(readFileSync(join(clips, 'site', 'site.json'), 'utf8'));
  const dir = join(out, 'checker');
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });
  const frames = site.events.filter((e) => e.what === 'frame');
  for (const f of frames) copyFileSync(join(clips, 'site', 'checker', `f${String(f.frame).padStart(4, '0')}.jpg`), join(dir, `f${String(f.frame + 1).padStart(5, '0')}.jpg`));
  const events = {};
  for (const e of site.events) if (!['frame', 'glide', 'landing', 'viewport'].includes(e.what)) events[e.what] = e.t;
  // When the reading appeared: the first frame after Check that matches the settled one.
  const sheet = spawnSync(ffmpeg, ['-loglevel', 'error', '-framerate', '10', '-i', join(dir, 'f%05d.jpg'), '-vf', `scale=${TW}:${TH * 2}:flags=area,format=gray`, '-f', 'rawvideo', '-'], { maxBuffer: 1 << 30 });
  if (sheet.status !== 0) throw new Error('could not read the checker frames');
  const size = TW * TH * 2;
  const frameOf = (i) => sheet.stdout.subarray(i * size, (i + 1) * size);
  const times = frames.map((f) => f.t);
  const settledAt = times.findIndex((tt) => tt >= events.result_settled);
  let found = -1;
  for (let i = times.findIndex((tt) => tt >= events.click_check + 0.05); i <= settledAt; i++) {
    if (diff(frameOf(i), frameOf(settledAt)) < 0.35) {
      found = i;
      break;
    }
  }
  if (found < 0) throw new Error('the checker never settled');
  events.result = times[found];
  manifest.clips.checker = {
    frames: frames.length,
    times: frames.map((f) => f.t),
    width: site.viewport[0] * site.scale,
    height: site.viewport[1] * site.scale,
    scale: site.scale,
    events,
    glides: site.events.filter((e) => e.what === 'glide').map((e) => ({ t: e.t, from: e.from, to: e.to, seconds: e.seconds })),
  };
  for (const f of ['site_top.png', 'site_full.png']) if (existsSync(join(clips, 'site', f))) copyFileSync(join(clips, 'site', f), join(out, f));
  console.log(`checker: ${frames.length} frames`);
}
writeFileSync(join(out, 'footage.json'), JSON.stringify(manifest, null, 1));
console.log(`wrote ${join(out, 'footage.json')}`);
