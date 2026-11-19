// Synthesize the ad's music and sound effects offline, deterministically, on the grid in data/ad/timeline.json.
//
//   node tools/music.mjs --cut ad --out <file.wav>
//
// Nothing is sampled or borrowed: every sound is oscillators, filtered noise, and envelopes computed here, so the track
// is ours and the same inputs always give the same samples. The tempo, the sections, and the cues (hit, tick, ding, pop)
// come from the timeline file, which the picture's scenes read too, so music and picture stay in step.

import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc), []));
// A cut named in timeline.json's aliases (the wide cut) has the music of the cut it is an alias of.
const asked = args.cut ?? 'ad';
if (!args.out) throw new Error('give --out <file.wav>');
const timeline = JSON.parse(readFileSync(join(here, '../data/ad/timeline.json'), 'utf8'));
const cutName = timeline.aliases?.[asked] ?? asked;

const SR = 44100;
const beat = 60 / timeline.bpm;
const bar = beat * timeline.beats_per_bar;
const bars = timeline.cuts[cutName].shots.reduce((a, s) => a + s.bars, 0);
const seconds = bars * bar;
const tail = 2.5;
const N = Math.ceil((seconds + tail) * SR);
const L = new Float32Array(N);
const R = new Float32Array(N);
const sendL = new Float32Array(N);
const sendR = new Float32Array(N);

/** A seeded generator (mulberry32), so the noise is the same every time. */
let seed = 0x6c69746d;
const rnd = () => {
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const noise = () => rnd() * 2 - 1;
const hz = (semitonesFromA4) => 440 * 2 ** (semitonesFromA4 / 12);

/** Add `fn(t)` (seconds since the note began) into the mix from `start` for `len` seconds, panned and sent to the reverb. */
const voice = (start, len, gain, pan, send, fn) => {
  const i0 = Math.max(0, Math.floor(start * SR));
  const i1 = Math.min(N, Math.ceil((start + len) * SR));
  const gl = gain * Math.cos(((pan + 1) * Math.PI) / 4);
  const gr = gain * Math.sin(((pan + 1) * Math.PI) / 4);
  for (let i = i0; i < i1; i++) {
    const v = fn((i - start * SR) / SR);
    L[i] += v * gl;
    R[i] += v * gr;
    if (send > 0) {
      sendL[i] += v * gl * send;
      sendR[i] += v * gr * send;
    }
  }
};

/** A one-pole low-pass with the cutoff given per sample. */
const lowpass = () => {
  let y = 0;
  return (x, fc) => {
    const a = 1 - Math.exp((-2 * Math.PI * fc) / SR);
    y += a * (x - y);
    return y;
  };
};
const highpass = () => {
  const lp = lowpass();
  return (x, fc) => x - lp(x, fc);
};
const saw = (phase) => 2 * (phase - Math.floor(phase + 0.5));

const kick = (t0, gain = 1) =>
  voice(t0, 0.45, 0.9 * gain, 0, 0.03, (t) => {
    const phase = 2 * Math.PI * (44 * t + 110 * 0.028 * (1 - Math.exp(-t / 0.028)));
    return (Math.sin(phase) * Math.exp(-t / 0.16) + (t < 0.004 ? noise() * 0.3 : 0)) * Math.min(1, t / 0.0008);
  });

const clap = (t0, gain = 1) => {
  const hp = highpass();
  voice(t0, 0.3, 0.32 * gain, 0.1, 0.18, (t) => {
    const bursts = [0, 0.011, 0.022].reduce((e, b) => e + (t >= b ? Math.exp(-(t - b) / 0.012) * 0.5 : 0), 0);
    return hp(noise(), 1200) * (bursts + Math.exp(-t / 0.09) * 0.6);
  });
};

const hat = (t0, open, gain = 1, pan = 0.3) => {
  const hp = highpass();
  voice(t0, open ? 0.3 : 0.07, 0.11 * gain, pan, 0.05, (t) => hp(noise(), 7000) * Math.exp(-t / (open ? 0.09 : 0.016)));
};

const bass = (t0, len, freq, gain = 1) => {
  const a = lowpass();
  const b = lowpass();
  voice(t0, len + 0.08, 0.34 * gain, 0, 0, (t) => {
    const env = Math.min(1, t / 0.004) * Math.exp(-t / (len * 1.2));
    const fc = 140 + 900 * Math.exp(-t / 0.08);
    return b(a(saw(freq * t) * 0.8 + Math.sin(2 * Math.PI * freq * 0.5 * t) * 0.6, fc), fc) * 2.2 * env;
  });
};

const pluck = (t0, freq, gain, pan, bright) => {
  const lp = lowpass();
  voice(t0, 0.32, 0.11 * gain, pan, 0.12, (t) => {
    const sq = Math.sign(Math.sin(2 * Math.PI * freq * t)) * 0.5 + saw(freq * 1.004 * t) * 0.5;
    return lp(sq, 500 + bright * 3500 * Math.exp(-t / 0.06)) * Math.exp(-t / 0.11) * Math.min(1, t / 0.002);
  });
};

const pad = (t0, len, notes, gain) => {
  const a = lowpass();
  voice(t0, len + 0.9, 0.05 * gain, 0, 0.35, (t) => {
    let s = 0;
    for (const n of notes) for (const d of [-0.07, 0, 0.07]) s += saw(hz(n + d) * t + d * 3);
    const env = Math.min(1, t / 0.45) * (t < len ? 1 : Math.max(0, 1 - (t - len) / 0.9));
    return a(s / (notes.length * 3), 900 + 700 * Math.min(1, t / len)) * env * 3;
  });
};

const riser = (t0, len, gain = 1) => {
  const hp = highpass();
  voice(t0, len, 0.2 * gain, 0, 0.1, (t) => {
    const k = t / len;
    return hp(noise(), 300 + 7000 * k * k) * k * k;
  });
};

const hit = (t0, gain = 1) => {
  voice(t0, 1.6, 0.7 * gain, 0, 0.3, (t) => Math.sin(2 * Math.PI * (38 + 45 * Math.exp(-t / 0.1)) * t) * Math.exp(-t / 0.5));
  const hp = highpass();
  voice(t0, 0.9, 0.3 * gain, 0, 0.45, (t) => hp(noise(), 2500) * Math.exp(-t / 0.18));
};

const tick = (t0, gain = 1) => voice(t0, 0.05, 0.16 * gain, 0.15, 0.1, (t) => Math.sin(2 * Math.PI * 2400 * t) * Math.exp(-t / 0.009));

const ding = (t0, gain = 1) =>
  voice(t0, 1.3, 0.2 * gain, 0, 0.5, (t) => (Math.sin(2 * Math.PI * hz(12) * t) + 0.5 * Math.sin(2 * Math.PI * hz(19) * t) + 0.25 * Math.sin(2 * Math.PI * hz(28) * t)) * Math.exp(-t / 0.4));

const pop = (t0, gain = 1) => {
  voice(t0, 0.25, 0.3 * gain, -0.1, 0.25, (t) => Math.sin(2 * Math.PI * (320 + 700 * (1 - Math.exp(-t / 0.02))) * t) * Math.exp(-t / 0.05));
  voice(t0 + 0.06, 0.9, 0.12 * gain, 0.2, 0.5, (t) => Math.sin(2 * Math.PI * hz(24) * t) * Math.exp(-t / 0.25));
};

// The song: chords by bar, the layers each section's energy switches on.
const CHORD_NOTES = { Am: [-12, -9, -5], F: [-16, -12, -9], C: [-21, -17, -14], G: [-14, -10, -7] }; // semitones from A4, an octave and more below
const ROOTS = { Am: -36, F: -40, C: -33, G: -38 }; // bass roots (A1, F1, C2, G1 as semitones from A4)
const ARP = { Am: [0, 3, 7, 12], F: [-4, 0, 3, 8], C: [-9, -5, -2, 3], G: [-2, 2, 5, 10] };

const sections = timeline.sections[cutName];
const energyAt = (b) => sections.find((s) => b >= s.from_bar && b < s.to_bar)?.energy ?? 0.5;
for (let b = 0; b < bars; b++) {
  const chord = timeline.chords[b % timeline.chords.length];
  const e = energyAt(b);
  const t0 = b * bar;
  pad(t0, bar, CHORD_NOTES[chord].map((n) => n + 12), e < 0.6 ? 1.2 : 0.8);
  const solo = cutName === 'ad' && b === 0;
  for (let k = 0; k < 4; k++) {
    const tb = t0 + k * beat;
    if (e >= 0.5 && !solo && (e >= 0.65 || k === 0)) kick(tb, e >= 0.65 ? 1 : 0.8);
    if (e >= 0.65 && (k === 1 || k === 3)) clap(tb);
    if (e >= 0.65) hat(tb + beat / 2, false, 0.8, -0.3);
    if (e >= 0.9) {
      hat(tb + beat / 4, false, 0.5, 0.4);
      hat(tb + (3 * beat) / 4, false, 0.5, -0.4);
      hat(tb + beat / 2, true, 0.9, 0.2);
    }
    if (e >= 0.65 && !solo) for (const step of [0, 0.75]) bass(tb + step * beat, 0.2, hz(ROOTS[chord] + (k % 2 === 1 && step === 0.75 ? 12 : 0)));
    if (e >= 0.75) for (let s = 0; s < 4; s++) pluck(tb + (s * beat) / 4, hz(ARP[chord][(k * 4 + s) % 4] + (s % 2 === 1 ? 12 : 0)), 0.7 + 0.3 * e, s % 2 ? 0.4 : -0.4, e);
  }
  if (b > 0 && energyAt(b - 1) !== e && e >= energyAt(b - 1)) riser(t0 - beat * 4, beat * 4, 0.8 + 0.2 * e);
}
for (const cue of timeline.cues[cutName]) {
  const t = cue.beat * beat;
  ({ hit, tick, ding, pop })[cue.sound](t);
}

// A small reverb (four combs and two all-passes per side) for what was sent to it.
const reverb = (src, delays) => {
  const out = new Float32Array(N);
  for (const d of delays) {
    const n = Math.floor(d * SR);
    const buf = new Float32Array(n);
    let p = 0;
    for (let i = 0; i < N; i++) {
      const y = buf[p];
      buf[p] = src[i] + y * 0.78;
      out[i] += y * 0.25;
      p = (p + 1) % n;
    }
  }
  return out;
};
const wetL = reverb(sendL, [0.0297, 0.0371, 0.0411, 0.0437]);
const wetR = reverb(sendR, [0.0319, 0.0353, 0.0427, 0.0451]);

// The master: wet added, a gentle soft clip, and a fade over the last second so the file ends in silence.
const pcm = Buffer.alloc(N * 4);
for (let i = 0; i < N; i++) {
  const t = i / SR;
  const fade = t > seconds + tail - 0.4 ? Math.max(0, (seconds + tail - t) / 0.4) : 1;
  const l = Math.tanh((L[i] + wetL[i] * 0.6) * 1.1) * 0.9 * fade;
  const r = Math.tanh((R[i] + wetR[i] * 0.6) * 1.1) * 0.9 * fade;
  pcm.writeInt16LE(Math.round(l * 32767), i * 4);
  pcm.writeInt16LE(Math.round(r * 32767), i * 4 + 2);
}
const header = Buffer.alloc(44);
header.write('RIFF', 0);
header.writeUInt32LE(36 + pcm.length, 4);
header.write('WAVEfmt ', 8);
header.writeUInt32LE(16, 16);
header.writeUInt16LE(1, 20);
header.writeUInt16LE(2, 22);
header.writeUInt32LE(SR, 24);
header.writeUInt32LE(SR * 4, 28);
header.writeUInt16LE(4, 32);
header.writeUInt16LE(16, 34);
header.write('data', 36);
header.writeUInt32LE(pcm.length, 40);
writeFileSync(resolve(args.out), Buffer.concat([header, pcm]));
console.log(`${asked}: ${bars} bars at ${timeline.bpm} bpm, ${seconds.toFixed(1)} s (+${tail} s tail) -> ${args.out}`);
