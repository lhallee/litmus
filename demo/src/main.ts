// The page that draws a cut. `?cut=ad` or `?cut=teaser` picks the cut; `&play` plays it in real time for review (space
// pauses, the arrow keys step, a click on the bar seeks). The renderer calls window.__film.render(t), which draws the
// frame at time t and waits for the footage images in it to decode.

import { PACE } from './ad/clock';
import { cuts, layouts } from './cuts';
import { dark, fonts } from './palette';
import { schedule } from './stage';

const params = new URLSearchParams(location.search);
const cut = cuts[params.get('cut') ?? 'ad'];
if (!cut) throw new Error(`no cut "${params.get('cut')}"; known: ${Object.keys(cuts).join(', ')}`);
const L = layouts[cut.name];
// Scenes draw in scene seconds, which last PACE real seconds each (ad/clock.ts); the film's clock, the renderer's and the stills' are real.
const { starts, duration: sceneDuration } = schedule(cut);
const duration = sceneDuration * PACE;

const css = `
html, body { margin: 0; background: #000; }
#stage { position: relative; width: ${L.W}px; height: ${L.H}px; overflow: hidden; background: ${dark.ground}; transform-origin: 0 0;
  font-family: ${fonts.body}; -webkit-font-smoothing: antialiased; text-rendering: geometricPrecision; font-kerning: normal; }
.abs { position: absolute; }
.w { display: inline-block; }
#bar { position: fixed; left: 0; right: 0; bottom: 0; height: 28px; background: #222; cursor: pointer; font: 12px monospace; color: #ccc; }
#bar i { position: absolute; left: 0; top: 0; bottom: 0; background: #8E5BC9; }
#bar span { position: absolute; left: 8px; top: 6px; }
`;
const style = document.createElement('style');
style.textContent = css;
document.head.appendChild(style);

const stage = document.createElement('div');
stage.id = 'stage';
document.body.appendChild(stage);

/** Draw the frame at `t`, then wait until every footage image in it has decoded, so the frame taken next is complete. */
const render = async (t: number): Promise<void> => {
  const layers: string[] = [];
  cut.shots.forEach((shot, i) => {
    const local = t / PACE - starts[i];
    if (local < 0 || local >= shot.dur) return;
    layers.push(`<div class="abs" data-shot="${shot.name}" style="left:0;top:0;width:${L.W}px;height:${L.H}px">${shot.scene(local, shot.dur, L, { abs: starts[i], cut: cut.name })}</div>`);
  });
  stage.innerHTML = layers.join('');
  const images = [...stage.querySelectorAll<HTMLImageElement>('img[data-src]')];
  await Promise.all(
    images.map(async (img) => {
      img.src = img.dataset.src ?? '';
      await img.decode();
    }),
  );
};

const stills = cut.shots.flatMap((shot, i) => shot.stills.map((s) => ({ name: `${shot.name}_${(s * PACE).toFixed(2)}`, t: (starts[i] + s) * PACE })));

const ready = (async () => {
  await Promise.all([
    document.fonts.load(`600 40px ${fonts.head}`),
    document.fonts.load(`640 40px ${fonts.head}`),
    document.fonts.load(`400 40px ${fonts.body}`),
    document.fonts.load(`600 40px ${fonts.body}`),
    document.fonts.load(`400 40px ${fonts.mono}`),
    document.fonts.load(`500 40px ${fonts.mono}`),
  ]);
  await document.fonts.ready;
  await render(0);
})();

declare global {
  interface Window {
    __film: { ready: Promise<void>; duration: number; width: number; height: number; fps: number; stills: { name: string; t: number }[]; render: (t: number) => Promise<void> };
  }
}
window.__film = { ready, duration, width: L.W, height: L.H, fps: cut.fps, stills, render };

if (params.has('play')) {
  void ready.then(() => {
    const fit = () => {
      stage.style.transform = `scale(${Math.min(innerWidth / L.W, (innerHeight - 28) / L.H)})`;
    };
    fit();
    addEventListener('resize', fit);
    const bar = document.createElement('div');
    bar.id = 'bar';
    bar.innerHTML = '<i></i><span></span>';
    document.body.appendChild(bar);
    let t = Number(params.get('t') ?? 0);
    let playing = true;
    let last = performance.now();
    bar.addEventListener('click', (e) => {
      t = (e.clientX / innerWidth) * duration;
    });
    addEventListener('keydown', (e) => {
      if (e.key === ' ') playing = !playing;
      if (e.key === 'ArrowRight') t += e.shiftKey ? 1 : 1 / cut.fps;
      if (e.key === 'ArrowLeft') t -= e.shiftKey ? 1 : 1 / cut.fps;
    });
    const tick = async (now: number) => {
      if (playing) t += (now - last) / 1000;
      last = now;
      if (t >= duration) t = 0;
      if (t < 0) t = 0;
      await render(t);
      (bar.firstChild as HTMLElement).style.width = `${(t / duration) * 100}%`;
      (bar.lastChild as HTMLElement).textContent = `${t.toFixed(2)} / ${duration.toFixed(2)} s`;
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });
}
