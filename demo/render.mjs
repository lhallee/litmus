// Render the ad, its teaser, and the wide cut to MP4, frame by frame: bundle src/ with esbuild, draw each frame in headless Chromium
// (Playwright) with the real footage in it, encode H.264, synthesize the music on the same beat grid, and mix and
// loudness-normalize the sound into the file. Dependencies, build output, frames and videos all live in the work folder,
// outside the tree.
//
//   node render.mjs ad     --work <dir> --footage <dir>   the ad          -> <out>/litmus_ad.mp4
//   node render.mjs teaser --work <dir> --footage <dir>   the teaser      -> <out>/litmus_teaser.mp4
//   node render.mjs wide   --work <dir> --footage <dir>   the wide cut   -> <out>/litmus_wide.mp4, with its web encodes
//   node render.mjs all    --work <dir> --stills          all three, with a still at each beat of review
//   node render.mjs ad     --work <dir> --preview         build only, and print an address that plays the cut live
//   node render.mjs wide   --work <dir> --web-only        remake only the wide cut's web encodes from its finished file
//
// Options: --gpu-use <json> (the measured GPU figures; default data/ad/gpu_use.json, and the speed shot has no GPU panel without it), --footage <dir> (the output of tools/prepare_footage.mjs; default <work>/footage), --workers 3 (pages drawing
// in parallel), --crf 16, --stills, --no-video, --no-audio, --from <s> --to <s> (a slice), --out <dir> (default
// <work>/out), --install (install the dependencies and stop). $LITMUS_DEMO_WORK sets --work.

import { spawn, spawnSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));

const parseArgs = (argv) => {
  const out = { _: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) out._.push(a);
    else if (['stills', 'no-video', 'no-audio', 'preview', 'install', 'web-only'].includes(a.slice(2))) out[a.slice(2)] = true;
    else out[a.slice(2)] = argv[++i];
  }
  return out;
};

const args = parseArgs(process.argv.slice(2));
const which = args._[0] ?? 'ad';
const cutNames = which === 'all' ? ['ad', 'teaser', 'wide'] : [which];
const workArg = args.work ?? process.env.LITMUS_DEMO_WORK;
if (!workArg) throw new Error('give --work <dir> (or set LITMUS_DEMO_WORK): a folder outside the tree for node_modules, the build and the output');
const work = resolve(workArg);
const footage = resolve(args.footage ?? join(work, 'footage'));
const outDir = resolve(args.out ?? join(work, 'out'));
const workers = Number(args.workers ?? 3);
const crf = String(args.crf ?? 16);

/** Install the pinned dependencies into the work folder when its lock differs from ours. */
const ensureDeps = () => {
  mkdirSync(work, { recursive: true });
  const lock = readFileSync(join(here, 'package-lock.json'), 'utf8');
  const theirs = existsSync(join(work, 'package-lock.json')) ? readFileSync(join(work, 'package-lock.json'), 'utf8') : '';
  if (lock === theirs && existsSync(join(work, 'node_modules'))) return;
  copyFileSync(join(here, 'package.json'), join(work, 'package.json'));
  copyFileSync(join(here, 'package-lock.json'), join(work, 'package-lock.json'));
  console.log(`installing dependencies in ${work}`);
  const r = spawnSync('npm', ['ci', '--no-audit', '--no-fund'], { cwd: work, stdio: 'inherit', shell: process.platform === 'win32' });
  if (r.status !== 0) throw new Error('npm ci failed');
};

/** Bundle the ad and its fonts into <work>/build, with the footage manifest the page reads. */
const build = async (req) => {
  const esbuild = req('esbuild');
  const dir = join(work, 'build');
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });
  await esbuild.build({
    entryPoints: { film: join(here, 'src/main.ts'), fonts: join(here, 'src/fonts.css') },
    bundle: true,
    outdir: dir,
    format: 'iife',
    target: 'chrome120',
    nodePaths: [join(work, 'node_modules')],
    loader: { '.svg': 'text', '.woff2': 'file', '.woff': 'file' },
    assetNames: 'fonts/[name]',
    logLevel: 'warning',
  });
  const manifestFile = join(footage, 'footage.json');
  if (!existsSync(manifestFile)) throw new Error(`no ${manifestFile}: run tools/prepare_footage.mjs first`);
  const manifest = JSON.parse(readFileSync(manifestFile, 'utf8'));
  const base = pathToFileURL(footage).href;
  // The measured GPU figures: data/ad/gpu_use.json, or the file --gpu-use names. With none the page has no GPU element.
  const gpuFile = args['gpu-use'] ? resolve(args['gpu-use']) : join(here, 'data/ad/gpu_use.json');
  if (args['gpu-use'] && !existsSync(gpuFile)) throw new Error(`no ${gpuFile}`);
  const gpuUse = existsSync(gpuFile) ? JSON.parse(readFileSync(gpuFile, 'utf8')) : null;
  console.log(gpuUse ? `GPU figures from ${gpuFile}${gpuUse.fixture ? ' (FIXTURE, not measured)' : ''}` : 'no GPU figures (data/ad/gpu_use.json is absent): the speed shot is drawn without the GPU panel');
  writeFileSync(
    join(dir, 'index.html'),
    `<!doctype html><html><head><meta charset="utf-8"><title>Litmus ad</title><link rel="stylesheet" href="fonts.css"><script>window.__footage=${JSON.stringify({ base, manifest })};window.__gpuUse=${JSON.stringify(gpuUse)}</script></head><body><script src="film.js"></script></body></html>`,
  );
  return join(dir, 'index.html');
};

const ffmpegArgs = (out, fps) => [
  '-y', '-loglevel', 'error',
  '-f', 'image2pipe', '-framerate', String(fps), '-c:v', 'png', '-i', '-',
  '-vf', 'scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p',
  '-c:v', 'libx264', '-preset', 'slow', '-crf', crf, '-maxrate', '14M', '-bufsize', '28M', '-profile:v', 'high', '-threads', '2',
  '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709', '-color_range', 'tv',
  out,
];

const run = (cmd, argv, opts = {}) =>
  new Promise((ok, fail) => {
    const p = spawn(cmd, argv, { stdio: ['ignore', 'inherit', 'inherit'], ...opts });
    p.on('exit', (code) => (code === 0 ? ok() : fail(new Error(`${cmd} exited ${code}`))));
  });

const openPage = async (browser, url, W, H) => {
  const context = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  page.on('pageerror', (e) => {
    console.error(`page error: ${e.message}`);
    process.exitCode = 1;
  });
  page.on('console', (m) => {
    if (m.type() === 'error') console.error(`console: ${m.text()}`);
  });
  await page.goto(url);
  await page.evaluate(() => window.__film.ready);
  const cdp = await context.newCDPSession(page);
  const shot = async (t) => {
    await page.evaluate((x) => window.__film.render(x), t);
    const { data } = await cdp.send('Page.captureScreenshot', { format: 'png', optimizeForSpeed: true });
    return Buffer.from(data, 'base64');
  };
  return { page, shot, close: () => context.close() };
};

/** loudnorm falls back to its dynamic mode when the gain it needs would push a peak past the ceiling, and that mode can leave overs (a teaser mixed at +1 dBFS true peak); a limiter at -2.5 dBFS after it keeps the encoded file under -2 dBTP with room for the codec. */
const LIMIT = 10 ** (-2.5 / 20);

/** Two-pass loudness normalization of the music mix into the video, to the timeline's target; returns the measured values. */
const mux = async (ffmpegPath, silent, wav, final, seconds, target) => {
  const first = spawnSync(ffmpegPath, ['-hide_banner', '-nostats', '-i', wav, '-af', `loudnorm=I=${target.integrated_lufs}:TP=${target.true_peak_dbtp}:LRA=11:print_format=json`, '-f', 'null', '-'], { encoding: 'utf8' });
  const m = /\{[\s\S]*"target_offset"[\s\S]*?\}/.exec(first.stderr);
  if (!m) throw new Error(`loudnorm gave no measurement:\n${first.stderr.slice(-800)}`);
  const s = JSON.parse(m[0]);
  const filter = `loudnorm=I=${target.integrated_lufs}:TP=${target.true_peak_dbtp}:LRA=11:measured_I=${s.input_i}:measured_TP=${s.input_tp}:measured_LRA=${s.input_lra}:measured_thresh=${s.input_thresh}:offset=${s.target_offset}:linear=true,alimiter=limit=${LIMIT}:level=disabled,atrim=0:${seconds},afade=t=out:st=${seconds - 0.5}:d=0.5`;
  await run(ffmpegPath, ['-y', '-loglevel', 'error', '-i', silent, '-i', wav, '-filter_complex', `[1:a]${filter}[a]`, '-map', '0:v', '-map', '[a]', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-movflags', '+faststart', '-shortest', final]);
  const check = spawnSync(ffmpegPath, ['-hide_banner', '-nostats', '-i', final, '-af', 'ebur128=peak=true', '-f', 'null', '-'], { encoding: 'utf8' });
  const summary = check.stderr.slice(check.stderr.lastIndexOf('Summary:'));
  const lufs = Number(/I:\s+(-?\d+(\.\d+)?) LUFS/.exec(summary)?.[1]);
  const peak = Number(/Peak:\s+(-?\d+(\.\d+)?) dBFS/.exec(summary)?.[1]);
  return { integrated_lufs: lufs, true_peak_dbfs: peak, lra: Number(/LRA:\s+(-?\d+(\.\d+)?) LU/.exec(summary)?.[1]) };
};

/** GitHub takes a video of up to 10 MB into a README on a free plan (docs.github.com, "Attaching files"); stay under it with room to spare. */
const README_BUDGET_BYTES = 9_500_000;

/**
 * The wide cut's encodes for the web, made from the finished file: `litmus_wide_site.mp4` (1920 by 1080, for the site's own
 * video element), `litmus_wide_readme.mp4` (1280 by 720, the highest quality that fits README_BUDGET_BYTES, for the
 * README's upload), and `litmus_wide_poster.png` (the last frame, the whole end card with the site's address, for the video element's poster and as
 * the README's linked image). All H.264 high profile, yuv420p, BT.709, AAC, with the index at the front (faststart).
 */
const webEncodes = async (ffmpegPath, final, timeline) => {
  const tags = ['-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709', '-color_range', 'tv'];
  const common = ['-y', '-loglevel', 'error', '-i', final, '-c:v', 'libx264', '-preset', 'slow', '-profile:v', 'high', '-pix_fmt', 'yuv420p', '-g', '60', ...tags, '-c:a', 'aac', '-ar', '48000', '-movflags', '+faststart'];
  const site = join(outDir, 'litmus_wide_site.mp4');
  await run(ffmpegPath, [...common, '-crf', '21', '-maxrate', '6M', '-bufsize', '12M', '-b:a', '128k', site]);
  const readme = join(outDir, 'litmus_wide_readme.mp4');
  let crf = 19;
  for (; crf <= 33; crf += 2) {
    await run(ffmpegPath, [...common, '-vf', 'scale=1280:720:flags=lanczos', '-crf', String(crf), '-maxrate', '3000k', '-bufsize', '6000k', '-b:a', '96k', readme]);
    if (statSync(readme).size <= README_BUDGET_BYTES) break;
  }
  if (statSync(readme).size > README_BUDGET_BYTES) throw new Error(`the README encode is over ${README_BUDGET_BYTES} bytes even at crf 33`);
  const seconds = timeline.cuts[timeline.aliases.wide].shots.reduce((a, shot) => a + shot.bars, 0) * timeline.beats_per_bar * (60 / timeline.bpm);
  const poster = join(outDir, 'litmus_wide_poster.png');
  // The poster is the last frame, the whole end card with the site's address.
  const posterAt = seconds - 0.1;
  await run(ffmpegPath, ['-y', '-loglevel', 'error', '-ss', String(posterAt), '-i', final, '-frames:v', '1', poster]);
  const sizes = { site_bytes: statSync(site).size, readme_bytes: statSync(readme).size, readme_crf: crf, poster_bytes: statSync(poster).size, poster_at_s: posterAt };
  console.log(`wide web encodes: ${JSON.stringify(sizes)}`);
  return sizes;
};

const renderCut = async (browser, htmlPath, name, ffmpegPath) => {
  const url = `${pathToFileURL(htmlPath).href}?cut=${name}`;
  const probe = await openPage(browser, url, 400, 400);
  const info = await probe.page.evaluate(() => ({ duration: window.__film.duration, W: window.__film.width, H: window.__film.height, fps: window.__film.fps, stills: window.__film.stills }));
  await probe.close();
  const fps = Number(args.fps ?? info.fps);
  mkdirSync(outDir, { recursive: true });
  const started = Date.now();

  if (args.stills) {
    const dir = join(outDir, 'stills', name);
    rmSync(dir, { recursive: true, force: true });
    mkdirSync(dir, { recursive: true });
    const p = await openPage(browser, url, info.W, info.H);
    for (const [i, s] of info.stills.entries()) {
      writeFileSync(join(dir, `${String(i + 1).padStart(2, '0')}_${s.name}.png`), await p.shot(s.t));
    }
    await p.close();
    console.log(`${name}: ${info.stills.length} stills in ${dir}`);
  }
  if (args['no-video']) return;

  const from = Number(args.from ?? 0);
  const to = Math.min(Number(args.to ?? info.duration), info.duration);
  const first = Math.round(from * fps);
  const total = Math.max(1, Math.ceil(to * fps) - first);
  const segDir = join(work, 'segments', name);
  rmSync(segDir, { recursive: true, force: true });
  mkdirSync(segDir, { recursive: true });
  const per = Math.ceil(total / workers);
  let done = 0;
  let reported = 0;
  const segs = [];
  await Promise.all(
    Array.from({ length: workers }, async (_, w) => {
      const a = first + w * per;
      const b = Math.min(first + total, a + per);
      if (a >= b) return;
      const seg = join(segDir, `seg_${w}.mp4`);
      segs[w] = seg;
      const enc = spawn(ffmpegPath, ffmpegArgs(seg, fps), { stdio: ['pipe', 'inherit', 'inherit'] });
      const finished = new Promise((ok, fail) => enc.on('exit', (code) => (code === 0 ? ok() : fail(new Error(`ffmpeg exited ${code}`)))));
      const p = await openPage(browser, url, info.W, info.H);
      for (let f = a; f < b; f++) {
        const png = await p.shot(f / fps);
        if (!enc.stdin.write(png)) await new Promise((ok) => enc.stdin.once('drain', ok));
        done++;
        const pct = Math.floor((done / total) * 20) * 5;
        if (pct > reported) {
          reported = pct;
          console.log(`${name}: ${pct}% (${done} of ${total} frames, ${((Date.now() - started) / 1000).toFixed(0)} s)`);
        }
      }
      enc.stdin.end();
      await finished;
      await p.close();
    }),
  );
  const list = join(segDir, 'list.txt');
  writeFileSync(list, segs.filter(Boolean).map((s) => `file '${s.replaceAll('\\', '/')}'`).join('\n'));
  const slice = args.from || args.to ? `_${from}-${to}s` : '';
  const silent = join(outDir, `litmus_${name}${slice}_silent.mp4`);
  await run(ffmpegPath, ['-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', list, '-c', 'copy', '-movflags', '+faststart', silent]);
  rmSync(segDir, { recursive: true, force: true });
  const secs = (Date.now() - started) / 1000;
  console.log(`${name}: ${total} frames at ${fps} fps, ${(total / fps).toFixed(1)} s, drawn and encoded in ${secs.toFixed(0)} s`);

  if (args['no-audio'] || args.from || args.to) return;
  const wav = join(work, 'audio', `music_${name}.wav`);
  mkdirSync(dirname(wav), { recursive: true });
  await run(process.execPath, [join(here, 'tools/music.mjs'), '--cut', name, '--out', wav]);
  const timeline = JSON.parse(readFileSync(join(here, 'data/ad/timeline.json'), 'utf8'));
  const final = join(outDir, `litmus_${name}.mp4`);
  const loud = await mux(ffmpegPath, silent, wav, final, info.duration, timeline.loudness);
  const bytes = statSync(final).size;
  const web = name === 'wide' ? await webEncodes(ffmpegPath, final, timeline) : undefined;
  const report = { cut: name, seconds: info.duration, fps, size: `${info.W}x${info.H}`, bytes, mbit_per_s: Number(((bytes * 8) / info.duration / 1e6).toFixed(2)), loudness: loud, ...(web ? { web } : {}), rendered_at: new Date().toISOString() };
  writeFileSync(join(outDir, `litmus_${name}.report.json`), JSON.stringify(report, null, 1));
  console.log(`${name}: ${final} ${JSON.stringify(report)}`);
};

ensureDeps();
if (args.install) process.exit(0);
const req = createRequire(join(work, 'package.json'));
const htmlPath = await build(req);
if (args.preview) {
  for (const n of cutNames) console.log(`${n}: ${pathToFileURL(htmlPath).href}?cut=${n}&play`);
  process.exit(0);
}
const { chromium } = req('playwright-core');
const ffmpegPath = req('ffmpeg-static');
const browser = await chromium.launch({
  headless: true,
  args: ['--disable-gpu', '--force-color-profile=srgb', '--font-render-hinting=none', '--hide-scrollbars', '--allow-file-access-from-files'],
});
try {
  if (args['web-only']) {
    // Remake only the wide cut's web encodes from its finished file (and record them in its report).
    const timeline = JSON.parse(readFileSync(join(here, 'data/ad/timeline.json'), 'utf8'));
    const web = await webEncodes(ffmpegPath, join(outDir, 'litmus_wide.mp4'), timeline);
    const reportFile = join(outDir, 'litmus_wide.report.json');
    writeFileSync(reportFile, JSON.stringify({ ...JSON.parse(readFileSync(reportFile, 'utf8')), web }, null, 1));
  } else for (const n of cutNames) await renderCut(browser, htmlPath, n, ffmpegPath);
} finally {
  await browser.close();
}
