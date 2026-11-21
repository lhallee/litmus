// Capture the built site and its in-browser checker in headless Chromium at a phone's size, for the ad's footage.
//
//   node capture/site_capture.mjs --site <built site folder> --text <text file> --out <dir> --work <work dir with node_modules>
//
// It serves the folder from a local server (127.0.0.1 only), loads the landing page and the checker, pastes the text,
// presses Check, and saves frames of the checker (JPEG, 1170 by 2532) while it reads, with a log of the clock and of
// where the pointer went. The checker is loaded at a pixel ratio of 1 and switched to 3 once it runs, because the page
// sizes its canvas when it starts and a headless browser started at 3 leaves the canvas at 1 (a test-harness quirk of
// the screenshot path, not a page defect: a person's browser reports its own ratio before the page loads).
// Nothing leaves the machine: the page, the model, and the text are all local.

import { createServer } from 'node:http';
import { createReadStream, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { extname, join, resolve } from 'node:path';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc), []));
for (const k of ['site', 'text', 'out', 'work']) if (!args[k]) throw new Error(`give --${k}`);
const site = resolve(args.site);
const out = resolve(args.out);
const { chromium } = createRequire(join(resolve(args.work), 'package.json'))('playwright-core');
mkdirSync(join(out, 'checker'), { recursive: true });

const W = 390;
const H = 844;
const SCALE = 3;
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.wasm': 'application/wasm', '.json': 'application/json', '.png': 'image/png' };
const server = createServer((req, res) => {
  const path = decodeURIComponent((req.url ?? '/').split('?')[0]);
  let file = join(site, path);
  if (existsSync(file) && statSync(file).isDirectory()) file = join(file, 'index.html');
  if (!file.startsWith(site) || !existsSync(file)) {
    res.writeHead(404).end();
    return;
  }
  res.writeHead(200, { 'content-type': MIME[extname(file)] ?? 'application/octet-stream' });
  createReadStream(file).pipe(res);
});
await new Promise((ok) => server.listen(0, '127.0.0.1', ok));
const base = `http://127.0.0.1:${server.address().port}`;

const browser = await chromium.launch({ headless: true, args: ['--force-color-profile=srgb', '--hide-scrollbars'] });
const log = [];
const clock = Date.now();
const now = () => (Date.now() - clock) / 1000;

// The landing page, at 3x: the first screen and the whole page for a slow scroll.
{
  const context = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: SCALE, colorScheme: 'dark' });
  const page = await context.newPage();
  await page.goto(`${base}/index.html`);
  await page.waitForLoadState('networkidle');
  await page.screenshot({ path: join(out, 'site_top.png') });
  await page.screenshot({ path: join(out, 'site_full.png'), fullPage: true });
  log.push({ what: 'landing', height: await page.evaluate(() => document.documentElement.scrollHeight) });
  await context.close();
}

const context = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: 1, colorScheme: 'dark' });
const page = await context.newPage();
page.on('pageerror', (e) => console.error('page error', e.message));
await page.goto(`${base}/app/`);
await page.waitForFunction(() => {
  const l = document.getElementById('loader');
  return !l || getComputedStyle(l).display === 'none';
}, null, { timeout: 120000 });
await page.waitForTimeout(1200);
const cdp = await context.newCDPSession(page);
await cdp.send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: SCALE, mobile: false });
await page.waitForTimeout(1500);
await page.mouse.move(5, 5);

let frame = 0;
const grab = async () => {
  const shot = await cdp.send('Page.captureScreenshot', { format: 'jpeg', quality: 92, optimizeForSpeed: true });
  writeFileSync(join(out, 'checker', `f${String(frame).padStart(4, '0')}.jpg`), Buffer.from(shot.data, 'base64'));
  log.push({ what: 'frame', frame, t: now() });
  frame++;
};
const hold = async (seconds) => {
  const end = now() + seconds;
  while (now() < end) await grab();
};

const text = readFileSync(resolve(args.text), 'utf8').trim();
await hold(0.9);
// The pointer glides to the box and clicks it, as a person's would; the glide is logged, and the footage has no pointer.
const glide = async (x, y, seconds) => {
  const from = log.findLast((e) => e.pointer)?.pointer ?? [5, 5];
  log.push({ what: 'glide', t: now(), from, to: [x, y], seconds, pointer: [x, y] });
  const steps = Math.max(2, Math.round(seconds * 20));
  const t0 = now();
  for (let i = 1; i <= steps; i++) {
    const k = i / steps;
    const e = k * k * (3 - 2 * k);
    await page.mouse.move(from[0] + (x - from[0]) * e, from[1] + (y - from[1]) * e);
    await grab();
    const wait = t0 + (seconds * i) / steps - now();
    if (wait > 0) await page.waitForTimeout(wait * 1000);
  }
};
await glide(195, 400, 0.9);
await page.mouse.click(195, 400);
log.push({ what: 'click_box', t: now(), at: [195, 400] });
await hold(0.5);
// A paste as the page receives one from Ctrl+V: a `paste` event on the document that carries the text. Typed text
// (`keyboard.insertText`) goes through the page's one-line text agent, which turns every paragraph break into spaces,
// and the page then counts other tokens than the app does for the same text.
await page.evaluate((pasted) => {
  const data = new DataTransfer();
  data.setData('text/plain', pasted);
  document.dispatchEvent(new ClipboardEvent('paste', { clipboardData: data, bubbles: true, cancelable: true }));
}, text);
log.push({ what: 'paste', t: now() });
await hold(1.4);
await glide(76, 765, 0.9);
await page.mouse.click(76, 765);
log.push({ what: 'click_check', t: now(), at: [76, 765] });
await hold(7);
log.push({ what: 'result_settled', t: now() });
await hold(1.0);
await page.mouse.wheel(0, 500);
log.push({ what: 'scroll', t: now() });
await hold(1.6);
writeFileSync(join(out, 'site.json'), JSON.stringify({ viewport: [W, H], scale: SCALE, events: log }, null, 1));
await browser.close();
server.close();
console.log('captured', frame, 'frames in', now().toFixed(1), 's');
