// The site's one request to another address, checked: `countOpening` in site/site.js asks the counter for its pixel
// only on the browser page, only when links.json names a counter, never for a browser that asks not to be tracked,
// an automated browser, or a page off HTTPS, and with nothing in the request but the page's name.
//
//     node --test tools/security/tests/site_count.test.mjs      (python tools/run_tests.py site)
//
// site.js runs here against stand-ins for the page: no browser, no network.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

const SOURCE = readFileSync(new URL("../../../site/site.js", import.meta.url), "utf8");
const COUNTER = "https://litmus.goatcounter.com/count";

// Run site.js as a page would and return every request it made to an address that is not the site's own data.
async function requests({ counter = COUNTER, app = true, protocol = "https:", navigator = {} } = {}) {
  const outside = [];
  const frame = { getAttribute: () => "app/", hidden: true, src: "" };
  const page = {
    document: {
      documentElement: { classList: { add() {} }, setAttribute() {}, getAttribute: () => null },
      readyState: "complete",
      querySelectorAll: () => [],
      querySelector: (selector) => (selector === "iframe[data-app]" ? (app ? frame : null) : {}),
      getElementById: () => null,
    },
    localStorage: { getItem: () => null, setItem() {} },
    window: {},
    navigator,
    location: { protocol, hostname: "lhallee.github.io" },
    fetch: async (address, options) => {
      if (address === "assets/links.json") return { ok: true, json: async () => ({ schema: 1, page_counter: counter }) };
      if (address === "assets/results.json") return { ok: true, json: async () => ({}) };
      outside.push({ address, options });
      return { ok: true };
    },
  };
  vm.runInNewContext(SOURCE, page);
  for (let turn = 0; turn < 5; turn += 1) await new Promise((resolve) => setImmediate(resolve));
  return outside;
}

test("the browser page counts its opening once, with the page's name and nothing else", async () => {
  const made = await requests();
  assert.equal(made.length, 1);
  assert.match(made[0].address, /^https:\/\/litmus\.goatcounter\.com\/count\?p=%2Fuse&rnd=[a-z0-9]+$/);
  assert.deepEqual({ ...made[0].options }, { mode: "no-cors", credentials: "omit", referrerPolicy: "no-referrer", cache: "no-store" });
});

test("nothing is counted while the release names no counter", async () => {
  assert.deepEqual(await requests({ counter: "" }), []);
});

test("only the page that holds the checker counts", async () => {
  assert.deepEqual(await requests({ app: false }), []);
});

test("a browser that asks not to be tracked, an automated one, and a page off HTTPS are not counted", async () => {
  assert.deepEqual(await requests({ navigator: { doNotTrack: "1" } }), []);
  assert.deepEqual(await requests({ navigator: { globalPrivacyControl: true } }), []);
  assert.deepEqual(await requests({ navigator: { webdriver: true } }), []);
  assert.deepEqual(await requests({ protocol: "http:" }), []);
});
