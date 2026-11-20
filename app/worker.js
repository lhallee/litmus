// The private page's Web Worker. It loads the same WebAssembly module as the page, fetches the release's three files
// from this page's own origin, and scores the texts the page posts to it, so the page never freezes. The three GETs
// below are the only network requests the app makes; a text never leaves the browser. Cancelling a reading
// terminates this worker and starts a new one. While the files download it reports the bytes so far and in all.
import init, { WorkerEngine } from "./litmus_app.js";

const FILES = ["model.safetensors", "config.json", "tokenizer.json"];
let engine = null;

// Fetch every file at once, reading each body as a stream so that the page can show the download's progress.
async function getAll(names) {
  const responses = await Promise.all(
    names.map((name) => fetch(new URL(name, self.location.href), { method: "GET", credentials: "same-origin" })),
  );
  responses.forEach((response, index) => {
    if (!response.ok) {
      throw new Error(`${names[index]}: HTTP ${response.status}`);
    }
  });
  const sizes = responses.map((response) => Number(response.headers.get("Content-Length")) || 0);
  const total = sizes.reduce((a, b) => a + b, 0);
  const done = names.map(() => 0);
  let last = 0;
  const report = (force) => {
    const now = performance.now();
    if (force || now - last >= 100) {
      last = now;
      self.postMessage({ type: "download", done: done.reduce((a, b) => a + b, 0), total });
    }
  };
  return Promise.all(
    responses.map(async (response, index) => {
      if (!response.body) {
        const bytes = new Uint8Array(await response.arrayBuffer());
        done[index] = bytes.length;
        report(true);
        return bytes;
      }
      const reader = response.body.getReader();
      const parts = [];
      for (;;) {
        const { done: finished, value } = await reader.read();
        if (finished) {
          break;
        }
        parts.push(value);
        done[index] += value.length;
        report(false);
      }
      const bytes = new Uint8Array(done[index]);
      let at = 0;
      for (const part of parts) {
        bytes.set(part, at);
        at += part.length;
      }
      report(true);
      return bytes;
    }),
  );
}

function words(error) {
  return String((error && error.message) || error);
}

self.onmessage = async ({ data }) => {
  if (data.type === "load") {
    try {
      await init();
      const [model, config, tokenizer] = await getAll(FILES);
      engine = new WorkerEngine(model, config, tokenizer);
      self.postMessage({ type: "loaded", config: new TextDecoder().decode(config), version: engine.version() });
    } catch (error) {
      self.postMessage({ type: "error", message: `The model did not load: ${words(error)}` });
    }
  } else if (data.type === "score") {
    const job = data.job;
    let last = 0;
    try {
      if (!engine) {
        throw new Error("The model is not loaded.");
      }
      const started = performance.now();
      const json = engine.score(data.text, (tokens_done, tokens_total, chunks_done, chunks_total, bytes_done, bytes_total) => {
        const now = performance.now();
        if (now - last >= 50) {
          last = now;
          self.postMessage({ type: "progress", job, tokens_done, tokens_total, chunks_done, chunks_total, bytes_done, bytes_total });
        }
      });
      self.postMessage({ type: "reading", job, json, ms: performance.now() - started });
    } catch (error) {
      self.postMessage({ type: "error", job, message: words(error) });
    }
  } else if (data.type === "surprise") {
    // The surprise reading, only when the page shows a reading's details; the text stays in this worker.
    const job = data.job;
    try {
      if (!engine) {
        throw new Error("The model is not loaded.");
      }
      self.postMessage({ type: "surprise", job, json: engine.surprise(data.text) });
    } catch (error) {
      self.postMessage({ type: "error", job, message: words(error) });
    }
  }
};
