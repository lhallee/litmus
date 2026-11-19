// Drive the engine's WebAssembly probe in Node (V8, the engine of Chrome): check the SIMD128 kernel against the
// scalar path, time the ternary product, and time whole-document scoring.
//
//   node scripts/wasm_probe.mjs <wasm_probe.wasm> <release folder> <tokenizer.json> [text file]

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { performance } from "node:perf_hooks";

const [wasmPath, release, tokenizerPath, textPath] = process.argv.slice(2);
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const probe = instance.exports;

function put(bytes) {
  const pointer = probe.probe_alloc(bytes.length);
  new Uint8Array(probe.memory.buffer, pointer, bytes.length).set(bytes);
  return [pointer, bytes.length];
}

const mismatches = probe.kernel_check();
console.log(`simd128 against scalar: ${mismatches === 0 ? "equal on every shape" : mismatches === 0xffffffff ? "built without simd128" : `${mismatches} shapes differ`}`);

for (const [rows, cols] of [[192, 192], [768, 192], [192, 768], [384, 1536]]) {
  const tokens = 256;
  probe.kernel_run(rows, cols, tokens, 1);
  const repeats = 20;
  const started = performance.now();
  probe.kernel_run(rows, cols, tokens, repeats);
  const seconds = (performance.now() - started) / 1e3;
  console.log(`ternary ${rows}x${cols}, ${tokens} tokens: ${((rows * cols * tokens * repeats) / seconds / 1e9).toFixed(1)} G adds/s`);
}

if (probe.column_run) {
  // The feed-forward's down product on 64 tokens with activations zero at a given rate: dense path, column path.
  for (const [rows, cols] of [[192, 768], [384, 1536]]) {
    const cells = [];
    for (const zeros of [0, 50, 70, 80, 90, 95]) {
      const micros = [0, 1].map((columns) => {
        probe.column_run(rows, cols, 64, zeros, 1, columns);
        const repeats = 20;
        const started = performance.now();
        probe.column_run(rows, cols, 64, zeros, repeats, columns);
        return ((performance.now() - started) * 1e3) / repeats / 64;
      });
      cells.push(`${zeros}% ${micros[0].toFixed(2)} / ${micros[1].toFixed(2)}`);
    }
    console.log(`down ${rows}x${cols}, us a token, dense / column path: ${cells.join(", ")}`);
  }
}

const attentionMismatches = probe.attention_check();
console.log(`attention (float and integer) and float64 products against the portable path: ${attentionMismatches === 0 ? "equal on every shape" : `${attentionMismatches} shapes differ`}`);
for (const [d, heads] of [[192, 3], [384, 6]]) {
  for (const integer of [0, 1]) {
    for (const portable of [1, 0]) {
      probe.attention_run(256, d, heads, 1, integer, portable);
      const repeats = 20;
      const started = performance.now();
      probe.attention_run(256, d, heads, repeats, integer, portable);
      const micros = ((performance.now() - started) * 1e3) / repeats;
      console.log(`${integer ? "integer" : "float"} attention, 256 tokens, d ${d}, ${portable ? "portable" : "fastest"}: ${micros.toFixed(0)} us a block`);
    }
  }
  const repeats = 50;
  probe.norm_run(256, d, 1);
  const started = performance.now();
  probe.norm_run(256, d, repeats);
  console.log(`norm and quantize, 256 rows of ${d}: ${(((performance.now() - started) * 1e3) / repeats).toFixed(0)} us`);
}

const model = put(readFileSync(join(release, "model.safetensors")));
const config = put(readFileSync(join(release, "config.json")));
const tokenizer = put(readFileSync(tokenizerPath));
const loadStart = performance.now();
if (probe.engine_load(...model, ...config, ...tokenizer) !== 0) throw new Error("engine_load failed");
console.log(`load: ${(performance.now() - loadStart).toFixed(0)} ms`);
let readMillis = Infinity;
for (let repeat = 0; repeat < 4; repeat++) {
  const readStart = performance.now();
  probe.engine_read_pass(8192);
  readMillis = Math.min(readMillis, performance.now() - readStart);
}
console.log(`read pass (head and bank) for a chunk of 8,192 tokens and its 61 windows: ${readMillis.toFixed(1)} ms (the best of 4)`);

if (textPath) {
  // A real document, read whole: what a trained release's activations look like.
  // With the column path build, the dense path alone and the measured choice alternate run by run.
  const [pointer, length] = put(readFileSync(textPath));
  probe.engine_score(pointer, length);
  const modes = probe.engine_dense_only ? [["measured", 0], ["dense only", 1]] : [["this build", 0]];
  const times = Object.fromEntries(modes.map(([name]) => [name, []]));
  let tokens = 0;
  for (let repeat = 0; repeat < 5; repeat++) {
    for (const [name, dense] of modes) {
      if (probe.engine_dense_only) probe.engine_dense_only(dense);
      const started = performance.now();
      tokens = probe.engine_score(pointer, length);
      times[name].push((performance.now() - started) / 1e3);
    }
  }
  if (probe.engine_dense_only) probe.engine_dense_only(0);
  for (const [name] of modes) {
    const micros = times[name].map((seconds) => ((seconds * 1e6) / tokens).toFixed(1));
    console.log(`score ${textPath}, ${tokens} tokens, ${name}: ${micros.join(", ")} us a token (5 runs)`);
  }
}

const words = "the river model carried a quiet ledger of signals through careful measure and several readers found that during long winters".split(" ");
for (const target of [1024, 8192]) {
  let text = "";
  let state = 7;
  while (text.length < target * 5.2) {
    state = (state * 1103515245 + 12345) % 2147483648;
    text += words[state % words.length] + (state % 11 === 0 ? ". " : " ");
  }
  const [pointer, length] = put(new TextEncoder().encode(text));
  probe.engine_score(pointer, length);
  let seconds = Infinity;
  let tokens = 0;
  for (let repeat = 0; repeat < 3; repeat++) {
    const started = performance.now();
    tokens = probe.engine_score(pointer, length);
    seconds = Math.min(seconds, (performance.now() - started) / 1e3);
  }
  console.log(`score ${tokens} tokens: ${(tokens / seconds).toFixed(0)} tokens/s (${seconds.toFixed(3)} s, the best of 3)`);
}
