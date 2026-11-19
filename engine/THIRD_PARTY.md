# Third-party code in the Litmus engine

Litmus itself is under PolyForm Noncommercial 1.0.0 ([../LICENSE](../LICENSE), [../LICENSING.md](../LICENSING.md)).
The crates below are compiled into its binaries. Every one is under a permissive license (MIT, Apache-2.0, BSD,
Zlib, Unicode-3.0, Unlicense or 0BSD, often with a choice among them); **none is copyleft and none restricts the field
of use**, so each can be redistributed inside a noncommercial release with its notice kept. Versions are those of
`Cargo.lock` on 2026-11-17; `cargo metadata` regenerates the list.

## `litmus-engine` (the library: native, WebAssembly and the Python binding)

| Crate | Version | License | Why |
| --- | --- | --- | --- |
| serde, serde_core, serde_derive | 1.0.229 | MIT OR Apache-2.0 | Reading `config.json` and `tokenizer.json`; serializing readings |
| serde_json | 1.0.154 | MIT OR Apache-2.0 | JSON |
| itoa, zmij | 1.0.19, 1.0.23 | MIT OR Apache-2.0; MIT | Number formatting inside serde_json |
| memchr | 2.8.3 | Unlicense OR MIT | Byte search inside serde_json |
| unicode-normalization | 0.1.26 | (MIT OR Apache-2.0) AND Unicode-3.0 | NFC of the standard reading and the tokenizer |
| tinyvec | 1.13.3 | Zlib OR Apache-2.0 OR MIT | Inside unicode-normalization |
| unicode-properties | 0.1.5 | (MIT OR Apache-2.0) AND Unicode-3.0 | Letter and number classes of the pre-tokenizer |
| proc-macro2, quote, syn, unicode-ident | build time | MIT OR Apache-2.0 (unicode-ident adds Unicode-3.0) | serde's derive macros; not in the binary |

Development only (benchmarks): criterion 0.5 and its dependencies, MIT OR Apache-2.0.

## `litmus-cli` (the `litmus` command), in addition

| Crate | Version | License | Why |
| --- | --- | --- | --- |
| sha2, digest, block-buffer, crypto-common, generic-array, typenum, cpufeatures, cfg-if | 0.10 line | MIT OR Apache-2.0 (generic-array MIT) | Content hashes of the reading cache |
| miniz_oxide, adler2, simd-adler32 | 0.8.9 | MIT OR Zlib OR Apache-2.0; 0BSD OR MIT OR Apache-2.0; MIT | Inflating `.docx` archives |
| pdf-extract | 0.7.12 | MIT | Text of PDF files |
| lopdf | 0.34.0 | MIT | PDF parsing, inside pdf-extract |
| adobe-cmap-parser, type1-encoding-parser, postscript, pom, euclid, rangemap | | MIT, or MIT/Apache-2.0 | Fonts and encodings, inside pdf-extract |
| encoding_rs | 0.8.44 | (Apache-2.0 OR MIT) AND BSD-3-Clause | Legacy text encodings in PDFs |
| flate2, crc32fast, zlib-rs, miniz_oxide 0.9 | | MIT OR Apache-2.0; Zlib | Compressed PDF streams |
| nom, minimal-lexical, md-5, weezl, log, indexmap, hashbrown, equivalent, time, time-core, deranged, powerfmt, num-conv, num-traits, scopeguard, libc, core_detect, multiversion_no_op, simdutf8 | | MIT, or MIT OR Apache-2.0 | Inside lopdf and pdf-extract |

## `litmus-files` (shared by the command line and the app)

The crates the command line and the app use for files: sha2 and its dependencies,
miniz_oxide, pdf-extract and lopdf with theirs (native only; the WebAssembly build leaves PDF out), as listed above.

## `litmus-app` (the app in `../app`, its own workspace and `Cargo.lock`), in addition

Versions are those of `app/Cargo.lock` on 2026-11-17. The native build on Windows compiles 216 crates and the
WebAssembly build 150 (223 native with quick check's crates); `cargo metadata --filter-platform <target>` lists them.

| Crate | Version | License | Why |
| --- | --- | --- | --- |
| eframe, egui, egui-winit, egui_glow, epaint, emath, ecolor | 0.37.0 | MIT OR Apache-2.0 | The UI on desktop and in the browser |
| epaint_default_fonts | 0.37.0 | (MIT OR Apache-2.0) AND OFL-1.1 AND Ubuntu-font-1.0 | egui's built-in fonts; the three the app ships are listed under "Fonts" below |
| winit, glutin, glutin_wgl_sys, glutin_egl_sys, glutin_glx_sys, accesskit_winit | | Apache-2.0 | Windows and OpenGL contexts, accessibility (native) |
| glow, glutin-winit, raw-window-handle, accesskit and its platform crates | | MIT, or MIT OR Apache-2.0 (glow adds Zlib) | OpenGL and the screen-reader tree |
| rfd | 0.18.0 | MIT | Native open and save dialogs |
| global-hotkey | 0.10.0 | Apache-2.0 OR MIT | Quick check's hotkey (opt-in) |
| tray-icon, muda, keyboard-types, dpi | 0.26.1, 0.21.2, 0.8.3 | MIT OR Apache-2.0 | Quick check's tray icon and menu (opt-in) |
| windows (features for UI Automation, the clipboard, input, the registry) | 0.62.2 | MIT OR Apache-2.0 | Quick check on Windows: the selected text, the clipboard saved and put back, the Explorer entry |
| clipboard-win, error-code | | BSL-1.0 | The Windows clipboard, inside egui-winit |
| libloading | | ISC | Loading OpenGL, inside glutin |
| self_cell | 1.3.0 | Apache-2.0 OR GPL-2.0-only (Apache-2.0 taken) | Inside the accessibility tree |
| r-efi | | MIT OR Apache-2.0 OR LGPL-2.1-or-later (MIT taken) | Inside getrandom; not used on these targets |
| option-ext | 0.2.0 | MPL-2.0 | Inside dirs, which tray-icon uses on Linux; not compiled into the Windows build. MPL-2.0's copyleft is per file and is met by using the crate unmodified, as published on crates.io |
| windows, windows-sys and their parts | | MIT OR Apache-2.0 | Win32 calls (native, Windows) |
| wasm-bindgen, wasm-bindgen-futures, js-sys, web-sys | 0.2.126, 0.4, 0.3.103 | MIT OR Apache-2.0 | The browser build; wasm-bindgen-cli 0.2.126 (MIT OR Apache-2.0) writes the glue and is not shipped |
| The rest (ab_glyph, ahash, bytemuck, smallvec, log, profiling, and the like) | | MIT, Apache-2.0, Zlib, Unicode-3.0, Unlicense, 0BSD or BSD, alone or as a choice | Inside the above |

| miniz_oxide (build script) | 0.8 | MIT OR Zlib OR Apache-2.0 | `app/build.rs` compresses the 256-pixel image of the Windows icon resource; nothing of it is linked |

One crate of the app is copyleft without a permissive choice: option-ext (MPL-2.0, file-level copyleft), only in the
Linux build, as the table says. `tools/security/check_dependencies.py` checks every locked crate's license against this
file.

**Fonts.** The app enables eframe's `default_fonts` feature and not `monochrome_emoji_fonts`, so these three font
files from `epaint_default_fonts` 0.37.0 are compiled into every build (desktop and browser):

| Font | File | License | Use |
| --- | --- | --- | --- |
| Ubuntu Light | `Ubuntu-Light.ttf` (362 KB) | Ubuntu Font Licence 1.0 (allows redistribution inside software; `fonts/UFL.txt` in the crate) | The UI text in the browser, and on the desktop where the system font below is missing |
| Hack Regular | `Hack-Regular.ttf` (309 KB) | MIT for Hack, with the Bitstream Vera License of its base (`fonts/Hack-Regular.txt`) | Numbers and code in the browser, and the desktop's fallback |
| egui icons | `egui-icons.ttf` (3 KB) | MIT, a subset of emoji-icon-font (`fonts/emoji-icon-font-mit-license.txt`) | Five logos egui exposes; the app shows none |

On Windows the desktop app also reads Segoe UI and Segoe UI Semibold (`segoeui.ttf`, `seguisb.ttf`) and Cascadia
Mono or Consolas from the system's own font folder at start (`app/src/theme.rs`). These are Microsoft's fonts,
installed with Windows; the app neither ships nor copies them, as a web page uses the visitor's `system-ui` font.
The Ubuntu Font Licence is the one license here outside MIT, Apache-2.0, BSD, Zlib and OFL.

## `litmus-py` (the Python module `litmus_engine`), in addition

| Crate | Version | License | Why |
| --- | --- | --- | --- |
| pyo3, pyo3-ffi, pyo3-macros, pyo3-macros-backend, pyo3-build-config | 0.25.1 | MIT OR Apache-2.0 | The Python binding |
| once_cell, portable-atomic, memoffset, indoc, unindent, heck, libc | | MIT OR Apache-2.0 (memoffset MIT) | Inside pyo3 |
| sha2 and its dependencies | 0.10 line | MIT OR Apache-2.0 (generic-array MIT) | The release version when its provenance names none |
| target-lexicon | 0.13.6 | Apache-2.0 WITH LLVM-exception | pyo3's build script; not in the binary |

Build tools, not linked: maturin 1.16.0 (MIT OR Apache-2.0), and for the Linux wheel its Docker image
`ghcr.io/pyo3/maturin:v1.16.0` (maturin on the PyPA manylinux2014 image), which compiles the wheel and is not
distributed with it.

## Data

- The tokenizer, `tokenizer.json` of `answerdotai/ModernBERT-large` (Apache-2.0), is read at run time and not
  distributed in this tree.
- `litmus-engine/src/unicode14.rs` lists the code points assigned in Unicode 14.0, generated from Python's
  `unicodedata` (Unicode Data, Unicode-3.0 license).
