---
name: litmus
description: Checks whether a text, document, or piece of code looks human-written, AI-written, or mixed, with Litmus, a local detector that says how sure it is and runs with no network. Use when asked whether writing or code was produced by an AI, to screen generated output before it is sent or committed, to check an essay, report, README, or source file for AI authorship, or to add an AI-text check to a workflow. Installs the litmus program when it is missing.
license: PolyForm-Noncommercial-1.0.0
compatibility: Windows x86-64, Linux x86-64, or macOS on Apple silicon or Intel (the macOS program is run only on the release workflow's runners). Needs no network after the install, and the install downloads one file from github.com/lhallee/litmus releases.
---

# Litmus

Litmus is a free detector of AI written text that says how sure it is. It reads a passage and calls it human, AI, or
mixed (written by a person and a model together), and when the evidence cannot separate the origins it says so and lists
the ones that remain. It runs on this computer's processor, with no graphics card, and sends the text nowhere.

A reading is evidence about where a text came from, never proof of who wrote it. Read [Trust](#trust) before you act on
one.

## Check a text

1. **See what is available.** If your tools include `litmus_check_text` and `litmus_check_file` (from the `litmus` MCP
   server), use them and skip to step 3. Otherwise run `litmus info`. If that prints the model, use the command line.
2. **Install it when it is missing.** Run the script in this skill's `scripts/` folder for your system. It downloads one
   file from this project's GitHub releases, checks it against the release's published file list, installs it for the
   current user without elevation, and prints what it did, including the full path of `litmus`:

   ```sh
   sh scripts/install.sh                                                  # Linux, macOS
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install.ps1   # Windows
   ```

   If the script cannot download (no release yet for this system), or you want to build it, see
   [From source](#from-source). If the install put `litmus` outside `PATH`, call it by the full path the script printed.
   To offer the tools to your agent as well, register the server (`litmus mcp`) with the commands for Claude Code, Codex,
   and Antigravity in [references/clients.md](references/clients.md).
3. **Read a text, a file, or a folder.**

   ```sh
   litmus score --json --no-cache essay.md          # one JSON object a line, one line per file
   printf '%s' "$TEXT" | litmus score --json --no-cache -     # standard input
   ```

   With the MCP tools, call `litmus_check_text` with `{"text": "..."}` or `litmus_check_file` with an absolute `path`.
   Add `"detail": "full"` for every chunk, window, and heat-map cell, since the default `"summary"` returns the verdict
   alone. `--no-cache` keeps `litmus score` from writing a cache of readings to the user's cache folder, and the MCP
   server never writes one. `litmus score` exits 0 when every input read, 1 when any failed, and 2 for a usage error. It never exits
   non-zero because of a verdict, so decide from the JSON.
4. **Report the reading, not a ruling.** Say the label, the origins not ruled out, and the chance of AI with its range,
   and say when the reading makes no call. Use the sentence template in [Say it plainly](#say-it-plainly).

## Read the result

The fields that matter, in `reading.verdict` of the CLI's JSON and of the MCP result's `structuredContent` unless the
row says otherwise:

| Field | Meaning |
| --- | --- |
| `label` (beside `reading`) | `human`, `AI`, or `mixed` when one origin remains at the 10% level; `AI or mixed` when human is ruled out and the other two remain (a model wrote or edited the text, and the reading cannot say whether all of it or part); `uncertain` when human remains beside another; `unlike reference` when none does. The last two make no call. Litmus 1.1.0 and earlier printed `uncertain` for `AI or mixed` |
| `sets["0.1"]`, `sets["0.05"]` | The origins the reading cannot rule out at the 10% and the 5% level. A wider set means less evidence |
| `chance` (beside `reading`): `value`, `interval` | The chance of AI, the number to report: how likely it is that a language model wrote or edited some of the text, taking a text to be as likely human as not beforehand, with its range. The range is how far the chance moves from one kind of text to another (it holds the chance for 90% of pairings of a human source with an AI or mixed source), so it is short near 0% or 100% and wide in the middle: always say it with the chance. Litmus 1.1.0 and earlier gave only the calibration's own interval, usually under a point wide |
| `probability`, `interval` | The same evidence before it is restated at even odds: the probability of not human among texts mixed as the calibration texts are, where `chance.reference_share` of them are not human. Report `chance`, not this |
| `out_of_reference` | The text resembles none of the reference texts closely; treat the reading as no evidence either way |
| `mixed_document` | One chunk reads as model-written and another as human; ask for `detail: "full"` and read `cells` |
| `chunks`, `n_tokens` | A text over 8,192 tokens is read in equal chunks and the verdict combines them |
| `counts` (CLI), `words` and `characters` (MCP) | The text's word, character, and token counts, for reporting its size; they carry no evidence about its origin |
| `cells` (full detail) | The heat map, one cell per 128 tokens with `start_char` and `end_char`; it shows where the reading changes, and the verdict decides, not the most extreme cell |

The MCP result also carries `caveats` (the ones this reading triggers) and a standing `guidance` sentence. Every field
is explained in [references/reading-results.md](references/reading-results.md).

## Trust

- **Probabilistic.** The 10% level promises that among human texts like the calibration texts, at most 10% have `human`
  ruled out at the 10% level (5% at the 5% level). That bounds false flags and says nothing about how much AI text is
  caught.
- **Short texts are less reliable.** A sentence or two carries little evidence, so the set stays wide and the chance of
  AI stays nearer one half. The
  MCP result flags a text under 128 tokens, which marks where one sentence or two ends and is not a measured cliff.
- **English only so far.** This release is an English detector: <!--litmus:record.corpus.english_share as pct1-->98.8%<!--/litmus--> of the training text is English,
  and it was tested mainly on English text. Other languages are not supported yet, and a reading of one is not evidence.
- **Code is weaker than prose.** Source files read as plain text, and the reference texts are mostly prose.
- **Text unlike the references.** New models, rare formats, heavy editing after generation, and quoted or templated text
  fall outside what it learned. Every human reference text predates 2022, and long AI text is scarce in public data.
- **An accuracy figure holds for its test set only.** For the release model, <!--litmus:t:accuracy_summary-->on held-out test texts it wrongly called 7.4% of human texts not human at the 5% level, and it caught 67.2% of AI texts when 5% of human texts are wrongly flagged (53.0% at 1%)<!--/litmus-->. A model built from source with
  random placeholder weights (`litmus info` says so) reads meaninglessly. Never quote an accuracy from memory.
- **Not an accusation.** A reading is not proof that a person used a model, so do not report it as one. What to do with
  a reading is the user's decision; our recommendation is not to use one alone for a decision about a person, and to
  weigh it with what it cannot see: drafts, version history, the writer's other work, and a conversation with the
  writer.

## Say it plainly

ALWAYS state a reading in this form, with the numbers from the result:

> Litmus reads this as **uncertain**: it cannot rule out human or AI at the 10% level. The chance of AI is 50%, likely
> between <!--litmus:ranges.at_50_low-->19<!--/litmus-->% and <!--litmus:ranges.at_50_high-->81<!--/litmus-->%, over 214 tokens. That is no call, and Litmus is weaker on short texts and on anything but English.

Give the chance as a whole percentage from `chance.value` with its range from `chance.interval` ("50%, likely between
<!--litmus:ranges.at_50_low-->19<!--/litmus-->% and <!--litmus:ranges.at_50_high-->81<!--/litmus-->%"), as the MCP result's sentence does. Leave the range out only when its ends round to one percentage. A
wide range means the same reading could lean either way in another kind of text, so say so and make no more of the
number than the range allows.

Use "human", "AI", or "mixed" for a call, and "unlike its reference texts, so no evidence either way" for
`unlike reference`. Include the reason a caveat applies (short, code, not English). Never say "this was written by an
AI" or "this person used AI".

## In a workflow

- We recommend checking generated prose or code once, after it is final, and treating the reading as one signal. We
  recommend against a loop of "regenerate until Litmus says human": it optimizes against the detector and tells you
  nothing about the text.
- A gate on a reading will be wrong in both directions. If a pipeline acts on one, we recommend gating on the set and
  the range (for example, flagging only when the set rules human out and the low end of `chance.interval` is high),
  and routing every flag to a person.
- Check files one at a time with the MCP tools, or run `litmus score --json` over a folder for a batch, which reads `.txt`,
  `.md`, `.html`, `.docx`, and `.pdf` (PDF text layer only, no OCR).
- The MCP tools read at most 131,072 tokens a text and one file at a time. Split longer inputs.

## From source

Needs Rust (the toolchain in `engine/rust-toolchain.toml`, which rustup installs) and a release folder holding
`model.safetensors`, `config.json`, and `tokenizer.json`:

```sh
cd engine && cargo build --release --locked      # writes engine/target/release/litmus
litmus score --model <release folder> --json --no-cache essay.md       # or set LITMUS_MODEL
litmus mcp --model <release folder>
```

A release's own `litmus` has the model built in and needs neither flag. Where the release folders are published is
<!--litmus:t:hub_model_link-->https://huggingface.co/lhallee/litmus<!--/litmus-->.
