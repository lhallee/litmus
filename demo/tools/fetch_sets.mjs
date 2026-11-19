// Choose and fetch the two sets of long documents the speed segment reads: whole works of popular genre fiction written by
// people and whole works of popular genre fiction written by a language model, each with provenance that can be stated.
//
//   node tools/fetch_sets.mjs --out <dir> [--per-set 60] [--min-chars 12000] [--max-chars 40000] [--user-agent "<contact>"]
//
// Written by people: single short stories and novelettes from Project Gutenberg (ebooks whose subjects include "Short
// stories", one story to an ebook), mostly magazine fiction of the 1930s to 1960s. Each ebook's header names its original
// publication and year, recorded per story, so the authorship is certain: every one was printed decades before any
// language model. Public domain in the United States. Only ebooks numbered 66143 to 79956 are drawn: Litmus trained on
// manu/project_gutenberg at revision 164853d2, whose English split ends at ebook 66142 (posted August 2021), so no story
// here comes from that training source. The catalog and the texts come from a Project Gutenberg mirror, as Gutenberg's
// robot policy asks, one request every 2 seconds. A text keeps everything between Gutenberg's start and end markers except
// what Gutenberg's volunteers added (transcriber's notes and "Notes on the Text", illustration markers, the underscores that
// mark italics, rule lines at either end); the author's own footnotes stay. A story whose notes cannot be separated from
// it, or whose header names no original publication before 2020, is left out.
//
// Written by AI: stories from LongStory (Hugging Face THU-KEG/LongStory, file sft_premise.jsonl at a pinned revision,
// checked by its SHA-256; MIT license, stated in the StoryWriter paper, arXiv 2506.16445), each written by GPT-4o-mini
// through the StoryWriter multi-agent pipeline from a one-sentence premise. One generator, so the set is balanced across
// generators trivially. The pipeline joins its sections with no space between sentences, so a paragraph break is put back
// at each join (whitespace only). It ends each finished story with its stop word (`TERMINATE`), which is removed; a story
// without it (unfinished), a story whose text holds the pipeline agents' messages to one another (requests for an
// outline, "sub-events", "I'm sorry, but"), and a text that appears twice in the file are left out. Neither LongStory nor
// StoryWriter is among Litmus's training sources.
//
// Both sets are drawn at random with the same fixed seed, each by its own generator so either draw can be repeated alone,
// and kept to the same length range, so that length cannot separate them; no document is cut. Both are put in the same
// plain-text shape: paragraphs on one line each, separated by a blank line. Writes <dir>/text/human/*.txt,
// <dir>/text/ai/*.txt and <dir>/chosen.json (the rule, the seed, the sources with their revisions, and one row per document
// with its hash: the generator for an AI row, the original publication and Gutenberg release for a human row). Keeps the
// downloaded catalog and LongStory file in <dir>, so a stopped run does not fetch them again.

import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc), []));
if (!args.out) throw new Error('give --out <dir>');
const out = resolve(args.out);
const perSet = Number(args['per-set'] ?? 60);
const minChars = Number(args['min-chars'] ?? 12000);
const maxChars = Number(args['max-chars'] ?? 40000);
const agent = args['user-agent'] ?? 'litmus-demo-fetch (research; contact: the repository owner)';
const SEED = 20261118;
const MIRROR = 'https://gutenberg.pglaf.org';
// Ebooks after the training snapshot's last (66142), up to the last one in the catalog read on 2026-11-19, issued by then.
const GUTENBERG = { after: 66142, through: 79956, issued_through: '2026-11-08', training_source: 'manu/project_gutenberg@164853d214065df26a630ee1ab91a0c39e461caf (English split ends at ebook 66142)' };
const LONGSTORY = {
  dataset: 'THU-KEG/LongStory',
  file: 'sft_premise.jsonl',
  revision: 'a722ee18129131fcb9649c96983b258a431801e3',
  sha256: 'c25cdb52bfbbcb9c3a8083ef7ef9cb8176eab550dc77943cfe9b59786081c52e',
  rows: 5279,
  generator: 'GPT-4o-mini',
};

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const get = async (url, wait) => {
  for (let attempt = 0; ; attempt++) {
    const res = await fetch(url, { headers: { 'user-agent': agent } });
    await sleep(wait);
    if ((res.status === 429 || res.status >= 500) && attempt < 8) {
      await sleep(5000 * 2 ** Math.min(attempt, 4)); // asked to slow down, or a passing server error: wait, doubling each time
      continue;
    }
    return res;
  }
};
const getOk = async (url, wait) => {
  const res = await get(url, wait);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res;
};
const sha = (text) => createHash('sha256').update(text).digest('hex');

/** A small seeded generator, so the same draw is made every time; each set has its own, from the same seed. */
const seeded = (seed) => {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
};
const shuffle = (list, rand) => {
  const a = [...list];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
};

/** One paragraph to a line, a blank line between paragraphs: the shape both sets are read in. */
const paragraphs = (text) =>
  text
    .replace(/\r/g, '')
    .split(/\n[ \t]*\n/)
    .map((p) => p.replace(/\s*\n\s*/g, ' ').replace(/[ \t]+/g, ' ').trim())
    .filter((p) => p.length > 0)
    .join('\n\n');

/** RFC 4180 rows: quoted fields may hold commas, doubled quotes and line breaks. */
const parseCsv = (text) => {
  const rows = [];
  let row = [];
  let field = '';
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') {
        field += '"';
        i++;
      } else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ',') {
      row.push(field);
      field = '';
    } else if (c === '\n') {
      row.push(field.replace(/\r$/, ''));
      rows.push(row);
      row = [];
      field = '';
    } else field += c;
  }
  if (field.length > 0 || row.length > 0) rows.push([...row, field]);
  return rows;
};

mkdirSync(join(out, 'text', 'human'), { recursive: true });
mkdirSync(join(out, 'text', 'ai'), { recursive: true });

// ---- written by people: Project Gutenberg short stories posted after the training snapshot ----

const catalogPath = join(out, 'pg_catalog.csv');
if (!existsSync(catalogPath)) writeFileSync(catalogPath, await (await getOk(`${MIRROR}/cache/epub/feeds/pg_catalog.csv`, 2000)).text());
const catalogText = readFileSync(catalogPath, 'utf8');
const [header, ...records] = parseCsv(catalogText);
const col = Object.fromEntries(header.map((h, i) => [h, i]));
for (const name of ['Text#', 'Type', 'Issued', 'Title', 'Language', 'Authors', 'Subjects']) if (!(name in col)) throw new Error(`the catalog has no ${name} column`);
const pool = records
  .filter((r) => r.length === header.length)
  .map((r) => ({ id: Number(r[col['Text#']]), type: r[col.Type], issued: r[col.Issued], title: r[col.Title].replace(/\s+/g, ' '), language: r[col.Language], authors: r[col.Authors], subjects: r[col.Subjects] }))
  .filter((r) => r.id > GUTENBERG.after && r.id <= GUTENBERG.through && r.issued <= GUTENBERG.issued_through)
  .filter((r) => r.type === 'Text' && r.language === 'en' && /(^|; )Short stories($|;)/.test(r.subjects))
  .filter((r) => !/\b(stories|tales|and other)\b/i.test(r.title)) // a collection is several stories, not one
  .sort((a, b) => a.id - b.id);
if (pool.length < 500) throw new Error(`only ${pool.length} candidate ebooks: the catalog is not the one this rule was written for`);

const START = /\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^*]*\*\*\*/;
const END = /\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^*]*\*\*\*/;
const NOTE = /\n[ \t]*\[?[ \t]*(?:Transcriber(?:'|’)?s?[ \t]+Notes?|Notes? on the Text)\b/i;
const ADDED = /\[(?:Transcriber(?:'|’)?s?[ \t]+Notes?|Notes? on the Text|Illustrations?)\b[^\]]*\]/gi;
const LEFT = /transcriber|project gutenberg|etext|notes? on the text|public domain in the United States|copyright status/i;
const RULE = /^[-=_*~ ]+$/; // a line of rule characters, as the transcribers set before the end marker

/** The story as printed, without Gutenberg's license and what its volunteers added; null when that cannot be done. */
const storyOf = (raw) => {
  const text = raw.replace(/\r/g, '');
  const start = text.search(START);
  const end = text.search(END);
  if (start < 0 || end < start) return null;
  let body = text.slice(start, end).replace(START, '').replace(ADDED, '');
  const note = body.search(NOTE); // a closing section of notes, after the story
  if (note >= 0 && note > body.length * 0.8) body = body.slice(0, note);
  const parts = paragraphs(body).replace(/_([^_\n]+)_/g, '$1').split('\n\n');
  while (parts.length > 0 && RULE.test(parts[0])) parts.shift();
  while (parts.length > 0 && RULE.test(parts[parts.length - 1])) parts.pop();
  const story = parts.join('\n\n');
  return LEFT.test(story) ? null : story;
};
const field = (raw, name) => (raw.match(new RegExp(`^${name}:[ \\t]*(.+)$`, 'm')) ?? [])[1]?.trim() ?? null;

const chosen = {
  seed: SEED,
  rule: { per_set: perSet, min_chars: minChars, max_chars: maxChars },
  human: {
    source: `Project Gutenberg single short stories (subject "Short stories", English), ebooks ${GUTENBERG.after + 1} to ${GUTENBERG.through} issued by ${GUTENBERG.issued_through}, from the mirror ${MIRROR}`,
    revision: { catalog_sha256: sha(catalogText), candidates: pool.length, excluded_training_source: GUTENBERG.training_source, texts: 'each row\'s sha256 and the mirror\'s last-modified date' },
    license: 'public domain in the United States',
    documents: [],
  },
  ai: {
    source: `${LONGSTORY.dataset} ${LONGSTORY.file} (StoryWriter, arXiv 2506.16445), stories by ${LONGSTORY.generator}`,
    revision: { commit: LONGSTORY.revision, file_sha256: LONGSTORY.sha256 },
    license: 'MIT (stated in the StoryWriter paper)',
    documents: [],
  },
};

// The first half is saved when it is done, so a stopped run does not repeat it.
const humanDone = join(out, 'human.json');
if (existsSync(humanDone)) {
  chosen.human.documents = JSON.parse(readFileSync(humanDone, 'utf8'));
  console.log(`human: ${chosen.human.documents.length} documents from an earlier run`);
} else {
  console.log(`human: ${pool.length} candidate ebooks`);
  const skipped = {};
  for (const book of shuffle(pool, seeded(SEED))) {
    if (chosen.human.documents.length >= perSet) break;
    const url = `${MIRROR}/cache/epub/${book.id}/pg${book.id}.txt`;
    const res = await get(url, 2000);
    if (!res.ok) {
      skipped[`HTTP ${res.status}`] = (skipped[`HTTP ${res.status}`] ?? 0) + 1;
      continue;
    }
    const raw = await res.text();
    const original = field(raw, 'Original publication');
    const year = Math.max(...(original?.match(/\b1[5-9]\d\d\b|\b20[01]\d\b/g) ?? []).map(Number), -Infinity);
    const story = storyOf(raw);
    const reason = !original || !(year > 0 && year < 2020) ? 'no original publication before 2020' : story === null ? 'notes not separable' : story.length < minChars || story.length > maxChars ? 'outside the length range' : null;
    if (reason) {
      skipped[reason] = (skipped[reason] ?? 0) + 1;
      continue;
    }
    const name = `h_${book.id}`;
    writeFileSync(join(out, 'text', 'human', `${name}.txt`), `${story}\n`);
    chosen.human.documents.push({
      file: `${name}.txt`,
      chars: story.length,
      sha256: sha(story),
      ebook: book.id,
      title: book.title,
      authors: book.authors,
      original_publication: original,
      year,
      gutenberg_release: field(raw, 'Release date'),
      last_modified: res.headers.get('last-modified'),
    });
    console.log(`human ${chosen.human.documents.length} #${book.id} ${year} ${story.length} chars`);
  }
  console.log('human skipped:', JSON.stringify(skipped));
  writeFileSync(humanDone, JSON.stringify(chosen.human.documents));
}

// ---- written by AI: LongStory, GPT-4o-mini through StoryWriter ----

const longPath = join(out, `longstory_${LONGSTORY.file}`);
if (!existsSync(longPath)) {
  const url = `https://huggingface.co/datasets/${LONGSTORY.dataset}/resolve/${LONGSTORY.revision}/${LONGSTORY.file}`;
  writeFileSync(longPath, Buffer.from(await (await getOk(url, 1000)).arrayBuffer()));
}
const longBytes = readFileSync(longPath);
if (sha(longBytes) !== LONGSTORY.sha256) throw new Error(`${longPath} is not the pinned file (SHA-256 differs): delete it and run again`);
const lines = longBytes.toString('utf8').split('\n').filter((l) => l.trim().length > 0);
if (lines.length !== LONGSTORY.rows) throw new Error(`${LONGSTORY.file} has ${lines.length} rows, not ${LONGSTORY.rows}`);
const texts = lines.map((l) => {
  const messages = JSON.parse(l).messages;
  const last = messages[messages.length - 1];
  if (last.role !== 'assistant') throw new Error('a LongStory row does not end with the story');
  return last.content;
});
const copies = new Map();
for (const t of texts) copies.set(t, (copies.get(t) ?? 0) + 1);
const STOP = /\s*(?:\*\*|```)?\s*TERMINATE\.?\s*(?:\*\*|```)?\s*$/;
// The pipeline joins the sections it writes with no space ("...this disorienting world.As they progressed..."): a
// paragraph break is put back at each such join, which changes whitespace only.
const JOIN = /([a-z][.!?]["”’]?)([A-Z][a-z])/g;
const AGENTS = /sub-events?\b|\bI(?:'|’)m sorry, but\b|\b(?:provide|share) (?:the|a|your) (?:events|outline|storyline|specific story|next sub)|\b(?:break|organize) (?:it|them|this) down into|^#+\s*(?:Sub-)?Event\b|[.!?](?:Sure|Certainly|Got it|Absolutely)!/im;
const skippedAi = {};
for (const row of shuffle(texts.map((_, i) => i), seeded(SEED))) {
  if (chosen.ai.documents.length >= perSet) break;
  const raw = texts[row];
  const story = paragraphs(raw.replace(STOP, '').replace(JOIN, '$1\n\n$2'));
  const reason = copies.get(raw) > 1 ? 'appears twice' : !STOP.test(raw) ? 'unfinished (no stop word)' : AGENTS.test(story) ? 'agent messages in the text' : story.length < minChars || story.length > maxChars ? 'outside the length range' : null;
  if (reason) {
    skippedAi[reason] = (skippedAi[reason] ?? 0) + 1;
    continue;
  }
  const name = `a_${String(row).padStart(4, '0')}`;
  writeFileSync(join(out, 'text', 'ai', `${name}.txt`), `${story}\n`);
  chosen.ai.documents.push({ file: `${name}.txt`, chars: story.length, sha256: sha(story), model: LONGSTORY.generator, row });
  console.log(`ai ${chosen.ai.documents.length} row ${row} ${story.length} chars`);
}
console.log('ai skipped:', JSON.stringify(skippedAi));
if (chosen.human.documents.length < perSet || chosen.ai.documents.length < perSet) throw new Error(`found ${chosen.human.documents.length} human and ${chosen.ai.documents.length} AI documents of ${perSet} wanted`);
const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
};
writeFileSync(join(out, 'chosen.json'), JSON.stringify(chosen, null, 1));
console.log(`done: ${chosen.human.documents.length} human (median ${median(chosen.human.documents.map((d) => d.chars))} chars), ${chosen.ai.documents.length} AI (median ${median(chosen.ai.documents.map((d) => d.chars))} chars)`);
