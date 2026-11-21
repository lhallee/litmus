# Storyboard

The ad is 1080 by 1920 at 30 fps and 41.5 seconds, 18 bars of 4 beats at 104 beats a minute; a bar is 2.31 seconds.
Every cut falls on a bar line and every named moment (the cues in `data/ad/timeline.json`) on a beat, and the music is
synthesized on the same grid (`tools/music.mjs`), so the picture and the sound share one clock. The teaser is 11.5 seconds
(5 bars) from the same scenes, and the wide cut is the ad again in a 1920 by 1080 frame (see below). Everything on screen
is real: footage of the app and the page, readings from the engine, numbers from the pinned run. The footage's own
moments (the paste, the press of Check, the moment a reading appears) are laid on the beats by remapping the footage's
time, so the ad follows the app whatever speed a release reads at.

**Pace.** The first cut ran at 120 beats a minute (36 s). It was too fast, so the tempo is 104 (+15.4% in every length).
The scenes' own durations (a word's arrival, a fade, a camera move, the dip through dark at a cut) were written at 120
and are stretched by 120/104 with the grid, so transitions slow by the same 15%, and the dip at each cut and the flash
that lands a shot are also made longer (the dip from 0.1 s to 0.24 s of scene time, 0.28 s real, on each side of a cut).
Footage is the exception: a recording plays at its own speed between the moments the cues pin, never slower, and
a number on screen that depends on time ("shown at ×5") is computed from the real seconds. The bars per shot did not
change, so the cuts stay on the bar line.

Text stays inside x 70 to 1010, and anything that must be read inside y 270 to 1480, clear of the margins the platforms
cover (top about 14%, bottom up to 35%, sides 6%). Captions are burned in, so the ad reads with the sound off.

## The ad

Times are real seconds from the start; the first cut's are in the last column. The "On screen" words are those of
the cut released with 1.1.0. The next render words the verdicts as the app does since: the hook's line reads
"Uncertain (human, mixed, or AI)", the quick check's first reading "AI or mixed" where the released model rules human
out and keeps both, and the speed shot's counts "read as human", "read as AI or mixed", and "uncertain".

| Time | Bars | Shot | What moves | On screen | First cut |
| --- | --- | --- | --- | --- | --- |
| 0:00.0 | 2 (4.6 s) | Hook | The question is slammed in on the first beat, and a strip of litmus paper beneath it takes its color; at 0:01.7 the question moves up and the answer arrives, a mark lands on the strip at the chance of AI and the pale band opens over that chance's range (the engine's reading of a 23-token sentence) | "Is this AI?" then "Litmus tells you how sure it is." and "Uncertain: human, mixed, or AI" | 4 s |
| 0:04.6 | 3 (6.9 s) | App | Footage of the desktop app (the Linux build on a virtual display): the pointer goes to the box, a text is pasted, Check is pressed, the verdict, the strip with its band and the tinted text appear (the reading takes about 0.2 s); three labels name the parts. The verdict lands at the middle of the shot, so each of the two later captions is on screen for about 1.6 s (the first cut left the third caption 0.25 s) | "Paste. Check." then "A verdict, and how sure." then "Tinted where it reads." | 6 s |
| 0:11.5 | 3 (6.9 s) | Quick check | Footage of a text editor: two paragraphs are selected, Ctrl+Alt+L is pressed, the small window by the pointer shows "AI" and stays about 1.4 s; then one short sentence is selected and the window shows "Uncertain" with the band over that sentence's range, wide because its chance is far from either end | "Select text in any app." then "Press Ctrl+Alt+L." then "It says when it can't tell." | 6 s |
| 0:18.5 | 2 (4.6 s) | Browser | The built page at a phone's size: the text is pasted, Check is pressed, the reading appears about 2.3 s into the shot, so the second caption is on screen for about 1.9 s | "Or in your browser." then "Nothing leaves the page." | 4 s |
| 0:23.1 | 5 (11.5 s) | Speed | The plain facts arrive in large type on the first beats; then 120 long documents (60 written by people, 60 written by AI) are read one after another by `litmus` pinned to one core, replayed from the recorded run: a process list with Litmus highlighted shows its share of the whole machine's processing and its memory live, beside one row, "Other tools", for the rest of the machine (the machine minus `litmus`, during the run), with the machine's processing drawn from the trace and the machine's memory as one bar with Litmus's slice at its edge; tiles fill as each document finishes and what each set reads as counts up. On the lock beat the list shows the run's average processing and peak memory | "One CPU core." "66 MB of memory." "23,549 tokens a second." "Written by people", "Written by AI", "read as human", "read as AI", "uncertain or mixed", "unlike reference" (when a document read so), "CPU, run average", "Memory, run peak", "shown at ×5" | 10 s, "×6" |
| 0:34.6 | 3 (6.9 s) | End card | The mark draws itself; the name; the site address (the largest line after the name, on screen to the last frame); the first line; Free, Private, and Fast; the three doors; two small lines | "lhallee.github.io/litmus", "Litmus is a free detector of AI written text that says how sure it is", "Free", "Private", "Fast", "The desktop app", "In your browser", "The contribute site", then on two lines "Open source" and "Free for noncommercial use" | 6 s |

The recorded run (43.2 s) is shown in 8.1 s, so the speed shot says "shown at ×5" (it said ×6 at 7 s); the factor is the
recorded duration over the real seconds the ad spends on it, computed from the data.

**GPU panel.** The speed shot adds a panel under the sets (tall) or under the facts (wide): Litmus first, with "0%" and "none" as measured, then
Fast-DetectGPT, Desklib v1.01, Binoculars, ModernBERT (Drayson), and e5-small-lora, each with the GPU it was measured on (A10, or GH200 for
Binoculars, which does not fit the A10), a bar for the share of time the GPU was busy, and its own GPU memory. Plain words only ("GPU busy and GPU memory, by
detector"); detectors are named one by one, never as a group, so the "Other tools" row (the rest of the machine during the Litmus run) is not confused with
them. Those detectors read only the first 512 tokens (Desklib 768) of each document and Litmus reads every token; the data file's basis says so and the
picture does not claim otherwise. With no `data/ad/gpu_use.json` the panel is not drawn. See README.md, "The GPU panel".

## The teaser

| Time | Shot | Notes |
| --- | --- | --- |
| 0:00.0 | Hook, 2.3 s | The question and the answer, faster |
| 0:02.3 | Quick check, 4.6 s | One selection and the reading, "AI" |
| 0:06.9 | End card, 4.6 s | The mark, the name, "lhallee.github.io/litmus", the first line, Free, Private, and Fast; the address is on screen from 0.75 s into the card to the last frame (3.9 s) |

The first cut's teaser was 8 s (2, 4, 2) and the slowed one 9.2 s (2.3, 4.6, 2.3); its end card got a second bar so the address can hold for more than 2.5 s.

## The wide cut

`wide` is the ad's shots, cues, music, and story in a 1920 by 1080 frame, for a README and a web page (ratio 16:9, 41.5 s,
the ad's length). It is not the vertical film scaled: the scenes read the frame they are given (`Layout` in
`src/stage.ts`) and compose for it, and there is no second set of scenes.

- **Frame and margins.** No platform covers a wide frame, so text keeps an even margin: 96 px at the sides, 60 px above
  and below. The ground's three washes sit at the frame's left, lower left, and lower right.
- **Captions and footage.** Each caption stands at the left, left-aligned, 100 to 120 px, at the frame's middle height; the
  footage card (or the phone) stands at the right, 930 by 940 px, the same crop of the capture as the tall cut, so the
  app, the editor, and the page read at about the same scale as they do on a phone.
- **Hook.** The question is 230 px and centered, with the strip below it; the answer takes the line under the question
  when the question moves up; the verdict line sits under the strip.
- **Browser.** The phone stands upright at the right (screen 440 by 952), the two captions at the left.
- **Speed.** The three facts at the left with the counts under them ("tokens read", the recorded clock and "shown at"),
  the process list and the two sets of tiles (432 px a column) at the right. With the GPU panel it sits under the facts at the
  left (784 px wide, one column of 6 rows) and the counts move under the sets.
- **End card.** The mark, the name, "lhallee.github.io/litmus" (78 px, the largest line after the name), and the first line
  at the left; Free, Private, and Fast (blue, purple, and red, the scale's order, 240 px each), the three doors, and the two
  small lines at the right. In the tall cut the same lines stack in the centered column (the address at 76 px, the doors
  tightened to 100 px apart). The address arrives right after the name and stays to the last frame, which is also the
  wide poster. No cut dips out at the end: the last frame is the whole card, so a player that stops
  there shows it, and a page that loops the video cuts from it to the hook.
- **Sound.** The same music and cues as the ad; every caption is burned in, so it reads muted.

`node render.mjs wide` also writes the web encodes (see README.md, "Embedding").

## Cues

The sound of a cue (`hit`, `tick`, `ding`, `pop`) is made at its beat; the scene anchors its picture to the same beat.
`app_check` and `web_check` are placed where the footage's press of Check lands given the measured delay before the
reading appears (0.19 s in the app, 0.28 s in the page, at 104 beats a minute), so a release that reads at a different
speed moves them by that difference.

## Rules the ad keeps

- No disclaimer or explanation of a figure appears inside the picture; nothing shown needs one, because every reading
  is the engine's and every number is measured and stored in `data/ad/` with its basis.
- No document is named. The speed segment draws anonymous tiles and aggregates, and no reading is presented as proof
  about a person or an organization. The segment says only what is true of any machine (one core, the measured
  megabytes, the measured speed); the machine it was measured on is in the data file.
- Words are claims about readings ("read as AI", "said uncertain"), never about who wrote what. Litmus is "free",
  "private", and "fast". The end card's last two lines read "Open source" and "Free for noncommercial use" on Logan's word
  (2026-11-18); the license is PolyForm Noncommercial, which is source available and not open source in the OSI sense, so
  that wording is his call and is open to review before the final render.
