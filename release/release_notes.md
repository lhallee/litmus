Litmus is a free detector of AI written text that says how sure it is.

This is version <!--litmus:release.version-->1.1.0<!--/litmus-->. Its model is <!--litmus:t:released_size-->`s512_l6` (width 512, 6 layers, 18.94M ternary weights)<!--/litmus-->. For this model,
<!--litmus:t:accuracy_summary-->on held-out test texts it wrongly called 7.4% of human texts not human at the 5% level, and it caught 67.2% of AI texts when 5% of human texts are wrongly flagged (53.0% at 1%)<!--/litmus-->. Measured <!--litmus:t:speed_sentence-->on one processor core (AMD Ryzen 7 3800X 8-Core Processor), reading a 28,920-token document, the app read about 4,971 tokens (word pieces) a second and the browser page about 879 (Chromium 156, 1 worker), so the document took about 5.8 seconds in the app and about 32.9 in the page<!--/litmus-->.

A reading is evidence about where a text came from, never proof of who wrote it. It is built for English, and it is
weaker on short texts and on code.

## What changed in 1.1

- **A larger model.** Width 512 and 6 layers in place of 1.0.0's width 192 and 4, trained with masked language
  modeling beside the pair loss. It reads about a sixth as many tokens a second as 1.0.0 on one core (the speed
  above), so a long document takes seconds rather than a second.
- **Surprise.** The masked language head ships, so a reading's details also show how surprising the text is to the
  model (its mean log probability per token, seeing the rest of the text) and where that falls among human, AI, and
  mixed calibration texts. It is read only when you open a reading's details, and it is evidence about the text and
  the model together, not a verdict.
- **Fewer tells.** 1.0.0 read Markdown, typography, spacing, line wrapping, and how a text ends as evidence of who
  wrote it. 1.1 trained on texts presented anew at every draw: Markdown rendered or kept, dashes and quotes in random
  styles, paragraphs joined or wrapped, a few synonyms swapped, and texts cut mid-sentence, at the same rates for
  every origin. Changing a text's formatting now moves its reading far less, though not to nothing: removing a human
  text's Markdown still makes it look less human, as in 1.0.0 (the model card's limitations give the numbers).
- **A reference set no training text shares.** A reading compares a text with <!--litmus:t:bank_counts-->8,192 human, 8,192 AI, and 8,192 mixed<!--/litmus--> texts from
  clusters no other set holds, so the model never trained on the texts it compares against; calibration is fit on
  another such set.
- **Against 1.0.0.** <!--litmus:record.findings.against_previous-->On the 1,983 test passages 1.0.0 never trained on (642 human, 642 AI, 699 mixed), 1.1 reads PR AUC 0.942, ROC AUC 0.890 and balanced MCC 0.586, against 1.0.0's 0.955, 0.917 and 0.635: 1.1 reads the corpus's own held-out text a little less well, and reads text whose formatting changed far more steadily (above).<!--/litmus-->

## Which file

| File | What it is |
| --- | --- |
| `litmus-app.exe` | The Windows app: one file, no install, runs offline |
| `litmus.exe` | The command line for Windows |
| `litmus-app-linux-x86_64`, `litmus-linux-x86_64` | The app and the command line for Linux on x86-64; a download loses the executable bit, so run `chmod +x` on each |
| `litmus-macos-arm64`, `litmus-macos-x86_64` | The command line for macOS, when attached |
| `model.safetensors`, `config.json`, `tokenizer.json` | The model's files, for `litmus --model <folder>` |
| `manifest.json`, `manifest-linux.json` | What each build holds: the model, the source commit, and the toolchain |
| `results.json` | The numbers this release states |

The programs are not code-signed, so Windows shows "unknown publisher" the first time. Each program has the model
built in. The browser page at <!--litmus:target.github.pages_url-->https://lhallee.github.io/litmus/<!--/litmus-->app/ runs the same model and keeps your text on
your computer.

Everything is described in the repository's [README](https://github.com/<!--litmus:target.github.repo-->lhallee/litmus<!--/litmus-->#readme),
and the model's page is <!--litmus:t:hub_model_link-->https://huggingface.co/lhallee/litmus<!--/litmus-->.
