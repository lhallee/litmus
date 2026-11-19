Litmus is a free detector of AI written text that says how sure it is.

This is version <!--litmus:release.version-->1.0.0<!--/litmus-->. Its model is <!--litmus:t:released_size-->`s192_l4` (width 192, 4 layers, 1.79M ternary weights)<!--/litmus-->. For this model,
<!--litmus:t:accuracy_summary-->on held-out test texts it wrongly called 5.9% of human texts not human at the 5% level, and it caught 62.1% of AI texts when 5% of human texts are wrongly flagged (35.1% at 1%)<!--/litmus-->. Measured <!--litmus:t:speed_sentence-->on one processor core (AMD Ryzen 7 3800X 8-Core Processor), reading a 28,920-token document, the app read about 28,267 tokens (word pieces) a second and the browser page about 5,540 (Chromium 156, 1 worker), so the document took about 1.0 seconds in the app and about 5.2 in the page<!--/litmus-->.

A reading is evidence about where a text came from, never proof of who wrote it. It is built for English, and it is
weaker on short texts and on code.

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

Everything is described in the <!--litmus:t:latest_release_link-->[the latest release](https://github.com/lhallee/litmus/releases/latest)<!--/litmus--> and its [README](https://github.com/<!--litmus:target.github.repo-->lhallee/litmus<!--/litmus-->#readme),
and the model's page is <!--litmus:t:hub_model_link-->https://huggingface.co/lhallee/litmus<!--/litmus-->.
