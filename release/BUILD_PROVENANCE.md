# Verifying a release build

A release's binaries are built by `release/build_release.py` on the maintainer's machine, through the functions
`tools/verify_build.py --model` runs: from a fresh copy of `engine/` and `app/` at one fixed path (`C:\litmus-build`
on Windows), with the release's model files at `C:\litmus-build\model` and built in, the toolchain
`rust-toolchain.toml` pins, `--locked` dependencies, the C runtime linked in, and flags that keep the source's path,
the Cargo registry's path, and the link time out of the output. They are not built on a hosted runner and carry no
signed build attestation. A download is checked in three ways:

1. **Its hash.** The release attaches `SHA256SUMS`, and `release/SHA256SUMS` in this repository holds the same lines.
   `sha256sum -c --ignore-missing SHA256SUMS`, or `Get-FileHash` in PowerShell, compares a downloaded file with them.
2. **A rebuild.** With this repository at the release's commit, the release's three model files in a folder, and the
   release's `SHA256SUMS`:

   ```sh
   python tools/verify_build.py --model <model folder> --sums <SHA256SUMS>
   ```

   rebuilds `litmus-app.exe` and `litmus.exe` and prints each one's SHA-256 beside the published one. When both
   agree it ends with `MATCH: litmus.exe and litmus-app.exe, built here from this source with this model, are byte
   for byte the files <SHA256SUMS> lists.` `--twice` builds them a second time from a clean state as well. The
   programs reproduce only when built at the same fixed path as the release, which the script uses.
3. **The model's files.** `python tools/verify_model.py <release folder> --sums release/SHA256SUMS` checks
   `model.safetensors`, `config.json`, and `tokenizer.json` against the hashes the release states.

Reproducibility is verified between builds on one Windows x86-64 machine; a build on another machine can differ
when its MSVC linker or Windows SDK differs. [AUDIT.md](../docs/AUDIT.md#rebuild-and-compare-hashes) and
[DEVELOPING.md](../docs/DEVELOPING.md#verify-the-build) give the measured results and the limits.
