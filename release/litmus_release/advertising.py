"""Stage `ad`: the marketing video, rendered again from this release's programs by `demo/make.mjs`.

The stage runs the demo's own command, as `demo/README.md` documents it, on the release's Windows build and a folder of
desktop footage captured from the Linux app (`demo/capture/run_linux.sh`, which needs a machine the person names). It
is optional: nothing else needs its videos, and `assemble` does not attach them.
"""

from __future__ import annotations

from pathlib import Path

from .commands import StageError, run
from .stages import Context, Stage


def ad(context: Context) -> None:
    clips = context.options.ad_clips
    if clips is None or not clips.is_dir():
        raise StageError("the ad needs footage of the Linux app: capture it with `demo/capture/run_linux.sh` (demo/README.md) and pass the folder as --ad-clips "
                         "(it holds app.mp4, app.json, quick.mp4, quick.json, and memory.json)")
    build = context.work / "build" / "windows"
    run(["node", "make.mjs", "--work", str(context.work / "ad"), "--litmus", str(build / "litmus.exe"), "--site", str(build / "site"),
         "--manifest", str(build / "manifest.json"), "--clips", str(clips)], context.repository / "demo")


def outputs(context: Context) -> list[Path]:
    return [context.work / "ad" / "out"]


STAGES = (Stage("ad", ("verify",), "re-render the ad from this release", ad, outputs, platform="windows", required=False),)
