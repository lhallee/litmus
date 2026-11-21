"""Record the Linux app on a virtual display, driven the way a person drives it, for the ad's footage.

Run inside a container with Xvfb, xdotool, xclip, ffmpeg and mousepad (`capture/Dockerfile`):

    python3 linux_capture.py <scene> <app> <text folder> <out folder>

The scenes are `app` (paste a text, press Check, read the verdict, the strip and the tinted text), `quick` (select
text in another program and press Ctrl+Alt+L) and `memory` (no recording: the app's memory and processor time, sampled
from /proc, while it holds the model and checks a long document). A scene writes `<scene>.mp4` (lossless-looking H.264 of the whole display,
the pointer included, at 2560x1600 with the app drawn at 2x) and `<scene>.json` (the scene's clock and what happened
when). Nothing here reaches the network or touches a real desktop: it drives its own X server.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

from dataclasses import dataclass, field
from pathlib import Path


WIDTH = 2560
HEIGHT = 1600
SCALE = 2
DISPLAY = ":99"
FPS = 30
# The app's window for the vertical ad: tall and narrow (physical pixels, the app draws at 2x), so its text is large in the frame.
APP_WINDOW = (1300, 1500)
APP_BOX = (650, 760)
APP_CHECK = (200, 1354)


@dataclass
class Session:
    """One virtual display, one recording, and a log of what the scene did and when (seconds on the recording's clock)."""

    out: Path
    scene: str
    env: dict[str, str]
    started: float = 0.0
    events: list[dict[str, object]] = field(default_factory=list)
    processes: list[subprocess.Popen[bytes]] = field(default_factory=list)
    recorder: subprocess.Popen[bytes] | None = None

    def now(self) -> float:
        return round(time.monotonic() - self.started, 3)

    def note(self, what: str, **extra: object) -> None:
        self.events.append({"t": self.now(), "what": what} | extra)

    def spawn(self, argv: list[str], extra_env: dict[str, str] | None = None) -> subprocess.Popen[bytes]:
        process = subprocess.Popen(argv, env=self.env | (extra_env or {}), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.processes.append(process)
        return process

    def key(self, *keys: str) -> None:
        subprocess.run(["xdotool", "key", "--delay", "60", *keys], env=self.env, check=True, timeout=30)

    def pointer(self) -> tuple[int, int]:
        text = subprocess.run(["xdotool", "getmouselocation", "--shell"], env=self.env, capture_output=True, text=True, check=True, timeout=30).stdout
        values = dict(line.split("=") for line in text.split())
        return int(values["X"]), int(values["Y"])

    def glide(self, x: int, y: int, seconds: float = 0.7) -> None:
        """Move the pointer to (x, y) along an eased path, so the recording shows a hand's motion and hovers register."""
        x0, y0 = self.pointer()
        steps = max(2, int(seconds * 90))
        chain: list[str] = []
        for i in range(1, steps + 1):
            k = i / steps
            e = k * k * (3 - 2 * k)
            chain += ["mousemove", str(round(x0 + (x - x0) * e)), str(round(y0 + (y - y0) * e)), "sleep", f"{seconds / steps:.4f}"]
        self.note("glide", x0=x0, y0=y0, x1=x, y1=y, seconds=seconds)
        subprocess.run(["xdotool", *chain], env=self.env, check=True, timeout=60)

    def click(self, x: int, y: int, what: str, seconds: float = 0.7) -> None:
        self.glide(x, y, seconds)
        time.sleep(0.15)
        self.note(what, x=x, y=y)
        subprocess.run(["xdotool", "click", "1"], env=self.env, check=True, timeout=30)

    def drag(self, x0: int, y0: int, x1: int, y1: int, what: str, seconds: float = 0.9) -> None:
        self.glide(x0, y0, 0.6)
        subprocess.run(["xdotool", "mousedown", "1"], env=self.env, check=True, timeout=30)
        self.note(what, x=x0, y=y0)
        self.glide(x1, y1, seconds)
        subprocess.run(["xdotool", "mouseup", "1"], env=self.env, check=True, timeout=30)

    def shot(self, name: str) -> None:
        subprocess.run(["import", "-window", "root", str(self.out / f"{name}.png")], env=self.env, check=True, timeout=60)

    def rows(self, x: int, y: int, w: int, h: int) -> list[int]:
        """The mean gray (0 to 255) of each display row of the region, top to bottom: where text sits in a column."""
        raw = subprocess.run(["import", "-window", "root", "-crop", f"{w}x{h}+{x}+{y}", "+repage", "-colorspace", "Gray", "-resize", f"1x{h}!", "-depth", "8", "gray:-"],
                             env=self.env, check=True, capture_output=True, timeout=60).stdout
        assert len(raw) == h, f"the screenshot gave {len(raw)} rows for {h}"
        return list(raw)

    def record(self) -> None:
        target = self.out / f"{self.scene}.mp4"
        self.recorder = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "x11grab", "-framerate", str(FPS), "-video_size", f"{WIDTH}x{HEIGHT}", "-draw_mouse", "0",
             "-i", DISPLAY, "-c:v", "libx264", "-preset", "veryfast", "-crf", "12", "-pix_fmt", "yuv420p", "-vsync", "cfr", str(target)],
            env=self.env, stdin=subprocess.PIPE)
        time.sleep(1.0)
        self.started = time.monotonic()
        # The pointer is drawn later from the log (a hand-sized arrow), so the footage has none. A small white window flashes
        # in the bottom-right corner so the recording's clock can be found in the footage and matched to this log.
        self.note("pulse_on")
        flash = subprocess.Popen(["xlogo", "-geometry", f"80x80+{WIDTH - 80}+{HEIGHT - 80}"], env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.5)
        flash.terminate()
        flash.wait(timeout=10)
        self.note("pulse_off")
        time.sleep(0.4)

    def stop(self) -> None:
        if self.recorder is not None and self.recorder.stdin is not None:
            self.recorder.stdin.write(b"q")
            self.recorder.stdin.flush()
            self.recorder.wait(timeout=120)
        for process in reversed(self.processes):
            process.terminate()
        (self.out / f"{self.scene}.json").write_text(json.dumps({"display": [WIDTH, HEIGHT], "scale": SCALE, "fps": FPS, "events": self.events}, indent=1), encoding="utf-8")


def window(session: Session, name: str, timeout: float = 60.0) -> str:
    """The id of the first visible window whose name matches, waiting for it to appear."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = subprocess.run(["xdotool", "search", "--onlyvisible", "--name", name], env=session.env, capture_output=True, text=True, check=False, timeout=30).stdout.split()
        if found:
            return found[0]
        time.sleep(0.3)
    raise TimeoutError(f"no window named {name}")


def place(session: Session, win: str, x: int, y: int, w: int, h: int) -> None:
    subprocess.run(["xdotool", "windowsize", win, str(w), str(h), "windowmove", win, str(x), str(y), "windowfocus", win], env=session.env, check=True, timeout=30)


def clipboard(session: Session, text: str, selection: str = "clipboard") -> None:
    subprocess.run(["xclip", "-selection", selection], input=text.encode("utf-8"), env=session.env, check=True, timeout=30)


def start(scene: str, out: Path, work: Path) -> Session:
    out.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("LITMUS_")} | {
        "DISPLAY": DISPLAY, "LIBGL_ALWAYS_SOFTWARE": "1", "GALLIUM_DRIVER": "llvmpipe", "WINIT_X11_SCALE_FACTOR": str(SCALE),
        "GDK_SCALE": str(SCALE), "XCURSOR_THEME": "DMZ-White", "XCURSOR_SIZE": "32", "HOME": str(work), "NO_AT_BRIDGE": "1",
    }
    session = Session(out=out, scene=scene, env=env)
    session.spawn(["Xvfb", DISPLAY, "-screen", "0", f"{WIDTH}x{HEIGHT}x24", "-nolisten", "tcp"])
    for _ in range(50):
        if subprocess.run(["xdpyinfo"], env=env, capture_output=True, check=False).returncode == 0:
            break
        time.sleep(0.2)
    subprocess.run(["xsetroot", "-solid", "#16141A", "-cursor_name", "left_ptr"], env=env, check=True, timeout=30)
    return session


def launch_app(session: Session, app: Path, work: Path, settings: dict[str, object]) -> str:
    settings_folder = work / "settings_folder"
    shutil.rmtree(settings_folder, ignore_errors=True)
    settings_folder.mkdir(parents=True)
    (settings_folder / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    session.spawn([str(app)], {"LITMUS_DATA_DIR": str(settings_folder), "LITMUS_CACHE_DIR": str(work / "cache")})
    return window(session, "Litmus")


def scene_app(session: Session, app: Path, texts: Path, work: Path, probe: bool) -> None:
    win = launch_app(session, app, work, {"keep_history": False})
    place(session, win, 0, 0, *APP_WINDOW)
    time.sleep(2.0)
    if probe:
        session.shot("probe_app_empty")
    session.record()
    session.note("start")
    time.sleep(1.6)
    essay = (texts / "ai_essay.txt").read_text(encoding="utf-8").strip()
    clipboard(session, essay)
    session.click(APP_BOX[0], APP_BOX[1], "click_box", 1.0)
    time.sleep(0.5)
    session.key("ctrl+v")
    session.note("paste")
    time.sleep(1.8)
    if probe:
        session.shot("probe_app_pasted")
    session.click(APP_CHECK[0], APP_CHECK[1], "click_check", 1.0)
    time.sleep(0.2)
    session.note("checking")
    time.sleep(4.0)
    if probe:
        session.shot("probe_app_result")
    session.glide(APP_WINDOW[0] - 200, 900, 1.2)
    time.sleep(2.5)
    session.note("end")


# The editor's text in display pixels: the column that every line of the document has text in, and where a drag starts
# (inside the first letter's left half, so the selection begins at the line's start) and ends (past each selection's
# last word: the second model paragraph's last line is short, the sentence and the book's last line are longer).
TEXT_COLUMN = (292, 150, 250, 1350)
DRAG_FROM = 293
DRAG_TO = {"ai": 760, "short": 1880, "human": 2060}


def text_lines(rows: list[int]) -> tuple[list[list[int]], int]:
    """The paragraphs of a column of text, each the middle row of its lines, and the distance between lines.

    `rows` is the column's mean gray by row (light text on a dark ground). A line is a run of rows lighter than the
    ground; a paragraph ends where the next line is more than a line and a half away."""
    ground, light = min(rows), max(rows)
    assert light - ground > 12, "the editor's column holds no text"
    level = ground + 0.25 * (light - ground)
    lines: list[tuple[int, int]] = []
    for row, value in enumerate(rows):
        if value <= level:
            continue
        if lines and row - lines[-1][1] <= 4:
            lines[-1] = (lines[-1][0], row)
        else:
            lines.append((row, row))
    middles = [(first + last) // 2 for first, last in lines if last - first >= 6]
    gaps = sorted(b - a for a, b in zip(middles, middles[1:], strict=False))
    pitch = gaps[len(gaps) // 2]
    paragraphs = [[middles[0]]]
    for previous, middle in zip(middles, middles[1:], strict=False):
        if middle - previous > 1.5 * pitch:
            paragraphs.append([])
        paragraphs[-1].append(middle)
    return paragraphs, pitch


def selections(session: Session) -> tuple[dict[str, tuple[int, int, int, int]], dict[str, object]]:
    """Where the pointer drags in the editor, in display pixels, read from the screen: the two model paragraphs
    together, the short sentence, and the book's paragraph (`data/ad/texts/notes.txt`); and the layout found, for
    the log. The rows are found, never typed: the text sat 46 pixels higher on 2026-11-21 than where the drags were
    laid out for on 2026-11-19, in the same image, so a typed row selected the wrong lines."""
    x, y, w, h = TEXT_COLUMN
    paragraphs, pitch = text_lines(session.rows(x, y, w, h))
    assert len(paragraphs) == 4, f"the editor shows {len(paragraphs)} paragraphs, and notes.txt has four"
    first, second, short, book = ([y + middle for middle in paragraph] for paragraph in paragraphs)
    assert len(short) == 1, f"the short sentence wraps to {len(short)} lines"
    layout = {"lines": [len(first), len(second), len(short), len(book)], "first_line": first[0], "pitch": pitch}
    # The book's paragraph ends the document: a drag that ends below its last line takes it to its last word.
    drags = {"ai": (DRAG_FROM, first[0], DRAG_TO["ai"], second[-1]), "short": (DRAG_FROM, short[0], DRAG_TO["short"], short[0]),
             "human": (DRAG_FROM, book[0], DRAG_TO["human"], book[-1] + pitch)}
    return drags, layout


def quick_read(session: Session, what: str, x0: int, y0: int, x1: int, y1: int, hold: float) -> None:
    """Select with the pointer in the editor, press the hotkey, hold on the popup, then close it with Esc."""
    session.drag(x0, y0, x1, y1, f"select_{what}")
    time.sleep(0.9)
    session.note(f"hotkey_{what}")
    session.key("ctrl+alt+l")
    time.sleep(hold)
    session.key("Escape")
    session.note(f"closed_{what}")
    time.sleep(0.8)


def scene_quick(session: Session, app: Path, texts: Path, work: Path, probe: bool) -> None:
    session.spawn(["openbox"])  # a window manager, so the small window takes the focus and Esc closes it as it does on a desktop
    time.sleep(1.0)
    win = launch_app(session, app, work, {"keep_history": False, "quick_check": True})
    subprocess.run(["xdotool", "windowminimize", win], env=session.env, check=True, timeout=30)
    time.sleep(1.0)
    notes = work / "Documents" / "draft.txt"
    notes.parent.mkdir(parents=True, exist_ok=True)
    notes.write_text((texts / "notes.txt").read_text(encoding="utf-8"), encoding="utf-8")
    configure = ("gsettings set org.xfce.mousepad.preferences.view word-wrap true; "
                 "gsettings set org.xfce.mousepad.preferences.view use-default-monospace-font false; "
                 "gsettings set org.xfce.mousepad.preferences.view font-name 'Noto Sans 13'; "
                 f"exec mousepad {notes}")
    session.spawn(["dbus-run-session", "--", "sh", "-c", configure], {"GTK_THEME": "Adwaita:dark"})
    editor = window(session, "draft")
    place(session, editor, 282, 80, 2000, 1440)
    time.sleep(2.0)
    if probe:
        session.shot("probe_quick_editor")
    drags, layout = selections(session)
    session.record()
    session.note("start", **layout)
    time.sleep(1.4)
    if probe:
        session.shot("probe_quick_top")
    for what, (x0, y0, x1, y1) in drags.items():
        quick_read(session, what, x0, y0, x1, y1, 4.0)
    session.note("end")


def process_memory(pid: int) -> dict[str, int]:
    """The memory the operating system reports for a process, in kilobytes: resident now, resident at its peak, and the private part."""
    fields = dict(line.split(":", 1) for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines() if ":" in line)
    stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
    ticks = int(stat[11]) + int(stat[12])
    return {
        "rss_kb": int(fields["VmRSS"].split()[0]),
        "peak_rss_kb": int(fields["VmHWM"].split()[0]),
        "private_rss_kb": int(fields["RssAnon"].split()[0]),
        "cpu_ticks": ticks,
    }


def scene_memory(session: Session, app: Path, texts: Path, work: Path, probe: bool) -> None:
    """The app with the model loaded, then reading a long document: its memory and processor time, sampled from /proc. Nothing is recorded."""
    win = launch_app(session, app, work, {"keep_history": False})
    place(session, win, 0, 0, *APP_WINDOW)
    time.sleep(3.0)
    pid = int(subprocess.run(["pgrep", "-x", app.name], capture_output=True, text=True, check=True, timeout=30).stdout.split()[0])
    session.started = time.monotonic()
    session.note("start", pid=pid, ticks_per_second=os.sysconf("SC_CLK_TCK"))

    def sample(label: str, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            session.note(label, **process_memory(pid))
            time.sleep(0.1)

    sample("idle", 3.0)
    long_text = (texts / "long.txt").read_text(encoding="utf-8").strip()
    clipboard(session, long_text)
    session.click(APP_BOX[0], APP_BOX[1], "click_box", 0.6)
    session.key("ctrl+v")
    session.note("paste", characters=len(long_text))
    sample("pasted", 2.0)
    session.click(APP_CHECK[0], APP_CHECK[1], "click_check", 0.6)
    sample("checking", 12.0)
    session.note("end")


SCENES = {"app": scene_app, "quick": scene_quick, "memory": scene_memory}


def main() -> int:
    scene = sys.argv[1]
    app, texts, out = (Path(a).resolve() for a in sys.argv[2:5])
    probe = len(sys.argv) > 5 and sys.argv[5] == "probe"
    work = Path(os.environ["HOME"]) / "capture" / scene
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    session = start(scene, out, work)
    try:
        SCENES[scene](session, app, texts, work, probe)
    finally:
        session.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
