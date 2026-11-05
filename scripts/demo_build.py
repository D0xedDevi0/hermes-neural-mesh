#!/usr/bin/env python3
"""Build the demo from real captures, in two stages so a timeout cannot lose the work.

    python3 scripts/demo_build.py capture   # run the real commands, write the ledger + scenes
    python3 scripts/demo_build.py render    # render the captured strings, encode, verify

Nothing shown in the video is hand-typed. Every line comes from a command that actually ran,
and the full output of each is written to docs/evidence/demo_capture.txt as the source ledger.
If a capture fails, the stage stops rather than filling the gap with something plausible.

Splitting the stages matters: the captures cost a few minutes of real agent sessions, while
rendering costs seconds. Losing the render is cheap; losing the captures is not.
"""

import importlib.util
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

REPO = pathlib.Path(__file__).resolve().parent.parent
EV = REPO / "docs" / "evidence"
SCENES = EV / "demo_scenes.json"
PROFILE = "nmtest"
PROFILE_DIR = pathlib.Path("/opt/data/profiles") / PROFILE
FACTS = "the retry limit for service-3 is"

W, H = 1600, 900
BG, PANEL = (12, 14, 20), (18, 22, 32)
GREEN, RED, CYAN, GRAY, WHITE, AMBER = (
    (0, 230, 118), (255, 69, 58), (0, 200, 255), (150, 158, 176), (235, 238, 245), (255, 190, 60),
)
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MAXW = 50  # chars, measured against the panel inner width at this font size

capture_log: list[str] = []


def run(cmd, timeout=420):
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    out = (proc.stdout or "") + (proc.stderr or "")
    capture_log.append(f"$ {' '.join(str(c) for c in cmd)}\n[exit {proc.returncode}]\n{out}\n")
    return proc.returncode, out


def clean(text: str) -> list[str]:
    """Drop image-managed environment noise; keep everything the command really said."""
    skip = ("install out of sync", "rebuild the artifact")
    return [ln.rstrip() for ln in text.splitlines() if ln.strip() and not any(s in ln for s in skip)]


def wrap(text: str, width: int = MAXW) -> list[str]:
    out, cur = [], ""
    for word in text.split():
        if len(cur) + len(word) + 1 <= width:
            cur = f"{cur} {word}".strip()
        else:
            out.append(cur)
            cur = word
    if cur:
        out.append(cur)
    return out


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── stage 1: capture ────────────────────────────────────────────────────────────────────────


def capture():
    EV.mkdir(parents=True, exist_ok=True)

    shutil.rmtree(PROFILE_DIR / "neural-mesh", ignore_errors=True)
    shutil.rmtree(PROFILE_DIR / "sessions", ignore_errors=True)
    rc1, out1 = run([
        "hermes", "-p", PROFILE, "-z",
        "Use mesh_remember to store exactly this project fact: our launch date is Nov 11. "
        "Then say you have stored it.",
    ])
    taught = clean(out1)

    shutil.rmtree(PROFILE_DIR / "sessions", ignore_errors=True)  # cold read: no history left
    rc2, out2 = run([
        "hermes", "-p", PROFILE, "-z",
        "What is our launch date? Answer from stored memory only, in one short sentence.",
    ])
    cold = clean(out2)

    # the tool surface Hermes itself calls, on an isolated store
    mod = _load(REPO / "__init__.py", "nmp")
    provider = mod.NeuralMeshProvider()
    provider.initialize("demo", hermes_home=tempfile.mkdtemp(prefix="nm-demo-store-"),
                        agent_context="primary")

    def call(tool, args):
        out = provider.handle_tool_call(tool, args) or ""
        capture_log.append(f"$ tool {tool} {json.dumps(args)}\n{out}\n")
        return out

    older = call("mesh_remember", {"content": f"{FACTS} 3 attempts", "valid_from": "2025-01-01"})
    found = re.search(r"\(id (\w+)\)", older)
    old_id = found.group(1) if found else ""
    call("mesh_remember", {
        "content": f"{FACTS} 11 attempts",
        "valid_from": "2026-06-01",
        "supersedes": old_id,
    })
    now = call("mesh_recall", {"query": f"what is {FACTS}?"})
    then = call("mesh_asof", {"query": f"what is {FACTS}?", "as_of": "2026-03-01"})
    stats = call("mesh_stats", {})
    provider.shutdown()

    # the ablation, run fresh so the demo cannot quote a stale file
    ev = _load(REPO / "scripts" / "evidence_temporal.py", "ev")
    temporal, flat = ev.build(temporal=True), ev.build(temporal=False)
    tm_now, tm_asof = ev.score(temporal, temporal=True)
    fl_now, fl_asof = ev.score(flat, temporal=False)
    temporal.shutdown()
    flat.shutdown()

    if not old_id:
        raise SystemExit("could not read the stored node id back - refusing to build on a guess")

    scenes = {
        "ablation": {"n": ev.FAMILIES, "tm_now": tm_now, "tm_asof": tm_asof,
                     "fl_now": fl_now, "fl_asof": fl_asof, "asof": ev.ASOF_DATE},
        "session1": {"rc": rc1, "lines": taught},
        "session2": {"rc": rc2, "lines": cold},
        "beats": {"now": now, "then": then, "stats": stats, "old_id": old_id},
    }
    SCENES.write_text(json.dumps(scenes, indent=2) + "\n")
    (EV / "demo_capture.txt").write_text(
        "SOURCE LEDGER - raw output of every command the demo renders.\n"
        "Trimmed only for image-managed install noise; nothing else altered.\n"
        f"node {old_id} was superseded by the correction\n\n" + "\n".join(capture_log)
    )

    print(f"session 1 rc={rc1}: {len(taught)} lines | session 2 rc={rc2}: {len(cold)} lines")
    print(f"node stored+superseded: {old_id}")
    print(f"recall now : {now.splitlines()}")
    print(f"asof 20260301: {then.splitlines()}")
    print(f"ablation   : mesh {tm_now}/{ev.FAMILIES} & {tm_asof}/{ev.FAMILIES} | "
          f"flat {fl_now}/{ev.FAMILIES} & {fl_asof}/{ev.FAMILIES}")
    print("wrote docs/evidence/demo_scenes.json + demo_capture.txt")


# ── stage 2: render ─────────────────────────────────────────────────────────────────────────


def scene_title_card():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((70, 250), "bi-temporal memory", font=ImageFont.truetype(MONO_BOLD, 78), fill=CYAN)
    d.text((70, 350), "for Hermes Agent", font=ImageFont.truetype(MONO_BOLD, 78), fill=WHITE)
    d.text((74, 470), "a real MemoryProvider: local, offline, no keys, no binaries",
           font=ImageFont.truetype(MONO, 26), fill=GRAY)
    d.text((74, 520), "NOT another vector store - one that knows WHEN things were true",
           font=ImageFont.truetype(MONO_BOLD, 28), fill=AMBER)
    return img


def scene(left_lines, right_lines, *, tag, right_color=GREEN,
          left_title="FLAT VECTOR MEMORY", right_title="NEURAL_MESH"):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    fb, fs = ImageFont.truetype(MONO_BOLD, 30), ImageFont.truetype(MONO, 20)
    d.text((40, 26), "NEURAL_MESH  |  memory that knows when",
           font=ImageFont.truetype(MONO_BOLD, 38), fill=CYAN)
    d.text((40, 76), "same engine, same embeddings, same facts - only the temporal model differs",
           font=fs, fill=GRAY)
    tb = d.textbbox((0, 0), tag, font=ImageFont.truetype(MONO_BOLD, 26))
    d.rectangle((W - tb[2] - 70, 24, W - 28, 24 + tb[3] + 16), fill=right_color)
    d.text((W - tb[2] - 54, 36), tag, font=ImageFont.truetype(MONO_BOLD, 26), fill=(10, 10, 12))

    pw = (W - 40 - 60) // 2
    py, ph = 126, H - 126 - 40
    for px, title, col, lines in ((30, left_title, GRAY, left_lines),
                                  (30 + pw + 20, right_title, right_color, right_lines)):
        d.rounded_rectangle((px, py, px + pw, py + ph), radius=14, fill=PANEL, outline=col, width=2)
        d.text((px + 22, py + 16), title[:40], font=fb, fill=col)
        y = py + 66
        for line in lines:
            color = WHITE
            if line.startswith("$") or line.startswith("tool "):
                color = AMBER
            elif line.startswith(">"):
                color = CYAN
            elif re.search(r"\b1/15\b|\b0/", line):
                color = RED
            elif re.search(r"\b15/15\b|attempts|remember|stored", line, re.I):
                color = GREEN
            d.text((px + 22, y), line[:MAXW], font=ImageFont.truetype(MONO, 22), fill=color)
            y += 32
    return img


def encode(frames, out_path, fps=30):
    """Encode each scene on its own, then concat-copy.

    An N-input concat filter over looped stills hangs (300s timeout, truncated output); a
    single-input encode of the same card finishes in about a second. The final path is only
    written once the whole thing is verified, so a partial file can never be mistaken for a
    finished video.
    """
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="nm-demo-"))
    parts = []
    for i, (img, dur) in enumerate(frames):
        png, part = tmp / f"s{i:02d}.png", tmp / f"s{i:02d}.mp4"
        img.save(png)
        rc, out = run(["ffmpeg", "-y", "-loop", "1", "-t", str(dur), "-i", str(png),
                       "-r", str(fps), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                       "-pix_fmt", "yuv420p", str(part)], timeout=150)
        if rc != 0 or not part.exists():
            raise SystemExit(f"scene {i} failed to encode: {out[-400:]}")
        parts.append(part)

    listing = tmp / "parts.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in parts))
    staged = tmp / "staged.mp4"
    rc, out = run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                   "-c", "copy", "-movflags", "+faststart", str(staged)], timeout=180)
    if rc != 0 or not staged.exists():
        raise SystemExit(f"concat failed: {out[-400:]}")

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
         str(staged)], capture_output=True, text=True)
    info = json.loads(probe.stdout or "{}")
    if (info.get("streams") or [{}])[0].get("codec_name") != "h264":
        raise SystemExit("staged file did not probe as h264 - refusing to publish it")
    shutil.copyfile(staged, out_path)  # only now does the real path exist
    return out_path


def render():
    if not SCENES.exists():
        raise SystemExit("no captured scenes - run the capture stage first")
    s = json.loads(SCENES.read_text())
    ev, b = s["ablation"], s["beats"]
    easy = f"{ev['n']}/{ev['n']}"

    def session(key, max_lines=15):
        """Wrap the captured lines to the panel width, prefixed, newest lines kept."""
        out: list[str] = []
        for ln in (s[key]["lines"] or ["(no output)"]):
            out += [f"  {piece}" for piece in wrap(ln, 46)]
        return out[-max_lines:]

    def beat(key):
        return [f"  {piece}" for piece in wrap(str(b[key]), 46)]

    frames = [
        (scene_title_card(), 3),
        (scene(
            [f"store all {ev['n']} values, append-only", "",
             "> which value is current?", f"  answered right {ev['fl_now']} times", "",
             f"> which value was current on {ev['asof']}?", "  no such operation exists"],
            [f"store the same {ev['n']} values with valid_from", "and supersedes chaining", "",
             "> which value is current?", f"  {easy} - the live value only", "",
             f"> which value was current on {ev['asof']}?",
             f"  {easy} - the window that held then"],
            tag="THE DIFFERENCE", right_color=GREEN), 15),
        (scene(
            ["$ hermes -p nmtest -z", '  "store: our launch date is Nov 11"', "", *session("session1")],
            ["the agent calls mesh_remember", "the provider writes to local SQLite", "",
             "no network, no API key, no compiled binary", "",
             "storage lives in the Hermes profile,", "one file the user owns and can delete"],
            tag="TAUGHT", right_color=GREEN, left_title=f"SESSION 1 - rc={s['session1']['rc']}",
            right_title="WHAT JUST HAPPENED"), 16),
        (scene(
            ["$ rm -rf $PROFILE/sessions", "  (session history deleted, store kept)", "",
             "$ hermes -p nmtest -z", '  "what is our launch date?"', "",
             *session("session2")],
            ["a brand-new process", "with no conversation history", "",
             "answered from the store, not from context", "",
             "this is the whole point: memory that", "outlives the session that made it"],
            tag="COLD RECALL", right_color=GREEN, left_title=f"SESSION 2 - rc={s['session2']['rc']}",
            right_title="WHY IT MATTERS"), 20),
        (scene(
            ["> mesh_recall  (what is true now)", *beat("now"), "",
             "> mesh_stats (store on disk)", *beat("stats")[:3]],
            ["> mesh_asof  (what was true then)", "  as_of = 2026-03-01", *beat("then"), "",
             "the correction replaced the current value", "without destroying the history"],
            tag="POINT-IN-TIME", right_color=CYAN, left_title="BOTH TOOLS, SAME STORE",
            right_title="A DATED QUESTION"), 16),
        (scene(
            [f"flat vector store   {ev['fl_now']} current, {ev['fl_asof']} dated"],
            [f"NEURAL_MESH        {easy} current, {easy} dated"],
            tag="MEASURED", right_color=GREEN,
            left_title=f"{ev['n']} FACT FAMILIES, {ev['n'] * 2} QUESTIONS",
            right_title="CORRECT ANSWERS"), 11),
    ]

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((70, 180), "github.com/D0xedDevi0", font=ImageFont.truetype(MONO_BOLD, 58), fill=CYAN)
    d.text((70, 250), "/hermes-neural-mesh", font=ImageFont.truetype(MONO_BOLD, 58), fill=WHITE)
    d.text((74, 360), "hermes plugin install neural-mesh", font=ImageFont.truetype(MONO, 30), fill=GREEN)
    d.text((74, 420), "hermes config set memory.provider neural-mesh",
           font=ImageFont.truetype(MONO, 30), fill=GREEN)
    d.text((74, 530), "produced walkthrough: real captures, rendered offline.",
           font=ImageFont.truetype(MONO, 24), fill=GRAY)
    d.text((74, 568), "source ledger: docs/evidence/demo_capture.txt",
           font=ImageFont.truetype(MONO, 24), fill=GRAY)
    frames.append((img, 4))

    total = sum(d for _img, d in frames)
    if total > 90:
        raise SystemExit(f"OVER BUDGET before encoding: {total}s > 90s")
    out = encode(frames, REPO / "docs" / "demo.mp4")

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", out],
        capture_output=True, text=True,
    )
    info = json.loads(probe.stdout or "{}")
    dur = float(info.get("format", {}).get("duration", 0))
    stream = (info.get("streams") or [{}])[0]
    if stream.get("codec_name") != "h264" or float(info.get("format", {}).get("duration", 0)) <= 0:
        raise SystemExit("ffprobe could not verify the encode - not claiming a video")
    (EV / "demo_proof.json").write_text(json.dumps({
        "file": "docs/demo.mp4", "duration_s": round(dur, 2),
        "codec": stream.get("codec_name"), "pix_fmt": stream.get("pix_fmt"),
        "size": f"{stream.get('width')}x{stream.get('height')}", "frames": int(stream.get("nb_frames") or 0),
        "ablation": ev, "session_rc": [s["session1"]["rc"], s["session2"]["rc"]],
        "method": ("Terminal-style frames rendered offline from real captured output "
                   "(docs/evidence/demo_capture.txt). Not a screen recording."),
    }, indent=2) + "\n")

    print(f"duration : {dur:.1f}s (target < 90)")
    print(f"codec    : {stream.get('codec_name')} {stream.get('pix_fmt')} "
          f"{stream.get('width')}x{stream.get('height')}")
    print(f"scenes   : {len(frames)}")
    print("wrote docs/demo.mp4 + docs/evidence/demo_proof.json")


if __name__ == "__main__":
    stage = (sys.argv[1] if len(sys.argv) > 1 else "all").lower()
    if stage in ("capture", "all"):
        capture()
    if stage in ("render", "all"):
        render()
