"""Build the page's scroll-driven film from the caption-free render.

    python film-making/build_scrub_film.py [--crf 25] [--gop 6] [--width 1920] [--end 60]

Input: film-making/app/out/film-nocaptions.mp4, rendered in film-making/app with
    npx remotion render Main out/film-nocaptions.mp4 --props '{"captions":false}' --crf=16
Output, in the repository's film/ folder: the first --end seconds only (the page draws later scenes live from the data), as
film_frag.mp4 (fragmented, a keyframe every few frames so any moment shows quickly while scrolling), init.mp4 + seg*.m4s +
film.json (split by split_film.py into parts under 15 MB each), beats.js (the film's scenes and on-screen text from
app/src/beats.json, which the page shows as call-outs) and start.jpg (the frame the page opens on).
"""
import argparse, json, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE / "app"
# Remotion installs an ffmpeg build with its compositor; otherwise use one on PATH
FF = next((p for p in sorted((APP / "node_modules" / "@remotion").glob("compositor-*/ffmpeg*"))
           if p.stem == "ffmpeg"), shutil.which("ffmpeg"))
START_S = 2.7      # the page opens on the title card, after its fade-in

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=str(APP / "out" / "film-nocaptions.mp4"))
ap.add_argument("--crf", type=int, default=25)
ap.add_argument("--gop", type=int, default=6)
ap.add_argument("--width", type=int, default=1920)
ap.add_argument("--end", type=float, default=60.0, help="seconds of film the page uses (title to the letter zoom; later scenes are live)")
ap.add_argument("--beats-only", action="store_true", help="only rewrite beats.js (call-out text and placement)")
a = ap.parse_args()
out = HERE.parent / "film"
out.mkdir(exist_ok=True)
frag = out / "film_frag.mp4"


def write_beats():
    beats = json.loads((APP / "src" / "beats.json").read_text(encoding="utf-8"))
    beats["start_s"] = START_S
    (out / "beats.js").write_text("window.FILM_BEATS = " + json.dumps(beats, ensure_ascii=False) + ";\n", encoding="utf-8")


if a.beats_only:
    write_beats(); sys.exit()
vf = [] if a.width == 1920 else ["-vf", f"scale={a.width}:-2:flags=lanczos"]
subprocess.run([str(FF), "-v", "error", "-y", "-i", a.src, "-t", str(a.end), *vf, "-an", "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf),
                "-profile:v", "high", "-level", "4.0", "-pix_fmt", "yuv420p", "-g", str(a.gop), "-keyint_min", str(a.gop),
                "-sc_threshold", "0", "-bf", "0", "-movflags", "+frag_keyframe+empty_moov+default_base_moof", str(frag)], check=True)
for old in out.glob("seg*.m4s"):
    old.unlink()
subprocess.run([sys.executable, str(HERE / "split_film.py"), str(frag), str(out)], check=True)
subprocess.run([str(FF), "-v", "error", "-y", "-ss", str(START_S), "-i", a.src, "-frames:v", "1", "-q:v", "3", str(out / "start.jpg")], check=True)
write_beats()
m = json.loads((out / "film.json").read_text())
m["duration"] = a.end
(out / "film.json").write_text(json.dumps(m))
print(json.dumps({"end": a.end, "frag_bytes": frag.stat().st_size, "crf": a.crf, "gop": a.gop, "width": a.width}))
