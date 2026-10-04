"""Split a fragmented MP4 into init.mp4 (ftyp+moov) and segment files of whole moof+mdat pairs (<= LIMIT bytes each),
for Media Source Extensions playback from published files (15 MB per-file limit)."""
import json, struct, sys
from pathlib import Path
src, out, LIMIT = Path(sys.argv[1]), Path(sys.argv[2]), 12_000_000
data = src.read_bytes()
boxes, i = [], 0
while i < len(data):
    size, typ = struct.unpack(">I4s", data[i:i + 8])
    if size == 1:
        size = struct.unpack(">Q", data[i + 8:i + 16])[0]
    boxes.append((typ.decode(), i, size)); i += size
init_end = max(o + s for t, o, s in boxes if t in ("ftyp", "moov"))
(out / "init.mp4").write_bytes(data[:init_end])
frags, cur = [], None
for t, o, s in boxes:
    if t == "moof":
        cur = [o, o + s]; frags.append(cur)
    elif t == "mdat" and cur:
        cur[1] = o + s
segs, start, end = [], frags[0][0], frags[0][0]
for a, b in frags:
    if b - start > LIMIT and end > start:
        segs.append((start, end)); start = a
    end = b
segs.append((start, end))
names = []
for k, (a, b) in enumerate(segs):
    n = f"seg{k}.m4s"; (out / n).write_bytes(data[a:b]); names.append(n)
manifest = {"codec": 'video/mp4; codecs="avc1.640028"', "init": "init.mp4", "segments": names, "duration": 129.0}
(out / "film.json").write_text(json.dumps(manifest))
print(json.dumps({"boxes": len(boxes), "fragments": len(frags), "segments": [b - a for a, b in segs], "init": init_end}))
