"""'The cut' and 'the window': the text line, the 9.8 mm window along it, and CT cross-sections across it.

The line and window are chosen by rule from the published ink map (no hand-picking): the band row with the highest mean
ink over a 2 mm row window, then the 4096 px (9.8 mm) window along that row with the most ink. Writes xs_band_<row>.png
(109 planes x 4096 px, every 16 px over +-640 px about the line), ink_band.png (the published map over the same window),
surf_band.png (plane 54, the traced surface) and band.json (the geometry the other scripts and the film read).
Needs the surface volume over rows line-640..line+640 of the band (tracecheck fetch of the band region).
"""
import json

import numpy as np
import tifffile
from PIL import Image

from _common import APP_DATA, BAND_Y0, F

Y1, WIN, HALF = 15360, 4096, 640
ink = tifffile.imread(str(F.published_prediction()))
band = ink[BAND_Y0:Y1].astype(np.float32)
prof = np.convolve(band.mean(axis=1), np.ones(833) / 833, mode="same")          # 2 mm row window
line = int(np.argmax(prof[HALF:-HALF])) + HALF + BAND_Y0
row = ink[line - 200:line + 200].astype(np.float32).mean(axis=0)
csum = np.concatenate([[0], np.cumsum(row)])
x0 = int(np.argmax(csum[WIN:] - csum[:-WIN]))
x0 = (x0 // 128) * 128
vol = F.surface_volume()
ry0 = ((line - HALF) // 128) * 128
ry1 = ((line + HALF) // 128 + 1) * 128
cx1 = ((x0 + WIN) // 128 + 1) * 128
region = F.read_region(vol, ry0, x0, ry1 - ry0, cx1 - x0)[:, :, :WIN]
rows = list(range(line - HALF, line + HALF + 1, 16))
for r in rows:
    Image.fromarray(np.ascontiguousarray(region[:, r - ry0, :])).save(APP_DATA / f"xs_band_{r:05d}.png")
Image.fromarray(ink[line - HALF:line + HALF, x0:x0 + WIN]).save(APP_DATA / "ink_band.png")
Image.fromarray(np.ascontiguousarray(region[54, line - HALF - ry0:line + HALF - ry0])).save(APP_DATA / "surf_band.png")
json.dump({"line": line, "x0": x0, "win": WIN, "half": HALF, "rows": rows, "um_per_px": 2.399},
          open(APP_DATA / "band.json", "w"))
print(json.dumps({"line": line, "x0": x0, "rows": len(rows)}))
