"""The 'powers of ten' zoom: page -> band -> line -> letter -> papyrus fibres, every level a crop of real data.

Centre: the brightest 1.5 mm-smoothed spot of the published map inside the line window (band.json, so run band.py first).
  ink_L{0..3}.png  published ink prediction (grey 0-255), 2048 px wide each
  ct_L{3,4}.png    the surface volume's plane 54 (raw CT on the traced surface), same crop as ink L3 and a finer L4
  zoom.json        each level's centre and width (page px) and um per output px
"""
import json

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage

from _common import APP_DATA, F

OUT = APP_DATA / "zoom"
OUT.mkdir(parents=True, exist_ok=True)
band = json.loads((APP_DATA / "band.json").read_text())
ink = tifffile.imread(str(F.published_prediction()))
H, W = ink.shape
y0, y1 = band["line"] - band["half"], band["line"] + band["half"]
x0, x1 = band["x0"], band["x0"] + band["win"]
win = ndimage.gaussian_filter(ink[y0:y1, x0:x1].astype(np.float32), 120)
cy, cx = np.unravel_index(np.argmax(win), win.shape)
cy, cx = int(cy + y0), int(cx + x0)
widths = [20400, 8192, 3072, 900, 260]          # page, band, line, letter, fibres (page px at 2.399 um)
levels = []


def crop(arr, cy, cx, w, aspect=9 / 16):
    h = int(round(w * aspect))
    t, l = cy - h // 2, cx - w // 2
    out = np.zeros((h, w), arr.dtype)
    ys, xs = slice(max(0, t), min(arr.shape[0], t + h)), slice(max(0, l), min(arr.shape[1], l + w))
    out[ys.start - t:ys.stop - t, xs.start - l:xs.stop - l] = arr[ys, xs]
    return out, (t, l, h, w)


# CT surface plane over the region the two finest levels need
need = 1400
ry0, rx0 = ((cy - need) // 128) * 128, ((cx - need) // 128) * 128
reg = F.read_region(F.surface_volume(), ry0, rx0, 2 * need + 256, 2 * need + 256)[54]
# finest level: recentre on the most fibre-textured spot within 0.7 mm of the letter (highest local CT contrast)
rf = reg.astype(np.float32)
lstd = np.sqrt(np.maximum(ndimage.uniform_filter(rf ** 2, 120) - ndimage.uniform_filter(rf, 120) ** 2, 0))
ly, lx = cy - ry0, cx - rx0
sub = lstd[ly - 300:ly + 300, lx - 300:lx + 300]
dy, dx = np.unravel_index(np.argmax(sub), sub.shape)
fy, fx = int(cy - 300 + dy), int(cx - 300 + dx)
for i, w in enumerate(widths):
    c, box = crop(ink, cy if i else H // 2, cx if i else W // 2, w)
    if i < 4:
        Image.fromarray(c).resize((2048, 1152), Image.LANCZOS).save(OUT / f"ink_L{i}.png")
    cbox = None
    if i >= 3:
        ccy, ccx = (fy, fx) if i == 4 else (cy, cx)
        cw = 2048 if i == 3 else 520                 # CT crops wider than the ink ones so the view never runs past them
        ct, cbox = crop(reg, ccy - ry0, ccx - rx0, cw)
        cbox = (cbox[0] + ry0, cbox[1] + rx0, cbox[2], cbox[3])
        Image.fromarray(ct).resize((2048, 1152), Image.LANCZOS if w > 2048 else Image.BICUBIC).save(OUT / f"ct_L{i}.png")
    levels.append({"ct_box": cbox, "level": i, "cy": (fy if i == 4 else cy) if i else H // 2,
                   "cx": (fx if i == 4 else cx) if i else W // 2, "w": w, "um_per_out_px": w * 2.399 / 2048, "box": box})
json.dump({"centre": [cy, cx], "levels": levels, "page": [H, W]}, open(OUT / "zoom.json", "w"), indent=1)
print(json.dumps({"centre": [cy, cx], "mm_widths": [round(w * 2.399e-3, 2) for w in widths]}))
