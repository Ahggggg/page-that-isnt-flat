"""The 'crumpled page' shot (blender/relief.py): the real sheet surface around the zoom's letter (run zoom.py first).

Height: the CT-only sheet depth from `tracecheck depth` (planes of 2.399 um) on the 4x grid (9.6 um/px), unmeasured cells
filled from neighbours, a 9-cell median to remove one-layer jumps, then a Gaussian of 7 cells (~70 um). The film shows
it exaggerated x6 and says so. Textures: the published ink map and the surface volume's plane 54 over the same region
(7.4 x 4.1 mm).
"""
import json

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage

from _common import APP_DATA, BAND_Y0, BLENDER_DATA, F

cy, cx = json.loads((APP_DATA / "zoom" / "zoom.json").read_text())["centre"]
W, H = 3072, 1728                                    # page px (7.37 x 4.15 mm)
t, l = cy - H // 2, cx - W // 2
z = np.load(F.SHEET_DEPTH)
d, peak, cov = z["depth"], z["peak"], z["covered"]
gt, gl, gh, gw = (t - BAND_Y0) // 4, l // 4, H // 4, W // 4
dd = d[gt:gt + gh, gl:gl + gw].astype(np.float32)
ok = (peak[gt:gt + gh, gl:gl + gw] > 60) & cov[gt:gt + gh, gl:gl + gw]
idx = ndimage.distance_transform_edt(~ok, return_distances=False, return_indices=True)
filled = dd[tuple(idx)]
height = ndimage.gaussian_filter(ndimage.median_filter(filled, 9), 7)
np.savez_compressed(BLENDER_DATA / "relief.npz", height_planes=height.astype(np.float32), valid=ok, um_per_px=4 * 2.399,
                    region_page=np.array([t, l, H, W]))
ink = tifffile.imread(str(F.published_prediction()))[t:t + H, l:l + W]
Image.fromarray(ink).save(BLENDER_DATA / "relief_ink.png")
ry0, rx0 = (t // 128) * 128, (l // 128) * 128
reg = F.read_region(F.surface_volume(), ry0, rx0, ((t + H - ry0) // 128 + 1) * 128, ((l + W - rx0) // 128 + 1) * 128)[54]
Image.fromarray(np.ascontiguousarray(reg[t - ry0:t - ry0 + H, l - rx0:l - rx0 + W])).save(BLENDER_DATA / "relief_ct.png")
print(json.dumps({"grid": list(height.shape), "valid": round(float(ok.mean()), 3),
                  "height_planes_p5_p95": [round(float(v), 1) for v in np.percentile(height, [5, 95])]}))
