"""The cut through the whole scroll at w035's mid-height (level-4 CT, 38.4 um/px), and where w035 crosses that plane.
Inputs from fetch_film_inputs.py. Outputs for blender/scroll.py:
  slice_L4.png         the cut face (mean of 3 slices)
  slice_alpha.png      its opacity: grey level minus 20, ramped over 40 levels (reproduces the film's file exactly)
  slice_glow.png       the glow along w035: its crossing points blurred (sigma 3 px) and scaled to 255. The script that made
                       the film's glow was not kept; this rebuild correlates 0.986 with it.
  slice_w035_pts.npy   the crossing points (level-4 pixel x, y)"""
import json

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from _common import BLENDER_DATA, F

layer = np.load(F.SCROLL_L4_Z9)
Z0 = 18700                                                    # level-0 z of the cut (w035 mid-height)
k = Z0 // 16 - 9 * 128
img = layer[max(0, k - 1):k + 2].mean(axis=0).astype(np.uint8)
Image.fromarray(img).save(BLENDER_DATA / "slice_L4.png")
m = F.load_tifxyz(F.MESH)
xyz = m.xyz[m.valid]
near = xyz[np.abs(xyz[:, 2] - Z0) < 12]
pts = near[:, :2] / 16.0                                      # level-4 pixel (x, y)
np.save(BLENDER_DATA / "slice_w035_pts.npy", pts.astype(np.float32))
alpha = (np.clip(img.astype(float) - 20, 0, 40) / 40 * 255).astype(np.uint8)
Image.fromarray(alpha).save(BLENDER_DATA / "slice_alpha.png")
dots = np.zeros(img.shape, float)
dots[np.clip(pts[:, 1].round().astype(int), 0, img.shape[0] - 1), np.clip(pts[:, 0].round().astype(int), 0, img.shape[1] - 1)] = 1
glow = ndimage.gaussian_filter(dots, 3)
Image.fromarray(np.round(glow / max(glow.max(), 1e-12) * 255).astype(np.uint8)).save(BLENDER_DATA / "slice_glow.png")
prev = Image.fromarray(img).convert("RGB")
d = ImageDraw.Draw(prev)
for x, y in pts[::5]:
    d.point((float(x), float(y)), fill=(255, 170, 40))
prev.resize((828, 828)).save(BLENDER_DATA / "slice_preview.png")
print(json.dumps({"slice_px": img.shape, "w035_points": int(len(near)), "z_L0": Z0}))
