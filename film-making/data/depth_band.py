"""The film's whole-band sheet-offset map (depth_band.png) from the band's CT-only depth (`tracecheck depth`).

Smoothed for display with a Gaussian of 10 cells (96 um, "about 0.1 mm") over measured cells only (covered, peak > 60;
no data where under 30% of the kernel was measured),
then coloured on a straight ramp: dark on the trace, blue for papyrus above it, orange below, full colour at 20 layers
(48 um), then downsized by half (2550 x 896, Lanczos). This recipe reproduces the film's image to within a few grey
levels (the one-off script that first made it was not kept); the percentages on screen come from the unsmoothed depth.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

from _common import APP_DATA, F

z = np.load(F.SHEET_DEPTH)
d = z["depth"].astype(np.float64)
ok = z["covered"] & (z["peak"] > 60)
num = ndimage.gaussian_filter(np.where(ok, d, 0), 10)
den = ndimage.gaussian_filter(ok.astype(float), 10)
sm = np.where(den > 0.3, num / np.maximum(den, 1e-9), np.nan)          # no data where under 30% of the kernel was measured
t = np.clip(np.nan_to_num(sm) / 20, -1, 1)[..., None]
dark, blue, orange = np.array([18, 22, 30.]), np.array([20, 70, 160.]), np.array([235, 140, 50.])
rgb = np.where(t < 0, dark + (blue - dark) * -t, dark + (orange - dark) * t)
rgb[~np.isfinite(sm)] = [6, 8, 12]
img = Image.fromarray(np.round(rgb).astype(np.uint8))
img.resize((img.width // 2, img.height // 2), Image.LANCZOS).save(APP_DATA / "depth_band.png")
print("depth_band", (img.height // 2, img.width // 2))
