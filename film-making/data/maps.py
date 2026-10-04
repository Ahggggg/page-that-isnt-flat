"""Model-output images from the band run's maps (`tracecheck offsets ... --out runs/band`), one fixed gold ramp (0..1).

  arms/arm_<k>.jpg    each window offset's map over the line window (band.json; run band.py first), 1024 x 320
  slab/band_+0.png    the whole band, window on the trace, and slab/band_R0.png, the same layers back to front
Every image is a separate model run; nothing is blended here.
"""
import json

import numpy as np
from PIL import Image

from _common import APP_DATA, BAND_Y0, F, gold

band = json.loads((APP_DATA / "band.json").read_text())
line, x0, win, half = band["line"], band["x0"], band["win"], band["half"]
r0, r1 = (line - half - BAND_Y0) // 4, (line + half - BAND_Y0) // 4
c0, c1 = x0 // 4, (x0 + win) // 4
(APP_DATA / "arms").mkdir(exist_ok=True)
(APP_DATA / "slab").mkdir(exist_ok=True)
done = []
for path in sorted(F.MAPS.glob("*.npy")):
    kind, k = path.stem[0], int(path.stem[1:])                 # tracecheck names: k-16, k+0, R+0
    name = "R0" if kind == "R" else str(k)
    m = np.load(path).astype(np.float32)
    Image.fromarray(gold(m[r0:r1, c0:c1])).save(APP_DATA / "arms" / f"arm_{name}.jpg", quality=90)
    if k == 0:
        Image.fromarray(gold(m)).save(APP_DATA / "slab" / f"band_{'R0' if kind == 'R' else '+0'}.png")
    done.append(name)
print(json.dumps({"arms": done}))
