"""Shared by the film's data scripts: inputs via ../filmdata.py, outputs into the Remotion app and the Blender folder."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import filmdata as F  # noqa: E402,F401

APP_DATA = HERE.parent / "app" / "public" / "data"            # what the Remotion film reads (staticFile("data/..."))
BLENDER_DATA = HERE.parent / "blender" / "data"               # what the Blender shots read
APP_DATA.mkdir(parents=True, exist_ok=True)
BLENDER_DATA.mkdir(parents=True, exist_ok=True)
BAND_Y0 = 8192                                                 # the band's first row: maps and depth start here (4x grid)

STOPS = [[6, 8, 12], [70, 34, 10], [196, 110, 30], [242, 182, 90], [255, 236, 190]]


def gold(v):
    """The film's and page's fixed ink ramp, probability 0..1 to RGB."""
    import numpy as np
    stops = np.array(STOPS, float)
    v = np.clip(np.nan_to_num(v), 0, 1)
    xp = np.linspace(0, 1, len(stops))
    return np.stack([np.interp(v, xp, stops[:, c]) for c in range(3)], -1).astype(np.uint8)
