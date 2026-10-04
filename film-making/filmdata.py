"""Where the film's data scripts find their inputs, and the two readers they share.

Inputs follow tracecheck's local layout, so one download serves both (see FILM.md, step 1):
  - the w035 surface volume and the organizers' published prediction: `python -m tracecheck fetch --segment w035 ...`
  - the CT-only sheet depth over the band: `python -m tracecheck depth ... --out runs/band` (depth.npz)
  - the model's maps at each window offset: `python -m tracecheck offsets ... --out runs/band` (maps/k*.npy)
  - the traced mesh and the whole-scroll CT for the opening shot: `python film-making/fetch_film_inputs.py`
Override the locations with TRACECHECK_DATA, FILM_BAND_RUN and FILM_INPUTS.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]                     # the repository root
sys.path.insert(0, str(ROOT))
from tracecheck import sources  # noqa: E402
from tracecheck.volume import SurfaceVolume  # noqa: E402

DATA = Path(os.environ.get("TRACECHECK_DATA", ROOT / "data"))
BAND = Path(os.environ.get("FILM_BAND_RUN", ROOT / "runs" / "band"))
INPUTS = Path(os.environ.get("FILM_INPUTS", ROOT / "film-inputs"))

SEGMENT = "w035"
SHEET_DEPTH = BAND / "depth.npz"                               # keys: depth (planes, + = below the trace), peak, covered
MAPS = BAND / "maps"                                           # k-23.npy ... k+23.npy, R+0.npy (one value per 4 x 4 px)
MESH = INPUTS / "w035.tifxyz"                                  # meta.json, x.tif, y.tif, z.tif
SCROLL_L5 = INPUTS / "scroll_L5.npy"                           # PHerc0139 CT, pyramid level 5 (76.8 um), whole scroll
SCROLL_L4_Z9 = INPUTS / "scroll_L4_z9.npy"                     # level 4 (38.4 um), the 128 slices around w035's middle


def _segment():
    return sources.segment(sources.load_manifest(), SEGMENT)


def published_prediction() -> Path:
    """The organizers' published ink prediction for w035 (a greyscale .tif on the page canvas)."""
    path = _segment().published_path(DATA)
    if path is None or not path.exists():
        raise SystemExit(f"published prediction not found under {DATA}; run tracecheck fetch --segment {SEGMENT}")
    return path


def surface_volume() -> SurfaceVolume:
    """Level 0 of the w035 surface volume (109 planes, the traced surface at plane 54)."""
    return SurfaceVolume(_segment().surface_level(DATA))


def read_region(vol: SurfaceVolume, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """(109, h, w) uint8. Inside the volume every chunk must have been fetched (chunks the bucket does not have read as 0);
    anything past the volume's edge reads as 0, so a crop near the page's edge is padded rather than refused."""
    out = np.zeros((vol.shape[0], h, w), np.uint8)
    ya, xa = max(0, y0), max(0, x0)
    yb, xb = min(vol.shape[1], y0 + h), min(vol.shape[2], x0 + w)
    if yb > ya and xb > xa:
        out[:, ya - y0:yb - y0, xa - x0:xb - x0] = vol.read(ya, xa, yb - ya, xb - xa)
    return out


class Mesh:
    """A tifxyz sheet: node positions (rows, cols, 3) as x, y, z voxels, their validity, and the grid scale."""

    def __init__(self, xyz: np.ndarray, scale: float):
        self.xyz = xyz.astype(np.float64)
        self.scale = float(scale)
        self.valid = np.isfinite(self.xyz).all(axis=2) & (self.xyz > 0).all(axis=2)


def load_tifxyz(directory) -> Mesh:
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    xyz = np.stack([tifffile.imread(str(directory / f"{a}.tif")) for a in "xyz"], axis=-1)
    scale = meta.get("scale", [1.0, 1.0])
    if abs(scale[0] - scale[1]) > 1e-9:
        raise ValueError(f"{directory}: anisotropic tifxyz scale {scale} is not supported")
    return Mesh(xyz, scale[0])


def arm_map(k: int, reversed_order: bool = False) -> Path:
    """The band run's map with the window k planes off the trace (tracecheck's naming)."""
    return MAPS / (f"R{k:+d}.npy" if reversed_order else f"k{k:+d}.npy")
