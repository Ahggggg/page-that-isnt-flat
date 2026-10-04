"""Read regions of the organizers' surface volumes: zarr v2, uint8, `compressor: null`, so every chunk is a raw
C-order block of bytes that needs no zarr package. Axis order (planes, rows, columns); level 0 is full resolution.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


class SurfaceVolume:
    """Level 0 of a local surface-volume zarr: `root` is the level directory (holding `.zarray`)."""

    def __init__(self, root):
        self.root = Path(root)
        meta = json.loads((self.root / ".zarray").read_text(encoding="utf-8"))
        if meta.get("zarr_format") != 2 or meta.get("compressor") is not None or meta.get("dtype") != "|u1" \
                or meta.get("filters") not in (None, []) or meta.get("order", "C") != "C":
            raise ValueError(f"{self.root}: only uncompressed uint8 C-order zarr v2 arrays are supported, got {meta}")
        self.shape = tuple(meta["shape"])
        self.chunks = tuple(meta["chunks"])
        self.sep = meta.get("dimension_separator", ".")
        self.fill = meta.get("fill_value") or 0
        path = self.root / "absent.json"
        self.absent = set(json.loads(path.read_text())) if path.exists() else set()

    def key(self, c) -> str:
        return self.sep.join(str(v) for v in c)

    def chunk_coords(self, y0: int, x0: int, h: int, w: int) -> list[tuple[int, int, int]]:
        cz, cy, cx = self.chunks
        return [(z, a, b) for z in range(math.ceil(self.shape[0] / cz))
                for a in range(y0 // cy, (y0 + h - 1) // cy + 1)
                for b in range(x0 // cx, (x0 + w - 1) // cx + 1)]

    def check_region(self, y0, x0, h, w):
        if y0 < 0 or x0 < 0 or h <= 0 or w <= 0 or y0 + h > self.shape[1] or x0 + w > self.shape[2]:
            raise ValueError(f"region {(y0, x0, h, w)} is outside the surface volume ({self.shape[1]} x "
                             f"{self.shape[2]} px)")

    def read(self, y0: int, x0: int, h: int, w: int, out=None) -> np.ndarray:
        """(planes, h, w) uint8. A chunk missing on disk raises unless the fetch step recorded it as absent on the
        server (zarr: absent = fill value), so an interrupted download never reads as empty papyrus."""
        self.check_region(y0, x0, h, w)
        if out is None:
            out = np.empty((self.shape[0], h, w), np.uint8)
        cz, cy, cx = self.chunks
        size = cz * cy * cx
        for c in self.chunk_coords(y0, x0, h, w):
            k = self.key(c)
            path = self.root / k
            zs, ze = c[0] * cz, min((c[0] + 1) * cz, self.shape[0])
            ys, xs = max(y0, c[1] * cy), max(x0, c[2] * cx)
            ye, xe = min(y0 + h, (c[1] + 1) * cy), min(x0 + w, (c[2] + 1) * cx)
            if path.exists():
                data = np.fromfile(path, np.uint8)
                if data.size != size:
                    raise ValueError(f"{path}: {data.size} bytes, expected {size} (chunk {self.chunks})")
                block = data.reshape(self.chunks)
                out[zs:ze, ys - y0:ye - y0, xs - x0:xe - x0] = \
                    block[:ze - zs, ys - c[1] * cy:ye - c[1] * cy, xs - c[2] * cx:xe - c[2] * cx]
            elif k in self.absent:
                out[zs:ze, ys - y0:ye - y0, xs - x0:xe - x0] = self.fill
            else:
                raise FileNotFoundError(f"chunk {k} is not on disk under {self.root}; run `fetch` for this region")
        return out


def voxel_um(zarr_root, default: float) -> float:
    """Voxel size (um) from the OME-Zarr `.zattrs` of the surface-volume zarr, else `default`."""
    path = Path(zarr_root) / ".zattrs"
    try:
        attrs = json.loads(path.read_text(encoding="utf-8"))
        return float(attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"][0])
    except (OSError, KeyError, IndexError, ValueError):
        return default
