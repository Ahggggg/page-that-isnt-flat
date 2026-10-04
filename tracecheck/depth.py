"""Where the papyrus sits relative to the traced surface, from the CT alone.

Per output cell (4 x 4 surface-volume pixels): average the 4 x 4 pixels of every plane, smooth the (planes, rows,
columns) stack with a Gaussian of sd (1.5, 3, 3) cells, weight each plane by a Gaussian prior (sd 30 planes) centred
on the mesh plane, and take the brightest plane. depth = that plane minus the mesh plane (planes; multiply by the
voxel size for um). A cell is measured when its mesh-plane pixels are all inside the rendered surface (covered) and
the smoothed peak exceeds 60 (out of 255); elsewhere the brightest plane is noise.

The brightest layer can be a neighbouring winding rather than the traced sheet; the prior only discourages that.
Large regions are processed in row strips with a halo of the Gaussian's reach, which gives exactly the same result
as filtering the whole region at once.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .model import FRAMES, MESH_PLANE, OUT_SCALE

PRIOR_SD = 30.0
SMOOTH = (1.5, 3.0, 3.0)
PEAK_MIN = 60.0
TRUNCATE = 4.0
HALO = int(TRUNCATE * SMOOTH[1] + 0.5)            # 12 cells: the in-plane reach of the Gaussian
WINDOW_LO = -(FRAMES // 2)                         # the centred window reads planes mesh-31 .. mesh+30
WINDOW_HI = FRAMES - FRAMES // 2 - 1


def reduce_stack(stack: np.ndarray, out: int = OUT_SCALE) -> np.ndarray:
    z, h, w = stack.shape
    return stack.reshape(z, h // out, out, w // out, out).mean(axis=(2, 4)).astype(np.float32)


def prior(planes: int, mesh_plane: int = MESH_PLANE, sd: float = PRIOR_SD) -> np.ndarray:
    return np.exp(-0.5 * ((np.arange(planes) - mesh_plane) / sd) ** 2).astype(np.float32)


def depth_from_reduced(reduced: np.ndarray, mesh_plane: int = MESH_PLANE):
    """Whole-array version (small regions, tests): (depth, peak) per cell."""
    sm = ndimage.gaussian_filter(reduced, SMOOTH, truncate=TRUNCATE)
    pr = prior(reduced.shape[0], mesh_plane)[:, None, None]
    return np.argmax(sm * pr, axis=0).astype(np.float32) - mesh_plane, sm.max(axis=0)


def sheet_depth(read, gh: int, gw: int, planes: int, mesh_plane: int = MESH_PLANE, block_rows: int = 32,
                strip_rows: int = 128):
    """(depth, peak, covered) on a gh x gw output grid.

    read(r0, r1) must return the uint8 (planes, 4 (r1 - r0), 4 gw) surface-volume rows of output rows r0..r1."""
    cache = {}

    def block(k):
        if k not in cache:
            r0, r1 = k * block_rows, min(gh, (k + 1) * block_rows)
            stack = read(r0, r1)
            cov = stack[mesh_plane].reshape(r1 - r0, OUT_SCALE, gw, OUT_SCALE).min(axis=(1, 3)) > 0
            cache[k] = (reduce_stack(stack), cov)
        return cache[k]

    def rows(a, b):
        ks = range(a // block_rows, (b - 1) // block_rows + 1)
        red = np.concatenate([block(k)[0] for k in ks], axis=1)
        cov = np.concatenate([block(k)[1] for k in ks], axis=0)
        off = a - ks[0] * block_rows
        return red[:, off:off + b - a], cov[off:off + b - a]

    pr = prior(planes, mesh_plane)[:, None, None]
    depth = np.empty((gh, gw), np.float32)
    peak = np.empty((gh, gw), np.float32)
    covered = np.empty((gh, gw), bool)
    for r0 in range(0, gh, strip_rows):
        r1 = min(gh, r0 + strip_rows)
        a, b = max(0, r0 - HALO), min(gh, r1 + HALO)
        part, cov = rows(a, b)
        top = HALO if a == 0 else 0                 # true region edges: reflect as a whole-region filter would
        bot = HALO if b == gh else 0
        padded = np.pad(part, ((0, 0), (top, bot), (0, 0)), mode="symmetric") if (top or bot) else part
        sm = ndimage.gaussian_filter(padded, SMOOTH, truncate=TRUNCATE)
        keep = slice(top + r0 - a, top + r1 - a)
        sm = sm[:, keep]
        depth[r0:r1] = np.argmax(sm * pr, axis=0).astype(np.float32) - mesh_plane
        peak[r0:r1] = sm.max(axis=0)
        covered[r0:r1] = cov[r0 - a:r1 - a]
        for k in [k for k in cache if (k + 1) * block_rows <= r1 - HALO]:
            del cache[k]
    return depth, peak, covered


def coverage(depth: np.ndarray, peak: np.ndarray, covered: np.ndarray, mask=None, voxel_um: float = 2.399) -> dict:
    """How far the papyrus sits from the traced surface over the measured cells (covered, peak > 60, and mask)."""
    m = covered & (peak > PEAK_MIN)
    if mask is not None:
        m &= mask
    d = depth[m]
    n = int(d.size)
    cell_mm2 = (OUT_SCALE * voxel_um / 1000.0) ** 2
    rec = {"cells": n, "area_mm2": round(n * cell_mm2, 3)}
    if n == 0:
        return rec
    lo, hi = -MESH_PLANE, MESH_PLANE
    rec.update({
        "median_planes": float(np.median(d)), "median_um": float(np.median(d)) * voxel_um,
        "frac_more_than_10_planes_off": float((np.abs(d) > 10).mean()),
        "frac_more_than_20_planes_off": float((np.abs(d) > 20).mean()),
        "frac_outside_centred_window": float(((d < WINDOW_LO) | (d > WINDOW_HI)).mean()),
        "window_planes": [WINDOW_LO, WINDOW_HI],
        "histogram": {"from_plane": lo, "counts": np.bincount((d - lo).astype(np.int64).clip(0, hi - lo),
                                                             minlength=hi - lo + 1).tolist()},
    })
    return rec
