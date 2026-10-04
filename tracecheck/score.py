"""Scores of probability maps against labels and against the organizers' published map, on the output grid.

AUC: probability that a randomly chosen labelled-ink cell scores higher than a randomly chosen labelled-papyrus
cell (Mann-Whitney U / (n_ink n_papyrus), ties counted one half). 0.5 is chance.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, stats

MARGIN = 0.05            # tolerance: offsets whose AUC stays within 0.05 of the on-surface AUC
SIGMA = 8                # smoothing (output cells) for the agreement r with the published map


def auc(values: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> float:
    a, b = values[pos].astype(np.float64), values[neg].astype(np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")
    ranks = stats.rankdata(np.concatenate([a, b]))
    u = ranks[:a.size].sum() - a.size * (a.size + 1) / 2.0
    return float(u / (a.size * b.size))


def tolerance(curve: dict, side: int, margin: float = MARGIN) -> int:
    """Furthest tested offset on one side (side -1 or +1) reached before the AUC first falls more than `margin`
    below the AUC at offset 0. 0 when the nearest offset on that side already falls."""
    base = curve[0]
    ks = sorted((k for k in curve if (k > 0 if side > 0 else k < 0)), key=abs)
    last = 0
    for k in ks:
        if curve[k] < base - margin:
            break
        last = k
    return last


def shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """out[y, x] = a[y - dy, x - dx]; zero/False where shifted in from outside."""
    out = np.zeros_like(a)
    h, w = a.shape
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = a[ys, xs]
    return out


def label_masks(frac: np.ndarray, supervised: np.ndarray, valid: np.ndarray):
    scored = supervised & valid
    return scored & (frac >= 0.5), scored & (frac == 0)


def rotated_null(values, frac, supervised, valid) -> dict:
    """Labels and supervision rotated 180 degrees inside the region, scored against the same map: what the AUC
    looks like when labels and map are unrelated but keep their own spatial structure."""
    pos, neg = label_masks(np.rot90(frac, 2), np.rot90(supervised, 2), valid)
    return {"auc": auc(values, pos, neg), "pos": int(pos.sum()), "neg": int(neg.sum())}


def alignment(values, frac, supervised, valid, reach: int = 8, step: int = 2) -> dict:
    """AUC with the labels moved (dy, dx) output cells in-plane. A best shift far above zero shift flags a label to
    CT misregistration that would bias every offset's AUC."""
    res = {}
    for dy in range(-reach, reach + 1, step):
        for dx in range(-reach, reach + 1, step):
            pos, neg = label_masks(shift(frac, dy, dx), shift(supervised, dy, dx), valid)
            res[f"{dy},{dx}"] = auc(values, pos, neg)
    best = max(res, key=res.get)
    dy, dx = (int(v) for v in best.split(","))
    return {"auc_by_shift": res, "best_shift": best, "best_minus_zero": res[best] - res["0,0"],
            "flag_misregistration": bool(res[best] - res["0,0"] > 0.02),
            "best_at_search_edge": bool(max(abs(dy), abs(dx)) >= reach)}


def published_grid(published: np.ndarray, out_scale: int = 4) -> np.ndarray:
    """Organizers' published uint8 map (surface-volume pixels) -> mean probability per output cell."""
    h, w = published.shape
    p = published[:h // out_scale * out_scale, :w // out_scale * out_scale].astype(np.float64) / 255.0
    return p.reshape(h // out_scale, out_scale, w // out_scale, out_scale).mean(axis=(1, 3))


def agreement(values: np.ndarray, ref: np.ndarray, valid: np.ndarray, sigma: int = SIGMA) -> dict:
    """Pearson r with the published map over valid cells (raw), and after Gaussian smoothing (sd `sigma` cells)
    of both maps inside the valid area, on cells at least 2 sigma from its edge (r_smooth)."""
    values = values.astype(np.float64)
    rec = {"r_raw": float(np.corrcoef(values[valid], ref[valid])[0, 1]) if valid.sum() > 2 else float("nan")}
    inner = ndimage.binary_erosion(valid, iterations=2 * sigma)
    if inner.sum() > 2:
        vs = ndimage.gaussian_filter(np.where(valid, np.nan_to_num(values), 0.0), sigma)
        rs = ndimage.gaussian_filter(np.where(valid, ref, 0.0), sigma)
        rec["r_smooth"] = float(np.corrcoef(vs[inner], rs[inner])[0, 1])
    else:
        rec["r_smooth"] = float("nan")
    rec["cells_raw"], rec["cells_smooth"] = int(valid.sum()), int(inner.sum())
    return rec
