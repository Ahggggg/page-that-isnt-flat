"""report.json and report.png from a run directory (depth.json, maps.json, score.json). Curves only: no label or
CT image is drawn, so the figure can be shared without the data's image terms."""
from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import model as M

INK = (32, 32, 32)
GRID = (222, 222, 222)
BLUE = (31, 94, 168)
ORANGE = (204, 102, 0)
GREY = (130, 130, 130)
RED = (176, 40, 40)


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def build(out: Path) -> dict:
    depth, maps, score = (_load(Path(out) / n) for n in ("depth.json", "maps.json", "score.json"))
    rec = {"run": str(out)}
    if depth:
        rec["depth"] = {k: depth[k] for k in ("segment", "region", "voxel_um", "method") if k in depth}
        for part in ("all_measured", "label_scored"):
            if part in depth:
                rec["depth"][part] = {k: v for k, v in depth[part].items() if k != "histogram"}
    if maps:
        cfg = maps["config"]
        rec["inference"] = {k: cfg[k] for k in ("tile", "stride", "precision", "region") if k in cfg}
        arms = {p.stem: json.loads(p.read_text()) for p in sorted((Path(out) / "maps").glob("*.json"))}
        rec["inference"]["arms"] = {n: {k: a[k] for k in ("window", "tiles", "device", "seconds", "s_per_tile")}
                                    for n, a in arms.items()}
    if score:
        rows = []
        for n, a in sorted(score["arms"].items(), key=lambda t: (t[1]["reversed"], t[1]["offset_planes"])):
            rows.append({"arm": n, "offset_planes": a["offset_planes"], "offset_um": a["offset_um"],
                         "reversed": a["reversed"], "auc": a.get("auc"), "contrast": a.get("contrast"),
                         "r_published_raw": a.get("published", {}).get("r_raw"),
                         "r_published_smooth": a.get("published", {}).get("r_smooth")})
        rec["curve"] = rows
        for k in ("labels", "tolerance", "best_offset_planes", "nulls", "region_note"):
            if k in score:
                rec[k] = score[k]
        if "alignment" in score:
            al = score["alignment"]
            rec["alignment"] = {"best_shift_cells": al["best_shift"], "best_minus_zero": al["best_minus_zero"],
                                "cell_um": al.get("cell_um")}
    return rec


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                   # Pillow < 10.1
        return ImageFont.load_default()


def _panel(draw, box, xs, series, xlabel, ylabel, ylim, hlines=(), vlines=(), font=None, small=None,
           legend_top=False):
    x0, y0, x1, y1 = box
    xmin, xmax = min(xs), max(xs)
    if xmax == xmin:
        xmin, xmax = xmin - 1, xmax + 1
    ymin, ymax = ylim
    sx = lambda v: x0 + (v - xmin) / (xmax - xmin) * (x1 - x0)          # noqa: E731
    sy = lambda v: y1 - (v - ymin) / (ymax - ymin) * (y1 - y0)          # noqa: E731
    for i in range(6):
        v = ymin + i * (ymax - ymin) / 5
        draw.line([(x0, sy(v)), (x1, sy(v))], fill=GRID)
        draw.text((x0 - 8, sy(v)), f"{v:.2f}", fill=INK, font=small, anchor="rm")
    step = 10 if xmax - xmin <= 80 else 20
    v = math.ceil(xmin / step) * step
    while v <= xmax:
        draw.line([(sx(v), y0), (sx(v), y1)], fill=GRID)
        draw.text((sx(v), y1 + 6), f"{v:g}", fill=INK, font=small, anchor="mt")
        v += step
    for v, colour, text, *side in vlines:
        if xmin <= v <= xmax:
            draw.line([(sx(v), y0), (sx(v), y1)], fill=colour, width=1)
            left = side and side[0] == "left"
            draw.text((sx(v) - 3 if left else sx(v) + 3, y0 + 2), text, fill=colour, font=small,
                      anchor="ra" if left else "la")
    for v, colour, text in hlines:
        if v is not None and ymin <= v <= ymax:
            for xa in range(int(x0), int(x1), 12):
                draw.line([(xa, sy(v)), (min(xa + 6, x1), sy(v))], fill=colour, width=2)
            draw.text((x1 - 4, sy(v) - 3), text, fill=colour, font=small, anchor="rb")
    for ys, colour, text in series:
        pts = [(sx(x), sy(y)) for x, y in zip(xs, ys) if y is not None and not math.isnan(y)]
        if len(pts) > 1:
            draw.line(pts, fill=colour, width=3)
        for px, py in pts:
            draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=colour)
    draw.rectangle(box, outline=INK)
    draw.text(((x0 + x1) / 2, y1 + 26), xlabel, fill=INK, font=font, anchor="mt")
    draw.text((x0, y0 - 8), ylabel, fill=INK, font=font, anchor="ld")
    lx, ly = x0 + 10, (y0 + 16 if legend_top else y1 - 16)
    for _, colour, text in series:
        if text:
            draw.rectangle([lx, ly - 6, lx + 14, ly + 6], fill=colour)
            draw.text((lx + 20, ly), text, fill=INK, font=small, anchor="lm")
            lx += 30 + int(draw.textlength(text, font=small))


def plot(rec: dict, path: Path) -> None:
    W, H = 1200, 1000
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    font, small, title = _font(18), _font(14), _font(22)
    seg = rec.get("depth", {}).get("segment") or ""
    region = rec.get("depth", {}).get("region") or rec.get("inference", {}).get("region")
    draw.text((40, 20), f"trace check  {seg}  region {region}", fill=INK, font=title)
    curve = [r for r in rec.get("curve", []) if not r["reversed"]]
    if curve:
        xs = [r["offset_um"] for r in curve]
        aucs = [r["auc"] for r in curve]
        nulls = rec.get("nulls", {})
        rev = next((r for r in rec.get("curve", []) if r["reversed"] and r["offset_planes"] == 0), None)
        base = next((r["auc"] for r in curve if r["offset_planes"] == 0), None)
        if any(a is not None for a in aucs):
            lo = min([a for a in aucs if a is not None] + [v for v in (nulls.get("rotated_labels_k0", {}).get("auc"),
                     rev and rev["auc"]) if v is not None] + [0.4])
            hlines = [(base - 0.05 if base is not None else None, GREY, "on-surface AUC - 0.05"),
                      (rev["auc"] if rev else None, RED, "reversed depth order"),
                      (nulls.get("rotated_labels_k0", {}).get("auc"), ORANGE, "labels rotated 180 deg"),
                      (nulls.get("published_map_auc"), BLUE, "organizers' published map")]
            _panel(draw, (110, 90, 1150, 450), xs, [(aucs, INK, "model AUC vs labels")],
                   "window offset from the traced surface (um)", "label AUC (ink vs papyrus cells)",
                   (math.floor(lo * 10) / 10, 1.0), hlines, [(0, GREY, "trace")], font, small, legend_top=True)
        rs = [r["r_published_raw"] for r in curve]
        if any(v is not None for v in rs):
            rsm = [r["r_published_smooth"] for r in curve]
            vals = [v for v in rs + rsm if v is not None and not math.isnan(v)]
            _panel(draw, (110, 530, 600, 860), xs, [(rs, BLUE, "raw"), (rsm, ORANGE, "smoothed")],
                   "window offset (um)", "Pearson r with the published map",
                   (math.floor(min(vals + [0.0]) * 10) / 10, 1.0), [], [(0, GREY, "")], font, small)
    dep = None
    for part in ("label_scored", "all_measured"):
        if part in rec.get("depth", {}):
            dep = (part, rec["depth"][part])
            break
    hist = _load(Path(rec["run"]) / "depth.json") if rec.get("run") else None
    if dep and hist and dep[0] in hist and "histogram" in hist[dep[0]]:
        h = hist[dep[0]]["histogram"]
        vu = rec["depth"].get("voxel_um", 2.399)
        counts = h["counts"]
        total = max(1, sum(counts))
        xs = [(h["from_plane"] + i) * vu for i in range(len(counts))]
        ys = [c / total for c in counts]
        lo, hi = -(M.FRAMES // 2) * vu, (M.FRAMES - M.FRAMES // 2 - 1) * vu
        v = dep[1]
        _panel(draw, (680, 530, 1150, 860), xs, [(ys, INK, "")], "CT brightest layer minus trace (um)",
               f"share of {'label-scored' if dep[0] == 'label_scored' else 'measured'} cells",
               (0, max(ys + [0.01]) * 1.1), [], [(lo, RED, "62-layer window", "left"), (hi, RED, ""),
                                               (-10 * vu, GREY, "10"), (10 * vu, GREY, "10"),
                                               (-20 * vu, GREY, "20", "left"), (20 * vu, GREY, "20")],
               font, small)
        draw.text((680, 935), f">10 planes off {v.get('frac_more_than_10_planes_off', 0):.1%}   "
                  f">20 off {v.get('frac_more_than_20_planes_off', 0):.1%}   outside window "
                  f"{v.get('frac_outside_centred_window', 0):.2%}", fill=INK, font=small)
    tol = rec.get("tolerance")
    if tol:
        draw.text((110, 935), f"AUC within {tol['margin']} of on-surface from {tol['um']['minus']} to "
                  f"+{tol['um']['plus']} um (tested offsets)", fill=INK, font=small)
    draw.text((110, 965), "Labels may be training labels: no AUC here is a held-out accuracy.", fill=GREY, font=small)
    img.save(path)
