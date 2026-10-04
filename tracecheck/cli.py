"""trace-check: how far does the papyrus sit from a traced surface, and what does that cost the ink model?

    python -m tracecheck fetch   --segment w035 --region Y0 X0 H W --model     # inputs, every file checked
    python -m tracecheck depth   --segment w035 --region Y0 X0 H W --out run    # CT-only sheet depth + coverage
    python -m tracecheck offsets --segment w035 --region Y0 X0 H W --out run    # model maps, window moved k planes
    python -m tracecheck score   --segment w035 --region Y0 X0 H W --out run    # label AUC + published-map r
    python -m tracecheck report  --out run                                      # report.json + report.png

Regions are (first row, first column, height, width) in surface-volume pixels, zero-based, half-open, multiples
of 4. See REPRODUCE.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import depth as D
from . import labels as L
from . import model as M
from . import score as S
from . import sources
from .volume import SurfaceVolume, voxel_um

DEFAULT_OFFSETS = [-23, -16, -8, -4, 0, 4, 8, 16, 23]


def log(obj):
    print(json.dumps(obj), flush=True)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=True))
    tmp.replace(path)


def sha256_bytes(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------------- arguments
def add_segment(ap):
    ap.add_argument("--segment", help="preset name from manifest.json (w030, w035, w039, w040, w041, w043, w044, w045)")
    ap.add_argument("--prefix", help="any segment: its bucket prefix, e.g. PHerc0139/segments/<id>")
    ap.add_argument("--surface", help="surface-volume zarr relative to the segment (default: the 2.399 um volume)")
    ap.add_argument("--labels", help="label version directory relative to the segment")
    ap.add_argument("--published", help="organizers' published prediction (.tif) relative to the segment")
    ap.add_argument("--data-dir", default="data", help="local mirror of the public files (default ./data)")
    ap.add_argument("--manifest", default=str(sources.MANIFEST))


def add_region(ap, required=True):
    ap.add_argument("--region", nargs=4, type=int, metavar=("Y0", "X0", "H", "W"), required=required,
                    help="surface-volume pixels: first row, first column, height, width (multiples of 4)")


def get_segment(args):
    manifest = sources.load_manifest(args.manifest)
    return manifest, sources.segment(manifest, args.segment, args.prefix, args.surface, args.labels, args.published)


def check_region(region, minimum=4):
    y0, x0, h, w = region
    if any(v % 4 for v in region):
        raise SystemExit(f"--region values must be multiples of 4, got {region}")
    if h < minimum or w < minimum:
        raise SystemExit(f"--region must be at least {minimum} px on each side for this step, got {h} x {w}")
    return y0, x0, h, w


def run_meta(args, seg):
    level = seg.surface_level(args.data_dir)
    return {"segment": seg.name, "prefix": seg.prefix, "surface": seg.surface, "region": list(args.region),
            "voxel_um": voxel_um(level.parent, 2.399), "grid": "output cell (r, c) = surface-volume pixels "
            "(Y0 + 4r .. Y0 + 4r + 3, X0 + 4c .. X0 + 4c + 3)"}


# --------------------------------------------------------------------------------------------------------- fetch
def cmd_fetch(args):
    manifest, seg = get_segment(args)
    data = Path(args.data_dir)
    if args.list:
        for sub in ("ink-detection/", "ink-labels/", "surface-volumes/"):
            keys = sources.list_keys(seg.prefix + sub)
            depth = 2 if sub == "ink-labels/" else 1          # label version/date; file or zarr name otherwise
            tops = sorted({"/".join(k[len(seg.prefix + sub):].split("/")[:depth]) for k, _ in keys})
            log({"under": seg.prefix + sub, "objects": len(keys), "entries": tops[:50]})
        return 0
    total = {}
    if args.model:
        total["model"] = sources.run(sources.model_items(manifest, data / "model"), data, args.check_only)
        log({"model": total["model"]})
    want_labels, want_pub = not args.no_labels, not args.no_published
    total["metadata"] = sources.run(sources.meta_items(seg, data, want_labels, want_pub), data, args.check_only)
    log({"metadata": total["metadata"]})
    if args.region:
        check_region(args.region)
        items = sources.region_items(seg, data, args.region, want_labels)
        total["region"] = sources.run(items, data, args.check_only, workers=args.workers)
        log({"region": total["region"]})
    log({"fetch": "done", "check_only": args.check_only,
         "downloaded_bytes": sum(v.get("downloaded_bytes", 0) for v in total.values())})
    return 0


# --------------------------------------------------------------------------------------------------------- depth
def scored_mask(args, seg, region, valid=None):
    """Label-scored cells (supervised; ink or papyrus, not mixed) of a region, or None without labels."""
    if args.no_labels or not seg.labels:
        return None
    y0, x0, h, w = region
    ink = L.read_region(seg.label_array(args.data_dir, "inklabels.zarr"), y0, x0, h, w)
    sup = L.read_region(seg.label_array(args.data_dir, "supervision.zarr"), y0, x0, h, w)
    frac, supervised = L.label_grid(ink, sup)
    return frac, supervised


def cmd_depth(args):
    _, seg = get_segment(args)
    y0, x0, h, w = check_region(args.region)
    vol = SurfaceVolume(seg.surface_level(args.data_dir))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    planes = vol.shape[0]
    d, peak, covered = D.sheet_depth(lambda r0, r1: vol.read(y0 + 4 * r0, x0, 4 * (r1 - r0), w), h // 4, w // 4,
                                     planes, args.mesh_plane)
    np.savez_compressed(out / "depth.npz", depth=d, peak=peak, covered=covered)
    meta = run_meta(args, seg)
    vu = meta["voxel_um"]
    rec = {**meta, "method": {"prior_sd_planes": D.PRIOR_SD, "smooth_cells": list(D.SMOOTH), "peak_min": D.PEAK_MIN,
                              "mesh_plane": args.mesh_plane, "planes": planes},
           "all_measured": D.coverage(d, peak, covered, None, vu)}
    labs = scored_mask(args, seg, (y0, x0, h, w))
    if labs is not None:
        frac, supervised = labs
        rec["label_scored"] = D.coverage(d, peak, covered, supervised & ((frac >= 0.5) | (frac == 0)), vu)
    rec["seconds"] = round(time.monotonic() - t0, 1)
    rec["depth_sha256"] = sha256_bytes(out / "depth.npz")
    write_json(out / "depth.json", rec)
    log({"depth": "done", "seconds": rec["seconds"],
         "frac_more_than_10_planes_off": rec["all_measured"].get("frac_more_than_10_planes_off")})
    return 0


# ------------------------------------------------------------------------------------------------------- offsets
def cmd_offsets(args):
    manifest, seg = get_segment(args)
    y0, x0, h, w = check_region(args.region, M.TILE)
    out = Path(args.out)
    (out / "maps").mkdir(parents=True, exist_ok=True)
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    model_dir = Path(args.data_dir) / "model"
    ckpt = Path(args.checkpoint) if args.checkpoint else model_dir / manifest["checkpoint"]["file"]
    villa = Path(args.villa_dir) if args.villa_dir else model_dir / "villa"
    t0 = time.monotonic()
    vol = SurfaceVolume(seg.surface_level(args.data_dir))
    arms = [(k, False) for k in args.offsets] + ([(0, True)] if args.reverse_null else [])
    for k, _ in arms:
        M.window_for_offset(k, vol.shape[0], args.mesh_plane)          # fail early on an impossible offset
    config = {**run_meta(args, seg), "tile": M.TILE, "stride": args.stride, "precision": args.precision,
              "mesh_plane": args.mesh_plane, "planes": vol.shape[0],
              "checkpoint_sha256": manifest["checkpoint"]["sha256"]}
    config_path = out / "maps.json"
    if config_path.exists():
        old = json.loads(config_path.read_text())["config"]
        for key in ("region", "segment", "stride", "precision", "mesh_plane", "planes", "checkpoint_sha256"):
            if old.get(key) != config[key]:
                raise SystemExit(f"{out} holds maps made with {key}={old.get(key)!r}, not {config[key]!r}; "
                                 f"use another --out")
    else:
        write_json(config_path, {"config": config})
    todo = []
    for k, rev in arms:
        name = M.arm_name(k, rev)
        old = read_arms(out).get(name)
        npy = out / "maps" / f"{name}.npy"
        if old and npy.exists() and sha256_bytes(npy) == old["sha256"]:
            log({"arm": name, "skipped": "already done"})
        else:
            todo.append((k, rev))
    if not todo:
        return 0
    net = M.load_model(ckpt, villa, manifest, args.device)
    nbytes = vol.shape[0] * h * w
    if nbytes > args.ram_gb * 2 ** 30:
        region_file = out / f"region-{os.getpid()}.u8"
        stack = np.lib.format.open_memmap(region_file, "w+", np.uint8, (vol.shape[0], h, w))
        for r in range(0, h, 1024):
            vol.read(y0 + r, x0, min(1024, h - r), w, out=stack[:, r:r + min(1024, h - r)])
        stack.flush()
    else:
        region_file, stack = None, vol.read(y0, x0, h, w)
    load_s = round(time.monotonic() - t0, 1)
    forward = lambda tiles: M.forward_tiles(net, tiles, args.device, args.precision)  # noqa: E731
    for k, rev in todo:
        name = M.arm_name(k, rev)
        path = out / "maps" / f"{name}.npy"
        window = M.window_for_offset(k, vol.shape[0], args.mesh_plane)
        ta = time.monotonic()
        last = [0.0]

        def progress(done, total):
            if time.monotonic() - last[0] > 30 or done == total:
                last[0] = time.monotonic()
                log({"arm": name, "tiles": done, "of": total,
                     "s_per_tile": round((time.monotonic() - ta) / max(done, 1), 3)})

        values, tiles = M.infer_map(forward, stack, window, reverse=rev, batch=args.batch, stride=args.stride,
                                    progress=progress)
        np.save(path, values)
        secs = time.monotonic() - ta
        write_json(out / "maps" / f"{name}.json", {
            "arm": name, "offset_planes": k, "reversed": rev, "window": list(window), "tiles": tiles,
            "device": args.device, "batch": args.batch, "seconds": round(secs, 1), "s_per_tile": round(secs / tiles, 3),
            "process_load_seconds": load_s, "finite_frac": float(np.isfinite(values).mean()),
            "mean": float(np.nanmean(values)), "sha256": sha256_bytes(path)})
        log({"arm": name, "window": list(window), "seconds": round(secs, 1), "s_per_tile": round(secs / tiles, 3)})
    if region_file is not None and not args.keep_region:
        del stack
        os.remove(region_file)
    return 0


def read_arms(out: Path) -> dict:
    """Per-arm receipts (maps/<arm>.json), so several offsets processes (one per GPU) can share one --out."""
    return {p.stem: json.loads(p.read_text()) for p in sorted((Path(out) / "maps").glob("*.json"))}


# --------------------------------------------------------------------------------------------------------- score
def load_maps(out: Path) -> tuple[dict, dict]:
    receipt = json.loads((out / "maps.json").read_text())
    receipt["arms"] = read_arms(out)
    maps = {}
    for name, rec in receipt["arms"].items():
        path = out / "maps" / f"{name}.npy"
        if sha256_bytes(path) != rec["sha256"]:
            raise SystemExit(f"{path} does not match the hash in its receipt {name}.json")
        maps[name] = np.load(path)
    return maps, receipt


def cmd_score(args):
    _, seg = get_segment(args)
    out = Path(args.out)
    maps, receipt = load_maps(out)
    region = tuple(receipt["config"]["region"])
    if args.region and tuple(args.region) != region:
        raise SystemExit(f"--region {args.region} differs from the maps' region {list(region)}")
    vu = receipt["config"]["voxel_um"]
    z = np.load(out / "depth.npz")
    covered = z["covered"]
    fwd = {M.parse_arm(n)[0]: n for n in maps if not M.parse_arm(n)[1]}
    if 0 not in fwd:
        raise SystemExit("score needs the on-surface map (offset 0)")
    valid = covered & np.isfinite(maps[fwd[0]])
    vals = {n: np.nan_to_num(m, nan=0.0) for n, m in maps.items()}
    rec = {**run_meta(args, seg), "region": list(region), "valid_cells": int(valid.sum()), "arms": {}}
    rec["region_note"] = ("scores describe this region only; labels are the organizers' and may have been used to "
                          "train the model, so no AUC here is a held-out accuracy")
    ref = None
    if not args.no_published and seg.published:
        import tifffile
        y0, x0, h, w = region
        pub = tifffile.imread(seg.published_path(args.data_dir))
        ref = S.published_grid(pub[y0:y0 + h, x0:x0 + w])
        del pub
    labs = None if args.no_labels or not seg.labels else scored_mask(args, seg, region)
    if labs is not None:
        frac, supervised = labs
        pos, neg = S.label_masks(frac, supervised, valid)
        rec["labels"] = {"ink_cells": int(pos.sum()), "papyrus_cells": int(neg.sum()),
                         "scored_mm2": round(float(pos.sum() + neg.sum()) * (4 * vu / 1000) ** 2, 3),
                         "rule": "cell scored when all 16 pixels supervised; ink if >= half its pixels are ink, "
                                 "papyrus if none; mixed cells dropped"}
        if pos.sum() < args.min_cells or neg.sum() < args.min_cells:
            rec["labels"]["warning"] = (f"fewer than {args.min_cells} ink or papyrus cells: AUC differences of a few "
                                        f"hundredths are not resolved")
    for name, v in vals.items():
        k, rev = M.parse_arm(name)
        a = {"offset_planes": k, "offset_um": round(k * vu, 1), "reversed": rev}
        if labs is not None:
            a["auc"] = S.auc(v, pos, neg)
            a["contrast"] = float(v[pos].mean() - v[neg].mean()) if pos.any() and neg.any() else float("nan")
        if ref is not None:
            a["published"] = S.agreement(v, ref, valid)
        rec["arms"][name] = a
    if labs is not None:
        curve = {k: rec["arms"][n]["auc"] for k, n in fwd.items()}
        tol = {"minus": S.tolerance(curve, -1), "plus": S.tolerance(curve, +1)}
        rec["tolerance"] = {"margin": S.MARGIN, "planes": tol, "um": {s: round(t * vu, 1) for s, t in tol.items()},
                            "rule": "furthest tested offset reached before the AUC first falls more than the margin "
                                    "below the on-surface AUC"}
        rec["best_offset_planes"] = max(curve, key=curve.get)
        rec["nulls"] = {"rotated_labels_k0": S.rotated_null(vals[fwd[0]], frac, supervised, valid)}
        if ref is not None:
            rec["nulls"]["published_map_auc"] = S.auc(ref, pos, neg)
        if args.alignment:
            al = S.alignment(vals[fwd[0]], frac, supervised, valid)
            al["cell_um"] = round(4 * vu, 2)
            rec["alignment"] = al
    write_json(out / "score.json", rec)
    log({"score": "done", "auc": {n: round(a.get("auc", float("nan")), 4) for n, a in rec["arms"].items()}})
    return 0


# -------------------------------------------------------------------------------------------------------- report
def cmd_report(args):
    from . import report
    out = Path(args.out)
    rec = report.build(out)
    write_json(out / "report.json", rec)
    report.plot(rec, out / "report.png")
    log({"report": str(out / "report.json"), "png": str(out / "report.png")})
    return 0


# ---------------------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tracecheck", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download (or with --check-only, verify) the inputs")
    add_segment(f)
    add_region(f, required=False)
    f.add_argument("--model", action="store_true", help="also the model checkpoint (~1.5 GB) and network code")
    f.add_argument("--no-labels", action="store_true")
    f.add_argument("--no-published", action="store_true")
    f.add_argument("--check-only", action="store_true", help="download nothing; verify what is on disk")
    f.add_argument("--list", action="store_true", help="list what the bucket holds for this segment, then stop")
    f.add_argument("--workers", type=int, default=8)

    d = sub.add_parser("depth", help="CT-only brightest-layer depth and coverage")
    add_segment(d)
    add_region(d)
    d.add_argument("--out", required=True)
    d.add_argument("--mesh-plane", type=int, default=M.MESH_PLANE)
    d.add_argument("--no-labels", action="store_true")

    o = sub.add_parser("offsets", help="model maps with the depth window moved k planes")
    add_segment(o)
    add_region(o)
    o.add_argument("--out", required=True)
    o.add_argument("--offsets", nargs="+", type=int, default=DEFAULT_OFFSETS, metavar="K",
                   help=f"window offsets in planes (default {DEFAULT_OFFSETS})")
    o.add_argument("--reverse-null", action="store_true", help="also run the on-surface window in reversed order")
    o.add_argument("--device", default="cpu", help="cpu, cuda, xpu, mps ...")
    o.add_argument("--precision", choices=M.PRECISIONS, default="fp32",
                   help="fp32: strict FP32 on every device (default); tf32: allow NVIDIA TensorFloat-32, PyTorch's "
                        "CUDA default; bf16/fp16: autocast, faster, map values change by up to ~0.01")
    o.add_argument("--stride", type=int, default=M.STRIDE, help="tile stride (128: the organizers' blending)")
    o.add_argument("--batch", type=int, default=4)
    o.add_argument("--threads", type=int, default=0)
    o.add_argument("--checkpoint")
    o.add_argument("--villa-dir")
    o.add_argument("--mesh-plane", type=int, default=M.MESH_PLANE)
    o.add_argument("--ram-gb", type=float, default=4.0, help="larger regions are staged in a file under --out")
    o.add_argument("--keep-region", action="store_true")

    s = sub.add_parser("score", help="label AUC and published-map agreement per offset")
    add_segment(s)
    add_region(s, required=False)
    s.add_argument("--out", required=True)
    s.add_argument("--alignment", action="store_true", help="also the in-plane label-shift diagnostic")
    s.add_argument("--no-labels", action="store_true")
    s.add_argument("--no-published", action="store_true")
    s.add_argument("--min-cells", type=int, default=20000)

    r = sub.add_parser("report", help="report.json and report.png from a run directory")
    r.add_argument("--out", required=True)

    args = ap.parse_args(argv)
    return {"fetch": cmd_fetch, "depth": cmd_depth, "offsets": cmd_offsets, "score": cmd_score,
            "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
