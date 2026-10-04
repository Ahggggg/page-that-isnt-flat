"""Fetch the film's inputs that tracecheck does not: w035's traced mesh, and PHerc0139's CT at two coarse pyramid levels
for the opening shot. Downloads are checked like tracecheck's (byte count and the bucket's MD5), then assembled.

  python film-making/fetch_film_inputs.py            # about 1.9 GB of CT chunks plus the 14 MB mesh, into film-inputs/

Outputs (see filmdata.py): film-inputs/w035.tifxyz/, film-inputs/scroll_L5.npy (2405 x 829 x 829, 76.8 um voxels) and
film-inputs/scroll_L4_z9.npy (128 x 1657 x 1657, 38.4 um, the slices around w035's mid-height). Vesuvius Challenge public
data, CC BY-NC 4.0.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import filmdata as F
from tracecheck import sources

MESH_KEY = ("PHerc0139/segments/20260317000000-w035_2026031718/mesh/"
            "20260317000000-on-20260102150214-2.399um.tifxyz/")
CT = "PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr/"
CH = 128


def fetch(keys, workers=16):
    raw = F.INPUTS / "raw"

    def one(item):
        key, size = item
        dest = raw / key
        if dest.exists() and dest.stat().st_size == size:
            return 0
        sources.download(sources.S3 + key, dest)
        return size
    with ThreadPoolExecutor(workers) as pool:
        got = sum(pool.map(one, keys))
    print(json.dumps({"files": len(keys), "bytes_fetched": got}), flush=True)
    return raw


def level_meta(level):
    raw = fetch([(k, s) for k, s in sources.list_keys(f"{CT}{level}/.zarray")])
    meta = json.loads((raw / f"{CT}{level}/.zarray").read_text())
    assert meta["compressor"] is None and meta["dtype"] == "|u1" and tuple(meta["chunks"]) == (CH, CH, CH), meta
    return tuple(meta["shape"])


def assemble(level, shape, z_chunk=None):
    """Raw C-order uint8 chunks (no compressor) into one array; z_chunk keeps only that 128-slice layer."""
    prefix = f"{CT}{level}/" + (f"{z_chunk}/" if z_chunk is not None else "")
    keys = [(k, s) for k, s in sources.list_keys(prefix) if not k.endswith((".zarray", ".zattrs"))]
    raw = fetch(keys)
    depth = CH if z_chunk is not None else shape[0]
    out = np.zeros((depth,) + shape[1:], np.uint8)
    for key, _ in keys:
        cz, cy, cx = (int(v) for v in key[len(f"{CT}{level}/"):].split("/"))
        c = np.fromfile(raw / key, np.uint8).reshape(CH, CH, CH)
        z = 0 if z_chunk is not None else cz * CH
        y, x = cy * CH, cx * CH
        out[z:z + CH, y:y + CH, x:x + CH] = c[:depth - z, :shape[1] - y, :shape[2] - x]
    return out


def main():
    F.INPUTS.mkdir(parents=True, exist_ok=True)
    raw = fetch(sources.list_keys(MESH_KEY))
    mesh = F.MESH
    mesh.mkdir(parents=True, exist_ok=True)
    for name in ("meta.json", "x.tif", "y.tif", "z.tif"):
        (mesh / name).write_bytes((raw / MESH_KEY / name).read_bytes())
    np.save(F.SCROLL_L5, assemble(5, level_meta(5)))
    np.save(F.SCROLL_L4_Z9, assemble(4, level_meta(4), z_chunk=9))
    print(json.dumps({"mesh": str(mesh), "scroll_L5": str(F.SCROLL_L5), "scroll_L4_z9": str(F.SCROLL_L4_Z9)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
