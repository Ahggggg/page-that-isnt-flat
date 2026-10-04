"""Read a 2-D region of a zarr v3 array as the organizers publish their ink labels: `sharding_indexed` shards whose
inner chunks are `bytes` + `blosc` (or plain `bytes`), shard index at the end with a crc32c checksum.

Label pixel (y, x) is surface-volume pixel (y, x): the label canvas has the surface volume's height and width. The
label zarr also carries a `canvas_offset_yx` attribute; this reader does not apply it (see REPRODUCE.md, pitfalls).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import blosc

EMPTY = 2 ** 64 - 1
_CRC_TABLE = None


def crc32c(data: bytes) -> int:
    global _CRC_TABLE
    if _CRC_TABLE is None:
        table = []
        for i in range(256):
            c = i
            for _ in range(8):
                c = (c >> 1) ^ 0x82F63B78 if c & 1 else c >> 1
            table.append(c)
        _CRC_TABLE = table
    crc = 0xFFFFFFFF
    for b in data:
        crc = _CRC_TABLE[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def array_meta(array_root: Path, level: str = "0") -> dict:
    return json.loads((Path(array_root) / level / "zarr.json").read_text(encoding="utf-8"))


def shard_keys(meta: dict, y0: int, x0: int, h: int, w: int) -> list[tuple[str, int, int]]:
    """(key relative to the level directory, shard row, shard column) of every shard a region touches."""
    sy, sx = meta["chunk_grid"]["configuration"]["chunk_shape"]
    enc = meta.get("chunk_key_encoding", {"name": "default"})
    sep = enc.get("configuration", {}).get("separator", "/")
    if enc.get("name", "default") != "default":
        raise NotImplementedError(f"chunk key encoding {enc!r} is not supported")
    return [(sep.join(["c", str(a), str(b)]), a, b)
            for a in range(y0 // sy, (y0 + h - 1) // sy + 1) for b in range(x0 // sx, (x0 + w - 1) // sx + 1)]


def _decode_inner(codecs: list, data: bytes, shape, dtype) -> np.ndarray:
    for codec in reversed(codecs):
        name = codec["name"]
        if name == "blosc":
            data = blosc.decompress(data)
        elif name == "bytes":
            endian = codec.get("configuration", {}).get("endian", "little")
            dtype = np.dtype(dtype).newbyteorder("<" if endian == "little" else ">")
        elif name == "zstd":
            _, fn = blosc.zstd_backend()
            if fn is None:
                raise RuntimeError("zstd-compressed array: use Python 3.14+ or `pip install zstandard`")
            data = fn(data, int(np.prod(shape)) * np.dtype(dtype).itemsize)
        else:
            raise NotImplementedError(f"zarr codec {name!r} is not supported")
    return np.frombuffer(data, dtype).reshape(shape)


def read_region(array_root, y0: int, x0: int, h: int, w: int, level: str = "0") -> np.ndarray:
    """(h, w) region of level `level` of a 2-D zarr v3 array. Empty inner chunks read as the fill value; a missing
    shard reads as the fill value only when the fetch step recorded it as absent on the server (absent.json). Any
    other missing shard, or a shard index whose crc32c does not match, raises."""
    root = Path(array_root) / level
    meta = array_meta(array_root, level)
    shape = tuple(meta["shape"])
    if len(shape) != 2:
        raise ValueError(f"{root}: expected a 2-D array, got shape {shape}")
    if y0 < 0 or x0 < 0 or y0 + h > shape[0] or x0 + w > shape[1]:
        raise ValueError(f"region {(y0, x0, h, w)} is outside the array {shape}")
    dtype = np.dtype({"uint8": "u1", "uint16": "u2", "int32": "i4", "float32": "f4"}[meta["data_type"]])
    out = np.full((h, w), meta["fill_value"], dtype)
    sy, sx = meta["chunk_grid"]["configuration"]["chunk_shape"]
    codecs = meta["codecs"]
    sharded = len(codecs) == 1 and codecs[0]["name"] == "sharding_indexed"
    absent = set(json.loads((root / "absent.json").read_text())) if (root / "absent.json").exists() else set()
    for key, a, b in shard_keys(meta, y0, x0, h, w):
        path = root / key
        if not path.exists():
            if key in absent:                           # the server has no such shard: all fill value
                continue
            raise FileNotFoundError(f"{path} is missing (not downloaded?); a shard is read as empty only when "
                                    f"absent.json lists it as absent on the server")
        raw = path.read_bytes()
        if not sharded:
            chunk = _decode_inner(codecs, raw, (sy, sx), dtype)
            _paste(out, chunk, a * sy, b * sx, y0, x0, h, w, shape)
            continue
        cfg = codecs[0]["configuration"]
        iy, ix = cfg["chunk_shape"]
        gy, gx = sy // iy, sx // ix
        n = gy * gx
        index_codecs = [c["name"] for c in cfg.get("index_codecs", [])]
        tail = 4 if "crc32c" in index_codecs else 0
        size = n * 16 + tail
        index_bytes = raw[-size:] if cfg.get("index_location", "end") == "end" else raw[:size]
        if tail:
            body, crc = index_bytes[:-4], int.from_bytes(index_bytes[-4:], "little")
            if crc32c(body) != crc:
                raise ValueError(f"{path}: shard index crc32c mismatch (corrupt or truncated download)")
        else:
            body = index_bytes
        index = np.frombuffer(body, "<u8").reshape(n, 2)
        for i, (off, nb) in enumerate(index):
            cy, cx = divmod(i, gx)
            ty, tx = a * sy + cy * iy, b * sx + cx * ix
            if ty >= y0 + h or tx >= x0 + w or ty + iy <= y0 or tx + ix <= x0 or off == EMPTY:
                continue
            chunk = _decode_inner(cfg["codecs"], raw[int(off):int(off + nb)], (iy, ix), dtype)
            _paste(out, chunk, ty, tx, y0, x0, h, w, shape)
    return out


def _paste(out, chunk, ty, tx, y0, x0, h, w, shape):
    ys, ye = max(ty, y0), min(ty + chunk.shape[0], y0 + h, shape[0])
    xs, xe = max(tx, x0), min(tx + chunk.shape[1], x0 + w, shape[1])
    if ye > ys and xe > xs:
        out[ys - y0:ye - y0, xs - x0:xe - x0] = chunk[ys - ty:ye - ty, xs - tx:xe - tx]


def label_grid(ink: np.ndarray, supervision: np.ndarray, out_scale: int = 4):
    """Labels on the model's output grid (one cell per out_scale x out_scale pixels).

    frac: share of the cell's pixels labelled ink; supervised: every pixel of the cell is inside the supervision
    mask. A cell is scored ink when supervised and frac >= 0.5, papyrus when supervised and frac == 0; mixed cells
    are not scored. Unlabelled pixels are excluded, never counted as papyrus."""
    h, w = ink.shape
    gh, gw = h // out_scale, w // out_scale
    ink = ink[:gh * out_scale, :gw * out_scale]
    sup = supervision[:gh * out_scale, :gw * out_scale]
    frac = (ink > 0).reshape(gh, out_scale, gw, out_scale).mean(axis=(1, 3))
    supervised = (sup > 0).reshape(gh, out_scale, gw, out_scale).min(axis=(1, 3))
    return frac, supervised
