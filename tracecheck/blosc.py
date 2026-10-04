"""A small reader for Blosc (version 1 format) frames, enough to decode the organizers' label arrays without the
numcodecs/blosc stack: the frame layout and the byte/bit shuffles are done here with numpy; the inner compressor is
zstd (from the standard library on Python 3.14+, else the `zstandard` or `numcodecs` package) or zlib.

Frame layout (c-blosc README_HEADER): 16-byte header
    version u8 | versionlz u8 | flags u8 | typesize u8 | nbytes u32 | blocksize u32 | cbytes u32   (little-endian)
flags: 0x01 byte shuffle, 0x02 stored uncompressed (memcpy), 0x04 bit shuffle, 0x10 blocks not split,
bits 5-7 the compressor (0 blosclz, 1 lz4, 2 snappy, 3 zlib, 4 zstd). Unless memcpy'd, the header is followed by one
int32 start offset per block; each block holds one stream per split, each an int32 size then the data, stored raw
when that size equals the split's uncompressed size.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

MAX_SPLITS = 16
MIN_BUFFERSIZE = 128
CODECS = {0: "blosclz", 1: "lz4", 2: "snappy", 3: "zlib", 4: "zstd"}


# ------------------------------------------------------------------------------------------------ inner codecs
def zstd_backend():
    """(name, decompress(data, size) -> bytes) for the first zstd implementation available."""
    try:
        from compression import zstd as _std  # Python 3.14+
        return "compression.zstd", lambda data, size: _std.decompress(data)
    except ImportError:
        pass
    try:
        import zstandard
        dec = zstandard.ZstdDecompressor()
        return "zstandard", lambda data, size: dec.decompress(data, max_output_size=size)
    except ImportError:
        pass
    try:
        import numcodecs
        codec = numcodecs.Zstd()
        return "numcodecs", lambda data, size: bytes(codec.decode(data))
    except ImportError:
        pass
    return None, None


def _inner(code: int, data: bytes, size: int) -> bytes:
    if code == 4:
        name, fn = zstd_backend()
        if fn is None:
            raise RuntimeError("these arrays are zstd-compressed and no zstd decoder is available: use Python 3.14+ "
                               "or `pip install zstandard`")
        out = fn(data, size)
    elif code == 3:
        out = zlib.decompress(data)
    else:
        raise NotImplementedError(f"blosc inner codec {CODECS.get(code, code)!r} is not supported here")
    if len(out) != size:
        raise ValueError(f"blosc stream decoded to {len(out)} bytes, expected {size}")
    return out


# ---------------------------------------------------------------------------------------------------- shuffles
def byte_unshuffle(buf: bytes, typesize: int) -> bytes:
    """Inverse of the byte shuffle: byte j of element i is stored at j * n + i (n whole elements); tail unchanged."""
    if typesize <= 1:
        return buf
    a = np.frombuffer(buf, np.uint8)
    n = a.size // typesize
    body = a[:n * typesize].reshape(typesize, n).T.reshape(-1)
    return body.tobytes() + a[n * typesize:].tobytes()


def _bit_core(a: np.ndarray, n: int, typesize: int) -> np.ndarray:
    """Inverse bitshuffle of n elements (n % 8 == 0): row (8 j + b) holds bit b of byte j of every element, packed
    least-significant bit first (element i at bit i % 8 of byte i // 8)."""
    bits = np.unpackbits(a.reshape(typesize * 8, n // 8), axis=1, bitorder="little")    # (8 t, n)
    bits = bits.reshape(typesize, 8, n).transpose(2, 0, 1)                                # (n, t, 8)
    return np.packbits(bits, axis=2, bitorder="little").reshape(-1)


def bit_unshuffle(buf: bytes, typesize: int, version: int = 2) -> bytes:
    a = np.frombuffer(buf, np.uint8)
    n = a.size // typesize
    if version == 2:                           # format 2: whole block bitshuffled only when n is a multiple of 8
        if n % 8:
            return buf
        return _bit_core(a[:n * typesize], n, typesize).tobytes() + a[n * typesize:].tobytes()
    m = n - n % 8                              # format 3+: leading multiple of 8 elements, the rest copied
    return _bit_core(a[:m * typesize], m, typesize).tobytes() + a[m * typesize:].tobytes()


def bit_shuffle(buf: bytes, typesize: int) -> bytes:
    """Forward bitshuffle (n % 8 == 0 elements): for tests and for building frames."""
    a = np.frombuffer(buf, np.uint8)
    n = a.size // typesize
    assert n % 8 == 0 and a.size == n * typesize
    bits = np.unpackbits(a.reshape(n, typesize, 1), axis=2, bitorder="little")           # (n, t, 8)
    rows = bits.transpose(1, 2, 0).reshape(typesize * 8, n)
    return np.packbits(rows, axis=1, bitorder="little").tobytes()


# ------------------------------------------------------------------------------------------------------- frames
def header(frame: bytes) -> dict:
    if len(frame) < 16:
        raise ValueError("not a blosc frame: shorter than its 16-byte header")
    version, versionlz, flags, typesize, nbytes, blocksize, cbytes = struct.unpack("<BBBBiii", frame[:16])
    return {"version": version, "versionlz": versionlz, "flags": flags, "typesize": typesize, "nbytes": nbytes,
            "blocksize": blocksize, "cbytes": cbytes, "codec": flags >> 5}


def decompress(frame: bytes) -> bytes:
    """Decode one Blosc-1 frame to its original bytes."""
    h = header(frame)
    if h["version"] not in (1, 2, 3):
        raise NotImplementedError(f"blosc format version {h['version']} is not supported (Blosc2 frames differ)")
    if h["cbytes"] != len(frame):
        raise ValueError(f"blosc frame says {h['cbytes']} bytes, got {len(frame)}")
    nbytes, bsize, typesize, flags = h["nbytes"], h["blocksize"], h["typesize"], h["flags"]
    if flags & 0x02:                                          # stored as a plain copy
        out = frame[16:16 + nbytes]
        if len(out) != nbytes:
            raise ValueError("truncated memcpy'd blosc frame")
        return out
    if nbytes == 0:
        return b""
    nblocks = -(-nbytes // bsize)
    starts = struct.unpack(f"<{nblocks}i", frame[16:16 + 4 * nblocks])
    dont_split = bool(flags & 0x10)
    parts = []
    for b, start in enumerate(starts):
        leftover = b == nblocks - 1 and nbytes % bsize != 0
        size = nbytes % bsize if leftover else bsize
        split = (not dont_split and not leftover and typesize <= MAX_SPLITS and size // typesize >= MIN_BUFFERSIZE)
        nsplits = typesize if split else 1
        neblock = size // nsplits
        pos, block = start, []
        for _ in range(nsplits):
            (csize,) = struct.unpack("<i", frame[pos:pos + 4])
            pos += 4
            data = frame[pos:pos + csize]
            pos += csize
            block.append(data if csize == neblock else _inner(h["codec"], data, neblock))
        raw = b"".join(block)
        if flags & 0x01:
            raw = byte_unshuffle(raw, typesize)
        elif flags & 0x04:
            raw = bit_unshuffle(raw, typesize, h["version"])
        parts.append(raw)
    out = b"".join(parts)
    if len(out) != nbytes:
        raise ValueError(f"blosc frame decoded to {len(out)} bytes, expected {nbytes}")
    return out


def frame_uncompressed(data: bytes, typesize: int = 1, blocksize: int = 4096, shuffle: str = "bit") -> bytes:
    """Build a valid Blosc-1 frame whose streams are stored raw (no inner compressor): for tests."""
    nbytes = len(data)
    flags = (0x04 if shuffle == "bit" else 0x01 if shuffle == "byte" else 0) | 0x10 | (4 << 5)
    nblocks = -(-nbytes // blocksize)
    body, starts = [], []
    offset = 16 + 4 * nblocks
    for b in range(nblocks):
        chunk = data[b * blocksize:(b + 1) * blocksize]
        if shuffle == "bit" and (len(chunk) // typesize) % 8 == 0:
            chunk = bit_shuffle(chunk, typesize)
        elif shuffle == "byte" and typesize > 1:
            a = np.frombuffer(chunk, np.uint8)
            n = a.size // typesize
            chunk = a[:n * typesize].reshape(n, typesize).T.reshape(-1).tobytes() + a[n * typesize:].tobytes()
        starts.append(offset)
        body.append(struct.pack("<i", len(chunk)) + chunk)
        offset += 4 + len(chunk)
    payload = struct.pack(f"<{nblocks}i", *starts) + b"".join(body)
    return struct.pack("<BBBBiii", 2, 1, flags, typesize, nbytes, blocksize, 16 + len(payload)) + payload
