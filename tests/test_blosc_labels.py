"""Blosc frames, bit/byte shuffles and the zarr v3 sharded label reader, on synthetic data (no network, no files
from the bucket). Cross-checks against numcodecs run only when it is installed."""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracecheck import blosc, labels  # noqa: E402

try:
    import numcodecs
except ImportError:
    numcodecs = None


def reference_bitshuffle(data: bytes, typesize: int) -> bytes:
    """Bit (8 j + b) of row r: bit b of byte j of element i lands at bit position ((8 j + b) n + i)."""
    a = np.frombuffer(data, np.uint8)
    n = a.size // typesize
    out = np.zeros(a.size, np.uint8)
    for i in range(n):
        for j in range(typesize):
            v = int(a[i * typesize + j])
            for b in range(8):
                if v >> b & 1:
                    idx = (8 * j + b) * n + i
                    out[idx // 8] |= 1 << (idx % 8)
    return out.tobytes()


class ShuffleTests(unittest.TestCase):
    def test_bitshuffle_matches_bitwise_reference(self):
        rng = np.random.default_rng(1)
        for ts in (1, 2, 4):
            data = rng.integers(0, 256, 64 * ts, dtype=np.uint8).tobytes()
            self.assertEqual(blosc.bit_shuffle(data, ts), reference_bitshuffle(data, ts))
            self.assertEqual(blosc.bit_unshuffle(blosc.bit_shuffle(data, ts), ts), data)

    def test_format2_leaves_non_multiple_of_8_blocks_alone(self):
        data = bytes(range(12))
        self.assertEqual(blosc.bit_unshuffle(data, 1, version=2), data)

    def test_byte_unshuffle(self):
        data = np.arange(10, dtype=np.uint16).tobytes()
        shuffled = np.frombuffer(data, np.uint8).reshape(10, 2).T.reshape(-1).tobytes()
        self.assertEqual(blosc.byte_unshuffle(shuffled, 2), data)


class FrameTests(unittest.TestCase):
    def test_round_trip_uncompressed_frames(self):
        rng = np.random.default_rng(2)
        for ts in (1, 2, 4):
            for shuffle in ("bit", "byte", "none"):
                for n in (4096 * ts, 10000 * ts, 64 * ts):
                    data = rng.integers(0, 3, n, dtype=np.uint8).tobytes()
                    frame = blosc.frame_uncompressed(data, ts, 4096, shuffle)
                    self.assertEqual(blosc.decompress(frame), data, (ts, shuffle, n))

    def test_memcpy_frame(self):
        data = b"hello blosc"
        frame = struct.pack("<BBBBiii", 2, 1, 0x02, 1, len(data), len(data), 16 + len(data)) + data
        self.assertEqual(blosc.decompress(frame), data)

    def test_wrong_length_is_rejected(self):
        frame = blosc.frame_uncompressed(bytes(5000), 1, 4096)
        with self.assertRaises(ValueError):
            blosc.decompress(frame[:-1])

    @unittest.skipIf(numcodecs is None, "numcodecs not installed")
    def test_matches_numcodecs(self):
        rng = np.random.default_rng(3)
        for dtype in (np.uint8, np.uint16, np.uint32):
            for shuffle in (numcodecs.Blosc.NOSHUFFLE, numcodecs.Blosc.SHUFFLE, numcodecs.Blosc.BITSHUFFLE):
                for cname in ("zstd", "zlib"):
                    a = rng.integers(0, 4, 70001).astype(dtype)
                    frame = numcodecs.Blosc(cname=cname, clevel=3, shuffle=shuffle).encode(a)
                    self.assertEqual(blosc.decompress(bytes(frame)), a.tobytes(), (dtype, shuffle, cname))


# ------------------------------------------------------------------------------------------- synthetic zarr v3
def write_sharded(root: Path, arr: np.ndarray, shard=(512, 512), inner=(128, 128), empty=(), compress=False):
    """A zarr v3 array laid out as the organizers' labels: sharding_indexed, inner bytes + blosc, index at end
    with crc32c. Inner chunks listed in `empty` (shard-local index) are left out of the shard."""
    level = root / "0"
    level.mkdir(parents=True, exist_ok=True)
    meta = {"shape": list(arr.shape), "data_type": "uint8", "fill_value": 0, "zarr_format": 3, "node_type": "array",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(shard)}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "codecs": [{"name": "sharding_indexed", "configuration": {
                "chunk_shape": list(inner),
                "codecs": [{"name": "bytes"}, {"name": "blosc", "configuration": {"typesize": 1, "cname": "zstd",
                                                                                 "shuffle": "bitshuffle"}}],
                "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}],
                "index_location": "end"}}]}
    (level / "zarr.json").write_text(json.dumps(meta))
    H, W = arr.shape
    for sy in range(-(-H // shard[0])):
        for sx in range(-(-W // shard[1])):
            body, index = b"", []
            gy, gx = shard[0] // inner[0], shard[1] // inner[1]
            for i in range(gy * gx):
                cy, cx = divmod(i, gx)
                y, x = sy * shard[0] + cy * inner[0], sx * shard[1] + cx * inner[1]
                if i in empty or y >= H or x >= W:
                    index.append((2 ** 64 - 1, 2 ** 64 - 1))
                    continue
                block = np.zeros(inner, np.uint8)
                part = arr[y:y + inner[0], x:x + inner[1]]
                block[:part.shape[0], :part.shape[1]] = part
                if compress:
                    frame = bytes(numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)
                                  .encode(block))
                else:
                    frame = blosc.frame_uncompressed(block.tobytes(), 1, 8192, "bit")
                index.append((len(body), len(frame)))
                body += frame
            idx = b"".join(struct.pack("<QQ", a, b) for a, b in index)
            path = level / "c" / str(sy) / str(sx)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body + idx + struct.pack("<I", labels.crc32c(idx)))


class LabelReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "inklabels.zarr"
        rng = np.random.default_rng(4)
        self.arr = (rng.random((700, 1100)) < 0.3).astype(np.uint8) * 255

    def tearDown(self):
        self.tmp.cleanup()

    def test_crc32c_known_vector(self):
        self.assertEqual(labels.crc32c(b"123456789"), 0xE3069283)

    def test_reads_whole_array_and_regions_across_shards(self):
        write_sharded(self.root, self.arr)
        np.testing.assert_array_equal(labels.read_region(self.root, 0, 0, 700, 1100), self.arr)
        for y0, x0, h, w in ((0, 0, 1, 1), (500, 400, 200, 300), (130, 509, 7, 9), (699, 1099, 1, 1)):
            np.testing.assert_array_equal(labels.read_region(self.root, y0, x0, h, w),
                                          self.arr[y0:y0 + h, x0:x0 + w])

    def test_empty_inner_chunk_reads_fill(self):
        write_sharded(self.root, self.arr, empty={0})
        got = labels.read_region(self.root, 0, 0, 256, 256)
        self.assertTrue((got[:128, :128] == 0).all())
        np.testing.assert_array_equal(got[128:, :], self.arr[128:256, :256])

    def test_corrupt_index_raises(self):
        write_sharded(self.root, self.arr)
        path = self.root / "0" / "c" / "0" / "0"
        raw = bytearray(path.read_bytes())
        raw[-10] ^= 0xFF
        path.write_bytes(bytes(raw))
        with self.assertRaises(ValueError):
            labels.read_region(self.root, 0, 0, 10, 10)

    def test_missing_shard_raises_unless_recorded_absent(self):
        write_sharded(self.root, self.arr)
        (self.root / "0" / "c" / "0" / "1").unlink()
        with self.assertRaises(FileNotFoundError):
            labels.read_region(self.root, 0, 600, 10, 10)
        (self.root / "0" / "absent.json").write_text(json.dumps(["c/0/1"]))
        self.assertTrue((labels.read_region(self.root, 0, 600, 10, 10) == 0).all())

    def test_region_outside_raises(self):
        write_sharded(self.root, self.arr)
        with self.assertRaises(ValueError):
            labels.read_region(self.root, 690, 0, 20, 10)

    @unittest.skipIf(numcodecs is None, "numcodecs not installed")
    def test_zstd_compressed_shards(self):
        write_sharded(self.root, self.arr, compress=True)
        np.testing.assert_array_equal(labels.read_region(self.root, 0, 0, 700, 1100), self.arr)


class LabelGridTests(unittest.TestCase):
    def test_cell_rules(self):
        ink = np.zeros((8, 8), np.uint8)
        sup = np.full((8, 8), 255, np.uint8)
        ink[0:4, 0:4] = 255           # cell (0, 0): all ink
        ink[0:2, 4:8] = 255           # cell (0, 1): half ink -> scored ink
        ink[4, 4] = 255               # cell (1, 1): 1/16 ink -> mixed, not scored
        sup[7, 0] = 0                 # cell (1, 0): one unsupervised pixel -> not scored
        frac, supervised = labels.label_grid(ink, sup)
        np.testing.assert_allclose(frac, [[1.0, 0.5], [0.0, 1 / 16]])
        np.testing.assert_array_equal(supervised, [[True, True], [False, True]])


if __name__ == "__main__":
    unittest.main()
