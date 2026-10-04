"""Surface-volume reading and the CT-only sheet depth, on synthetic volumes."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracecheck import depth as D  # noqa: E402
from tracecheck.volume import SurfaceVolume  # noqa: E402


def write_zarr2(level: Path, arr: np.ndarray, chunks=(109, 128, 128), sep="/", skip=()):
    level.mkdir(parents=True, exist_ok=True)
    meta = {"zarr_format": 2, "shape": list(arr.shape), "chunks": list(chunks), "dtype": "|u1", "compressor": None,
            "fill_value": 0, "order": "C", "filters": None, "dimension_separator": sep}
    (level / ".zarray").write_text(json.dumps(meta))
    cz, cy, cx = chunks
    for z in range(-(-arr.shape[0] // cz)):
        for a in range(-(-arr.shape[1] // cy)):
            for b in range(-(-arr.shape[2] // cx)):
                key = sep.join(map(str, (z, a, b)))
                if key in skip:
                    continue
                block = np.zeros(chunks, np.uint8)
                part = arr[z * cz:(z + 1) * cz, a * cy:(a + 1) * cy, b * cx:(b + 1) * cx]
                block[:part.shape[0], :part.shape[1], :part.shape[2]] = part
                path = level / key
                path.parent.mkdir(parents=True, exist_ok=True)
                block.tofile(path)


class VolumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.arr = np.random.default_rng(5).integers(0, 255, (9, 300, 260), dtype=np.uint8)

    def tearDown(self):
        self.tmp.cleanup()

    def test_regions_both_separators(self):
        for sep in ("/", "."):
            level = Path(self.tmp.name) / f"v{sep == '.'}"
            write_zarr2(level, self.arr, (9, 128, 128), sep)
            vol = SurfaceVolume(level)
            for y0, x0, h, w in ((0, 0, 300, 260), (100, 120, 64, 100), (299, 259, 1, 1)):
                np.testing.assert_array_equal(vol.read(y0, x0, h, w), self.arr[:, y0:y0 + h, x0:x0 + w])

    def test_missing_chunk_raises_absent_chunk_reads_zero(self):
        level = Path(self.tmp.name) / "v"
        write_zarr2(level, self.arr, (9, 128, 128), "/", skip={"0/1/1"})
        vol = SurfaceVolume(level)
        with self.assertRaises(FileNotFoundError):
            vol.read(130, 130, 10, 10)
        (level / "absent.json").write_text(json.dumps(["0/1/1"]))
        vol = SurfaceVolume(level)
        self.assertTrue((vol.read(130, 130, 10, 10) == 0).all())
        np.testing.assert_array_equal(vol.read(0, 0, 128, 128), self.arr[:, :128, :128])

    def test_region_outside_raises(self):
        level = Path(self.tmp.name) / "v"
        write_zarr2(level, self.arr, (9, 128, 128))
        with self.assertRaises(ValueError):
            SurfaceVolume(level).read(290, 0, 20, 10)

    def test_compressed_volume_refused(self):
        level = Path(self.tmp.name) / "v"
        level.mkdir()
        (level / ".zarray").write_text(json.dumps({"zarr_format": 2, "shape": [1, 1, 1], "chunks": [1, 1, 1],
                                                   "dtype": "|u1", "compressor": {"id": "blosc"}}))
        with self.assertRaises(ValueError):
            SurfaceVolume(level)


def layered(depth_cells: np.ndarray, planes=109, mesh=54, bright=150, noise=20, seed=6):
    """A volume whose brightest layer (3 planes thick) sits depth_cells planes off the mesh plane, per 4x4 cell."""
    gh, gw = depth_cells.shape
    rng = np.random.default_rng(seed)
    vol = rng.integers(0, noise, (planes, gh * 4, gw * 4)).astype(np.uint8)
    d = np.repeat(np.repeat(depth_cells, 4, 0), 4, 1)
    yy, xx = np.indices(d.shape)
    for t in (-1, 0, 1):
        vol[mesh + d + t, yy, xx] = bright
    return vol


class DepthTests(unittest.TestCase):
    def test_recovers_piecewise_constant_depth(self):
        cells = np.zeros((96, 96), int)
        cells[:, 48:] = 12
        cells[48:, :48] = -25
        vol = layered(cells)
        d, peak = D.depth_from_reduced(D.reduce_stack(vol))
        inner = np.zeros_like(cells, bool)
        for (a, b), (c, e) in (((0, 36), (0, 36)), ((0, 36), (60, 96)), ((60, 96), (0, 36))):
            inner[a:b, c:e] = True
        np.testing.assert_array_equal(d[inner], cells[inner])
        self.assertTrue((peak[inner] > D.PEAK_MIN).all())

    def test_prior_prefers_the_layer_nearer_the_trace(self):
        cells = np.full((40, 40), 5)
        vol = layered(cells)
        vol[54 + 40 - 1:54 + 40 + 2] = 150            # an equally bright layer 40 planes off
        d, _ = D.depth_from_reduced(D.reduce_stack(vol))
        self.assertTrue((d[10:30, 10:30] == 5).all())

    def test_strips_equal_whole_region(self):
        rng = np.random.default_rng(7)
        vol = rng.integers(0, 255, (109, 4 * 70, 4 * 50), dtype=np.uint8)
        vol[54] = np.maximum(vol[54], 1)
        vol[54, :8, :8] = 0                                     # 2 x 2 uncovered cells
        whole_d, whole_p = D.depth_from_reduced(D.reduce_stack(vol))
        d, p, cov = D.sheet_depth(lambda r0, r1: vol[:, 4 * r0:4 * r1], 70, 50, 109, block_rows=8, strip_rows=16)
        np.testing.assert_array_equal(d, whole_d)
        np.testing.assert_array_equal(p, whole_p)
        self.assertFalse(cov[:2, :2].any())
        self.assertEqual(int((~cov).sum()), 4)

    def test_coverage_shares(self):
        d = np.array([[0, 11, -21, 31], [-32, 5, 30, 0]], np.float32)
        peak = np.full(d.shape, 100.0, np.float32)
        cov = np.ones(d.shape, bool)
        peak[0, 0] = 10                                           # unmeasured: peak too low
        rec = D.coverage(d, peak, cov, None, 2.4)
        self.assertEqual(rec["cells"], 7)
        self.assertAlmostEqual(rec["frac_more_than_10_planes_off"], 5 / 7)
        self.assertAlmostEqual(rec["frac_more_than_20_planes_off"], 4 / 7)
        self.assertAlmostEqual(rec["frac_outside_centred_window"], 2 / 7)   # 31 and -32 fall outside -31..30
        self.assertEqual(sum(rec["histogram"]["counts"]), 7)


if __name__ == "__main__":
    unittest.main()
