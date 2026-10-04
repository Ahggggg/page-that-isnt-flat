"""End to end on a synthetic segment: depth -> offsets -> score -> report through the command line, with a stand-in
network that only 'sees ink' near the middle of its window. Checks the wiring, not the real model."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tracecheck import cli  # noqa: E402
from tracecheck import model as M  # noqa: E402
from test_blosc_labels import write_sharded  # noqa: E402
from test_depth_volume import write_zarr2  # noqa: E402

PREFIX = "TEST/segments/s1/"


def fake_forward(model, tiles, device, precision="fp32"):
    """'Ink' = brightness of the window's middle planes (window index 27..35, around the mesh plane at 31)."""
    mid = tiles[:, 27:36].astype(np.float32).mean(axis=1) / 255.0
    return mid.reshape(len(tiles), 64, 4, 64, 4).mean(axis=(2, 4))


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.data, cls.rundir = root / "data", root / "run"
        rng = np.random.default_rng(13)
        H = W = 512
        vol = rng.integers(0, 30, (109, H, W)).astype(np.uint8)
        vol[50:59] += 60                                   # the papyrus sheet around the mesh plane (54)
        ink = np.zeros((H, W), np.uint8)
        for y in range(40, H - 40, 64):                    # 'letters': bars of ink
            ink[y:y + 24, 40:W - 40] = 255
        vol[52:57][:, ink > 0] += 120                      # ink brightens the sheet's middle planes
        seg = cls.data / PREFIX
        write_zarr2(seg / "surface-volumes/v.zarr/0", vol)
        (seg / "surface-volumes/v.zarr/.zattrs").write_text(json.dumps({"multiscales": [{"datasets": [
            {"coordinateTransformations": [{"type": "scale", "scale": [2.399, 2.399, 2.399]}], "path": "0"}]}]}))
        sup = np.zeros((H, W), np.uint8)
        sup[16:H - 16, 16:W - 16] = 255
        write_sharded(seg / "ink-labels/v1/inklabels.zarr", ink)
        write_sharded(seg / "ink-labels/v1/supervision.zarr", sup)
        tifffile.imwrite(seg / "pred.tif", (np.clip(ink * 0.8 + rng.integers(0, 40, (H, W)), 0, 255)).astype(np.uint8))
        manifest = json.loads((Path(cli.sources.MANIFEST)).read_text())
        manifest["segments"] = {"s1": {"scroll": "TEST", "id": "s1", "surface": "surface-volumes/v.zarr",
                                       "labels": "ink-labels/v1", "published": "pred.tif"}}
        cls.manifest = root / "manifest.json"
        cls.manifest.write_text(json.dumps(manifest))
        common = ["--manifest", str(cls.manifest), "--segment", "s1", "--data-dir", str(cls.data),
                  "--region", "0", "0", str(H), str(W), "--out", str(cls.rundir)]
        saved = M.load_model, M.forward_tiles
        M.load_model, M.forward_tiles = (lambda *a, **k: None), fake_forward
        try:
            assert cli.main(["depth"] + common) == 0
            assert cli.main(["offsets"] + common + ["--offsets", "-16", "-4", "0", "4", "16", "--reverse-null",
                                                     "--batch", "3"]) == 0
            assert cli.main(["offsets"] + common + ["--offsets", "0"]) == 0          # resumable: skips k+0
            try:                                                                     # refuses mixed settings
                cli.main(["offsets"] + common + ["--offsets", "8", "--stride", "256"])
                raise AssertionError("mixed strides accepted")
            except SystemExit:
                pass
            assert cli.main(["score"] + common + ["--alignment", "--min-cells", "100"]) == 0
            assert cli.main(["report", "--out", str(cls.rundir)]) == 0
        finally:
            M.load_model, M.forward_tiles = saved

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_depth_finds_the_sheet(self):
        rec = json.loads((self.rundir / "depth.json").read_text())
        self.assertLessEqual(abs(rec["all_measured"]["median_planes"]), 2)
        self.assertIn("label_scored", rec)
        self.assertLess(rec["all_measured"]["frac_more_than_10_planes_off"], 0.05)

    def test_maps_and_receipt(self):
        arms = cli.read_arms(self.rundir)
        self.assertEqual(sorted(arms), sorted(["k-16", "k-4", "k+0", "k+4", "k+16", "R+0"]))
        self.assertEqual(arms["k-16"]["window"], [7, 69])
        self.assertEqual(json.loads((self.rundir / "maps.json").read_text())["config"]["stride"], 128)
        self.assertEqual(np.load(self.rundir / "maps" / "k+0.npy").shape, (128, 128))

    def test_scores_follow_the_window(self):
        s = json.loads((self.rundir / "score.json").read_text())
        auc = {n: a["auc"] for n, a in s["arms"].items()}
        self.assertGreater(auc["k+0"], 0.95)
        self.assertLess(auc["k+16"], auc["k+0"] - 0.2)
        self.assertLess(auc["k-16"], auc["k+0"] - 0.2)
        self.assertEqual(s["tolerance"]["planes"], {"minus": -4, "plus": 4})
        self.assertAlmostEqual(s["arms"]["k+4"]["offset_um"], 9.6)
        self.assertEqual(s["alignment"]["best_shift"], "0,0")
        self.assertLess(s["nulls"]["rotated_labels_k0"]["auc"], auc["k+0"] - 0.2)
        self.assertGreater(s["arms"]["k+0"]["published"]["r_raw"], 0.5)

    def test_report(self):
        rec = json.loads((self.rundir / "report.json").read_text())
        self.assertEqual(len(rec["curve"]), 6)
        with Image.open(self.rundir / "report.png") as im:
            self.assertEqual(im.size, (1200, 1000))


if __name__ == "__main__":
    unittest.main()
