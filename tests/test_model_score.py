"""Depth-window geometry, tiling and blending (with a stand-in network), the checkpoint audit's allow-list, and the
scores. No checkpoint and no network access are needed."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracecheck import model as M  # noqa: E402
from tracecheck import score as S  # noqa: E402

try:
    import torch
except ImportError:
    torch = None


class WindowTests(unittest.TestCase):
    def test_centred_window_and_limits(self):
        self.assertEqual(M.window_for_offset(0), (23, 85))
        self.assertEqual(M.window_for_offset(-23), (0, 62))
        self.assertEqual(M.window_for_offset(24), (47, 109))
        self.assertEqual(M.window_for_offset(-8), (15, 77))
        for bad in (-24, 25):
            with self.assertRaises(ValueError):
                M.window_for_offset(bad)

    def test_arm_names(self):
        for k, rev in ((0, False), (-16, False), (8, False), (0, True)):
            self.assertEqual(M.parse_arm(M.arm_name(k, rev)), (k, rev))
        self.assertEqual(M.arm_name(0), "k+0")
        self.assertEqual(M.arm_name(0, True), "R+0")


class InferTests(unittest.TestCase):
    def test_window_planes_reach_the_network(self):
        stack = np.broadcast_to(np.arange(109, dtype=np.uint8)[:, None, None], (109, 256, 384)).copy()
        first = lambda t: np.full((len(t), 64, 64), t[:, 0].mean(axis=(1, 2)).reshape(-1, 1, 1), np.float32)  # noqa
        for k in (-23, 0, 16):
            out, n = M.infer_map(first, stack, M.window_for_offset(k))
            self.assertEqual(n, 2)
            np.testing.assert_allclose(out, 23 + k, rtol=1e-6)
        out, _ = M.infer_map(first, stack, M.window_for_offset(0), reverse=True)
        np.testing.assert_allclose(out, 84, rtol=1e-6)

    def test_output_cell_maps_to_4x4_pixels(self):
        rng = np.random.default_rng(8)
        stack = np.zeros((109, 512, 640), np.uint8)
        stack[23] = rng.integers(0, 255, (512, 640))

        def blocks(t):
            p = t[:, 0].astype(np.float32)
            return p.reshape(len(t), 64, 4, 64, 4).mean(axis=(2, 4))

        out, n = M.infer_map(blocks, stack, (23, 85), batch=3)
        self.assertEqual(out.shape, (128, 160))
        self.assertEqual(n, len(M.grid(512)) * len(M.grid(640)))
        ref = stack[23].astype(np.float32).reshape(128, 4, 160, 4).mean(axis=(1, 3))
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-3)

    def test_grid_flush_and_coverage(self):
        self.assertEqual(M.grid(256), [0])
        self.assertEqual(M.grid(300), [0, 44])
        self.assertEqual(M.grid(512, stride=256), [0, 256])
        with self.assertRaises(ValueError):
            M.grid(200)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_fp32_is_strict_and_tf32_opt_in(self):
        stub = lambda x: x.mean(dim=(2, 3, 4)).reshape(-1, 1, 1, 1).expand(-1, 1, 64, 64)  # noqa: E731
        tiles = np.zeros((1, 62, 256, 256), np.uint8)
        M.forward_tiles(stub, tiles, "cpu", "tf32")
        self.assertTrue(torch.backends.cudnn.allow_tf32)
        M.forward_tiles(stub, tiles, "cpu", "fp32")
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)

    def test_preprocess_clips(self):
        np.testing.assert_allclose(M.preprocess(np.array([0, 100, 200, 255], np.uint8)), [0, 0.5, 1, 1])


class AuditTests(unittest.TestCase):
    def test_allow_list(self):
        stubs = M.check_globals(["pytorch_lightning.utilities.parsing.AttributeDict"])
        self.assertEqual(stubs[0][1], "pytorch_lightning.utilities.parsing.AttributeDict")
        for bad in ("os.system", "builtins.eval", "subprocess.Popen"):
            with self.assertRaises(ValueError):
                M.check_globals([bad])

    @unittest.skipIf(torch is None, "torch not installed")
    def test_plain_state_dict_has_no_unsafe_globals(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "w.ckpt"
            torch.save({"state_dict": {"a": torch.zeros(3)}}, path)
            self.assertEqual(M.audit_checkpoint(path), [])

    def test_hash_check(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "f"
            path.write_bytes(b"abc")
            M.verify_hash(path, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
            with self.assertRaises(ValueError):
                M.verify_hash(path, "0" * 64)


class ScoreTests(unittest.TestCase):
    def test_auc_matches_pairwise_count_with_ties(self):
        rng = np.random.default_rng(9)
        v = rng.integers(0, 5, (30, 30)).astype(np.float32)
        pos = rng.random((30, 30)) < 0.3
        neg = ~pos & (rng.random((30, 30)) < 0.8)
        a, b = v[pos], v[neg]
        brute = ((a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()) / (a.size * b.size)
        self.assertAlmostEqual(S.auc(v, pos, neg), brute, places=12)

    def test_tolerance(self):
        curve = {-16: 0.80, -8: 0.88, -4: 0.90, 0: 0.90, 4: 0.89, 8: 0.86, 16: 0.84}
        self.assertEqual(S.tolerance(curve, -1), -8)        # -16 falls 0.10 below
        self.assertEqual(S.tolerance(curve, +1), 8)         # +16 falls 0.06 below
        curve[16] = 0.86
        self.assertEqual(S.tolerance(curve, +1), 16)
        curve[4] = 0.80
        self.assertEqual(S.tolerance(curve, +1), 0)

    def test_alignment_finds_a_known_shift(self):
        rng = np.random.default_rng(10)
        frac = (rng.random((80, 80)) < 0.4).astype(float)
        sup = np.ones((80, 80), bool)
        values = S.shift(frac, 2, 4) + 0.01 * rng.random((80, 80))
        al = S.alignment(values, frac, sup, np.ones((80, 80), bool))
        self.assertEqual(al["best_shift"], "2,4")
        self.assertGreater(al["best_minus_zero"], 0.3)

    def test_rotated_null_is_near_chance_on_unstructured_labels(self):
        rng = np.random.default_rng(11)
        frac = (rng.random((100, 100)) < 0.4).astype(float)
        values = frac + 0.1 * rng.random((100, 100))
        ok = np.ones((100, 100), bool)
        self.assertGreater(S.auc(values, *S.label_masks(frac, ok, ok)), 0.99)
        self.assertLess(abs(S.rotated_null(values, frac, ok, ok)["auc"] - 0.5), 0.05)

    def test_published_grid_and_agreement(self):
        pub = np.zeros((8, 8), np.uint8)
        pub[:4, :4] = 255
        np.testing.assert_allclose(S.published_grid(pub), [[1, 0], [0, 0]])
        rng = np.random.default_rng(12)
        ref = rng.random((64, 64))
        rec = S.agreement(ref.astype(np.float32), ref, np.ones((64, 64), bool), sigma=2)
        self.assertAlmostEqual(rec["r_raw"], 1.0, places=6)
        self.assertAlmostEqual(rec["r_smooth"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
