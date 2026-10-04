"""The organizers' canonical 2.4 um ink model (`scrollprize/ink_canonical_2um`, a ResNet-152 3D encoder with a 3D
decoder): checkpoint audit, weights-only loading, preprocessing and tiled inference with a movable depth window.

The network definition is not shipped here: `fetch --model` downloads three small files of the organizers' MIT
`ScrollPrize/villa` repository at a pinned commit, and they are hash-checked before they are imported.

Depth window. The organizers' surface volumes have 109 planes with the traced surface (the mesh plane) at index 54.
The model reads 62 consecutive planes; centred on the mesh plane that is planes 23..84 (start 23, stop 85). An
*offset* of k planes reads planes 23+k .. 84+k, i.e. a window moved k planes (k x voxel size) along the surface
normal; positive k moves toward higher plane indices. Offsets run from -23 to +24 on a 109-plane volume.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np

# ---- constants fixed by the organizers' pipeline (do not change without re-verifying) ------------------------------
CLIP = 200.0            # raw uint8 intensities are clipped to [0, CLIP] then scaled to [0, 1]
TILE = 256              # inference tile side, in surface-volume pixels
STRIDE = 128            # organizers' tile stride: 50% overlap, Hann-blended
OUT_SCALE = 4           # the network's output grid is 1/4 of its input tile
FRAMES = 62             # planes the network reads
PLANES = 109            # planes in the organizers' 2.4 um surface volumes
MESH_PLANE = 54         # index of the traced surface in those volumes
PRECISIONS = ("fp32", "tf32", "bf16", "fp16")   # tf32: NVIDIA TensorFloat-32 convolutions (PyTorch's CUDA default)

# Pickle globals the checkpoint audit accepts: plain tensors and containers. Anything else must be an allowed
# container class (loaded as an inert stand-in) or the load is refused.
ALLOWED_CONTAINER_MODULE_PREFIXES = ("lightning", "pytorch_lightning")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_hash(path, expected_sha256: str, label: str = "") -> None:
    got = sha256_file(path)
    if got != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {label or path}: expected {expected_sha256}, got {got}")


# ------------------------------------------------------------------------------------------------- depth window
def window_for_offset(k: int, planes: int = PLANES, mesh_plane: int = MESH_PLANE) -> tuple[int, int]:
    """(start, stop) of the 62-plane window moved k planes off the centred window (start mesh_plane - 31)."""
    start = mesh_plane - FRAMES // 2 + int(k)
    stop = start + FRAMES
    if start < 0 or stop > planes:
        lo, hi = FRAMES // 2 - mesh_plane, planes - FRAMES - mesh_plane + FRAMES // 2
        raise ValueError(f"offset {k} puts the window at planes {start}..{stop - 1}, outside 0..{planes - 1}; "
                         f"offsets must lie in [{lo}, {hi}] for a {planes}-plane volume")
    return start, stop


def arm_name(k: int, reverse: bool = False) -> str:
    return f"{'R' if reverse else 'k'}{int(k):+d}"


def parse_arm(name: str) -> tuple[int, bool]:
    if name[0] not in "kR":
        raise ValueError(f"not an arm name: {name!r}")
    return int(name[1:]), name[0] == "R"


# -------------------------------------------------------------------------------------------- villa staging
def stage_villa(villa_dir, manifest) -> None:
    """Hash-check the pinned villa files and put villa_dir on sys.path (their imports expect a `models` package)."""
    villa_dir = Path(villa_dir)
    for row in manifest["villa"]["files"]:
        if row["path"] == "LICENSE":
            continue
        target = villa_dir / row["path"]
        if not target.exists():
            raise FileNotFoundError(f"missing villa file {row['path']} under {villa_dir}; run `fetch --model` first")
        verify_hash(target, row["sha256"], row["path"])
    if str(villa_dir) not in sys.path:
        sys.path.insert(0, str(villa_dir))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------------------------- checkpoint audit
class _InertContainer(dict):
    """Stand-in for a non-tensor container class named in the checkpoint's pickle (e.g. a Lightning hyper-parameter
    namespace): holds the data pickle hands it; no method of the real class ever runs."""

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.update(state)


def audit_checkpoint(path) -> list[str]:
    """Every global the checkpoint's pickle stream references, found by parsing opcodes (nothing is executed)."""
    import torch
    scanner = getattr(torch.serialization, "get_unsafe_globals_in_checkpoint", None)
    if scanner is None:
        raise RuntimeError("the checkpoint audit needs PyTorch 2.6 or newer")
    return sorted(scanner(str(path)))


def check_globals(unsafe: list[str]) -> list[tuple[type, str]]:
    """Inert stand-ins for allowed container globals; raises on anything else."""
    stubs = []
    for g in unsafe:
        module, _, name = g.rpartition(".")
        if not module.startswith(ALLOWED_CONTAINER_MODULE_PREFIXES):
            raise ValueError(f"checkpoint references pickle global {g!r}, which is not a tensor/container or one of "
                             f"{ALLOWED_CONTAINER_MODULE_PREFIXES}; refusing to load rather than run arbitrary code")
        stubs.append((type(name, (_InertContainer,), {"__module__": module}), g))
    return stubs


def load_model(checkpoint_path, villa_dir, manifest, device="cpu"):
    """Hash-check, audit, then load the checkpoint weights-only into the exact architecture (strict)."""
    import torch
    verify_hash(checkpoint_path, manifest["checkpoint"]["sha256"], "checkpoint")
    stage_villa(villa_dir, manifest)
    decoder = _load_module("tracecheck_villa_resnet3d_3d_decoder", Path(villa_dir) / "model_resnet3d_3d_decoder.py")
    stubs = check_globals(audit_checkpoint(checkpoint_path))
    with torch.serialization.safe_globals(stubs):
        raw = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    state = decoder._strip_known_prefixes(decoder._extract_state_dict(raw))
    model = decoder.RegressionModel(with_norm=any(k.startswith("normalization.") for k in state))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"state_dict does not match RegressionModel: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.to(device).eval()
    return model


# --------------------------------------------------------------------------------------------------- inference
def preprocess(tiles: np.ndarray) -> np.ndarray:
    """uint8 (..., D, H, W) -> float32 in [0, 1], clipped at CLIP."""
    return np.minimum(tiles.astype(np.float32), CLIP) / CLIP


def set_tf32(enabled: bool) -> None:
    """PyTorch lets cuDNN convolutions on recent NVIDIA GPUs use TensorFloat-32 (about 3 significant digits) unless
    told otherwise, so an 'FP32' run on CUDA is not FP32 by default. "fp32" turns it off; "tf32" turns it on."""
    import torch
    torch.backends.cudnn.allow_tf32 = enabled
    torch.backends.cuda.matmul.allow_tf32 = enabled


def forward_tiles(model, tiles: np.ndarray, device, precision: str = "fp32") -> np.ndarray:
    """uint8 (B, FRAMES, TILE, TILE) -> sigmoid probabilities (B, TILE/4, TILE/4). fp32 is strict FP32 on every
    device; tf32 allows NVIDIA TensorFloat-32; bf16/fp16 run under autocast."""
    import torch
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, not {precision!r}")
    set_tf32(precision == "tf32")
    x = preprocess(tiles)
    dev = torch.device(device)
    with torch.inference_mode():
        inp = torch.from_numpy(x[:, None]).to(dev)
        if precision in ("fp32", "tf32"):
            y = model(inp)
        else:
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
            with torch.autocast(device_type=dev.type, dtype=dtype):
                y = model(inp)
        return torch.sigmoid(y.float())[:, 0].cpu().numpy()


def grid(n: int, tile: int = TILE, stride: int = STRIDE) -> list[int]:
    """Tile starts covering [0, n): stride apart, plus n - tile so the last tile is flush with the far edge."""
    if n < tile:
        raise ValueError(f"region side {n} is smaller than the tile ({tile})")
    pos = list(range(0, n - tile + 1, stride))
    if pos[-1] != n - tile:
        pos.append(n - tile)
    return pos


def hann_weights(tile: int = TILE, out_scale: int = OUT_SCALE) -> np.ndarray:
    ot = tile // out_scale
    return np.maximum(np.outer(np.hanning(ot), np.hanning(ot)), 1e-3).astype(np.float32)


def infer_map(forward, stack: np.ndarray, window: tuple[int, int], reverse: bool = False, batch: int = 4,
              tile: int = TILE, stride: int = STRIDE, out_scale: int = OUT_SCALE, progress=None):
    """Tiled inference over a (planes, H, W) uint8 stack. Returns (map, tiles_run).

    forward: callable uint8 (B, FRAMES, tile, tile) -> float (B, tile/4, tile/4), e.g.
    `lambda t: forward_tiles(model, t, device, precision)`. map: float32 (H/4, W/4), Hann-blended where tiles
    overlap. Output cell (r, c) covers stack pixels (4r..4r+3, 4c..4c+3). reverse=True feeds the window's planes in
    the opposite order (a null: the model was trained on one order).
    """
    a, b = window
    win = stack[a:b][::-1] if reverse else stack[a:b]
    if win.shape[0] != FRAMES:
        raise ValueError(f"window ({a}, {b}) does not select {FRAMES} planes: got {win.shape[0]}")
    H, W = stack.shape[1:]
    if H % out_scale or W % out_scale:
        raise ValueError(f"region {H} x {W} must be divisible by {out_scale}")
    oh, ow, ot = H // out_scale, W // out_scale, tile // out_scale
    hann = hann_weights(tile, out_scale)
    acc = np.zeros((oh, ow), np.float32)
    wsum = np.zeros((oh, ow), np.float32)
    coords = [(y, x) for y in grid(H, tile, stride) for x in grid(W, tile, stride)]
    for i in range(0, len(coords), batch):
        part = coords[i:i + batch]
        tiles = np.stack([np.ascontiguousarray(win[:, y:y + tile, x:x + tile]) for y, x in part])
        probs = forward(tiles)
        for (y, x), pmap in zip(part, probs):
            yo, xo = y // out_scale, x // out_scale
            acc[yo:yo + ot, xo:xo + ot] += pmap * hann
            wsum[yo:yo + ot, xo:xo + ot] += hann
        if progress is not None:
            progress(min(i + batch, len(coords)), len(coords))
    with np.errstate(invalid="ignore"):
        out = np.where(wsum > 0, acc / np.maximum(wsum, 1e-12), np.nan)
    return out.astype(np.float32), len(coords)
