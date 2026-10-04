"""The traced sheet and its texture, for the unrolling and scroll shots (Blender) and the title card (Remotion).

  sheet.npz   w035's traced sheet, grid decimated 4x: 3D positions (mm, scroll frame), flat positions (mm), validity
  ink_8k.png  the organizers' published w035 ink prediction, 8192 px wide, grey 0-255
"""
import numpy as np
import tifffile
from PIL import Image

from _common import APP_DATA, F

VOXEL_MM = 2.399e-3


def sheet(step=4):
    m = F.load_tifxyz(F.MESH)
    xyz = m.xyz[::step, ::step].astype(np.float64)
    valid = np.all(xyz > 0, axis=-1) & np.all(np.isfinite(xyz), axis=-1)
    rows, cols = np.mgrid[0:xyz.shape[0], 0:xyz.shape[1]]
    grid_px = step / m.scale
    flat = np.stack([cols * grid_px * VOXEL_MM, rows * grid_px * VOXEL_MM], axis=-1)
    pos = xyz * VOXEL_MM                                    # x, y, z in mm; z is the scroll's long axis
    np.savez_compressed(APP_DATA / "sheet.npz", pos=pos.astype(np.float32), flat=flat.astype(np.float32),
                        valid=valid, grid_px=grid_px, page_px=np.array([22640, 20400]))
    print("sheet", xyz.shape, f"{valid.mean():.3f} valid", "grid step px", grid_px)


def ink():
    a = tifffile.imread(str(F.published_prediction()))
    w = 8192
    Image.fromarray(a).resize((w, round(a.shape[0] * w / a.shape[1])), Image.LANCZOS).save(APP_DATA / "ink_8k.png")
    print("ink", a.shape)


if __name__ == "__main__":
    sheet()
    ink()
