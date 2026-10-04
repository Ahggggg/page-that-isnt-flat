"""Shot 2, 'The unwrap': the traced w035 sheet in its true shape inside PHerc. 139 unrolls into the flat page.

Every vertex comes from the organizers' tifxyz mesh (decimated 4x); the texture is their published ink prediction.
The unroll is an interpolation between the measured 3D positions and the measured flat (grid) positions, swept
across the page's columns; nothing is drawn by hand.

  blender -b --factory-startup -P unroll.py -- --out <dir> [--preview] [--frames 1 300] [--still 150]
"""
import argparse
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "app/public/data"
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--preview", action="store_true")
ap.add_argument("--frames", nargs=2, type=int, default=[1, 300])
ap.add_argument("--still", type=int, default=None)
ap.add_argument("--debug", action="store_true")
args = ap.parse_args(argv)

FPS, N_FRAMES = 30, 300
HOLD_3D, UNROLL_END = 75, 225          # orbit the curved sheet, unroll, settle on the page

# ------------------------------------------------------------------------------------------- geometry
d = np.load(DATA / "sheet.npz")
pos, flat, valid = d["pos"].astype(np.float64), d["flat"].astype(np.float64), d["valid"]
R, C = valid.shape
# 3D: scroll axis (z) is the page's row direction; flip it so the page's first row is at the top.
p3 = pos.copy()
p3[..., 2] *= -1
centre = p3[valid].mean(axis=0)
p3 -= centre
# turn the sheet about the vertical axis so its mean outward direction faces the camera (-Y)
pts = p3[valid]
u, s, vt = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
normal = vt[2]
normal[2] = 0
if np.linalg.norm(normal) > 0:
    normal /= np.linalg.norm(normal)
    ang = math.atan2(normal[0], -normal[1])
    ca, sa = math.cos(-ang), math.sin(-ang)
    x, y = p3[..., 0].copy(), p3[..., 1].copy()
    p3[..., 0], p3[..., 1] = ca * x - sa * y, sa * x + ca * y
# which way does the sheet bulge? put the flat page on the camera side of its mean depth
pf = np.zeros_like(p3)
pf[..., 0] = flat[..., 0] - flat[..., 0][valid].mean()
pf[..., 2] = -(flat[..., 1] - flat[..., 1][valid].mean())
pf[..., 1] = p3[..., 1][valid].min() - 2.0
# match the left-right sense of the curved sheet's columns to the flat page's
if np.corrcoef(p3[..., 0][valid], pf[..., 0][valid])[0, 1] < 0:
    p3[..., 0] *= -1
    p3[..., 1] *= -1

idx = -np.ones((R, C), int)
idx[valid] = np.arange(valid.sum())
faces = []
for r in range(R - 1):
    for c in range(C - 1):
        q = (idx[r, c], idx[r, c + 1], idx[r + 1, c + 1], idx[r + 1, c])
        if min(q) >= 0:
            faces.append(q)
rr, cc = np.nonzero(valid)
uv = np.stack([cc / (C - 1), 1 - rr / (R - 1)], 1)
colfrac = cc / (C - 1)
P3, PF = p3[valid], pf[valid]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
mesh = bpy.data.meshes.new("w035")
mesh.from_pydata([tuple(v) for v in P3], [], faces)
mesh.update()
layer = mesh.uv_layers.new(name="page")
loop_vidx = np.empty(len(mesh.loops), int)
mesh.loops.foreach_get("vertex_index", loop_vidx)
layer.data.foreach_set("uv", uv[loop_vidx].ravel())
for poly in mesh.polygons:
    poly.use_smooth = True
obj = bpy.data.objects.new("w035", mesh)
scene.collection.objects.link(obj)
solid = obj.modifiers.new("thickness", "SOLIDIFY")
solid.thickness = 0.06                      # ~ papyrus sheet thickness in mm, for edge light only
solid.offset = 0

# ------------------------------------------------------------------------------------------- material
mat = bpy.data.materials.new("page")
mat.use_nodes = True
nt = mat.node_tree
nt.nodes.clear()
out = nt.nodes.new("ShaderNodeOutputMaterial")
tex = nt.nodes.new("ShaderNodeTexImage")
tex.image = bpy.data.images.load(str(DATA / "ink_8k.png"))
tex.image.colorspace_settings.name = "Non-Color"
tex.interpolation = "Cubic"
ramp = nt.nodes.new("ShaderNodeValToRGB")         # ink probability -> emission weight (fixed curve)
ramp.color_ramp.elements[0].position = 0.35
ramp.color_ramp.elements[1].position = 0.85
bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
bsdf.inputs["Base Color"].default_value = (0.055, 0.035, 0.022, 1)      # charred papyrus
surf_png = DATA / "surface_8k.png"                 # CT surface plane (organizers' surface volume, plane 54), if built
if surf_png.exists():
    stex = nt.nodes.new("ShaderNodeTexImage")
    stex.image = bpy.data.images.load(str(surf_png))
    stex.image.colorspace_settings.name = "Non-Color"
    tint = nt.nodes.new("ShaderNodeMixRGB")
    tint.blend_type = "MULTIPLY"
    tint.inputs["Fac"].default_value = 1.0
    tint.inputs["Color2"].default_value = (0.42, 0.30, 0.20, 1)
    nt.links.new(stex.outputs["Color"], tint.inputs["Color1"])
    nt.links.new(tint.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(stex.outputs["Color"], bsdf.inputs["Roughness"])
bsdf.inputs["Roughness"].default_value = 0.7
bsdf.inputs["Emission Color"].default_value = (1.0, 0.62, 0.22, 1)      # ink glow, warm gold
nt.links.new(tex.outputs["Color"], ramp.inputs["Fac"])
mul = nt.nodes.new("ShaderNodeMath")
mul.operation = "MULTIPLY"
mul.inputs[1].default_value = 6.0
nt.links.new(ramp.outputs["Color"], mul.inputs[0])
nt.links.new(mul.outputs[0], bsdf.inputs["Emission Strength"])
nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
obj.data.materials.append(mat)

# ------------------------------------------------------------------------------------------- world, light
world = bpy.data.worlds.new("w")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.004, 0.006, 0.012, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
for name, loc, energy, size in (("key", (-40, -90, 50), 40000, 40), ("rim", (60, 40, 30), 5000, 20)):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy, ld.size = energy, size
    ld.color = (0.75, 0.82, 1.0) if name == "rim" else (1.0, 0.93, 0.85)
    lo = bpy.data.objects.new(name, ld)
    lo.location = loc
    lo.rotation_euler = (Vector((0, 0, 0)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(lo)

# ------------------------------------------------------------------------------------------- camera
cam_data = bpy.data.cameras.new("cam")
cam_data.lens = 50
cam_data.dof.use_dof = True
cam_data.dof.aperture_fstop = 5.6
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
target = bpy.data.objects.new("target", None)
scene.collection.objects.link(target)
cam_data.dof.focus_object = target
con = cam.constraints.new("TRACK_TO")
con.target, con.track_axis, con.up_axis = target, "TRACK_NEGATIVE_Z", "UP_Y"


def smooth(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def camera_at(f, centre):
    # orbit from high above the curved sheet (its arc visible) down to straight-on at the flat page, always aimed at
    # the sheet's current centre
    a = smooth((f - 1) / (UNROLL_END + 40))
    az = math.radians(-70 + 70 * a)
    el = math.radians(55 - 55 * a)
    dist = 125 + 25 * a
    return centre + Vector((dist * math.sin(az) * math.cos(el), -dist * math.cos(az) * math.cos(el), dist * math.sin(el)))


def update(scene_, *_):
    f = scene_.frame_current
    t = (f - HOLD_3D) / (UNROLL_END - HOLD_3D)
    front = -0.25 + 1.5 * min(max(t, 0.0), 1.0)
    b = np.clip((front - colfrac) / 0.25, 0, 1)
    b = b * b * (3 - 2 * b)
    co = P3 + (PF - P3) * b[:, None]
    obj.data.vertices.foreach_set("co", co.ravel())
    obj.data.update()
    centre = Vector(tuple(float(v) for v in (co.min(axis=0) + co.max(axis=0)) / 2))
    target.location = centre
    cam.location = camera_at(f, centre)


bpy.app.handlers.frame_change_pre.append(update)

# ------------------------------------------------------------------------------------------- render
scene.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
import os
DEVICE = os.environ.get("BLENDER_DEVICE", "ONEAPI")           # OPTIX on the RunPod 4090, ONEAPI on the laptop's Arc
prefs.compute_device_type = DEVICE
prefs.get_devices()
for dev in prefs.devices:
    dev.use = dev.type == DEVICE
scene.cycles.device = "GPU"
scene.cycles.samples = 24 if args.preview else 192
scene.cycles.use_denoising = True
if DEVICE in ("OPTIX", "CUDA"):
    scene.cycles.denoiser = "OPTIX"                              # GPU denoiser; OIDN fell back to a slow CPU path on the pod
else:
    scene.cycles.denoising_use_gpu = True
if os.environ.get("BLENDER_SAMPLES"):
    scene.cycles.samples = int(os.environ["BLENDER_SAMPLES"])
scene.render.resolution_x, scene.render.resolution_y = (960, 540) if args.preview else (1920, 1080)
scene.render.fps = FPS
if os.environ.get("BLENDER_SCALE"):
    scene.render.resolution_percentage = int(100 * float(os.environ["BLENDER_SCALE"]))   # 2 -> 3840 x 2160
scene.view_settings.view_transform = "AgX"
scene.view_settings.look = "AgX - Punchy"
scene.render.image_settings.file_format = "PNG"
scene.frame_start, scene.frame_end = args.frames
# compositor glare for the ink glow
scene.use_nodes = True
ct = scene.node_tree
for n in list(ct.nodes):
    ct.nodes.remove(n)
rl = ct.nodes.new("CompositorNodeRLayers")
glare = ct.nodes.new("CompositorNodeGlare")
glare.glare_type = "FOG_GLOW"
glare.quality = "HIGH"
glare.threshold = 0.8
glare.size = 7
comp = ct.nodes.new("CompositorNodeComposite")
ct.links.new(rl.outputs["Image"], glare.inputs["Image"])
ct.links.new(glare.outputs["Image"], comp.inputs["Image"])

outdir = Path(args.out).resolve()
outdir.mkdir(parents=True, exist_ok=True)
if args.debug:
    for fr in (1, 75, 150, 270):
        scene.frame_set(fr)
        bpy.context.view_layer.update()
        print("DBG", fr, "cam", tuple(round(v, 1) for v in cam.matrix_world.translation),
              "target", tuple(round(v, 1) for v in target.location), "rot", tuple(round(v, 2) for v in cam.matrix_world.to_euler()))
    sys.exit(0)
if args.still is not None:
    scene.frame_set(args.still)
    scene.render.filepath = str(outdir / f"still_{args.still:04d}.png")
    bpy.ops.render.render(write_still=True)
else:
    scene.render.filepath = str(outdir / "f_")
    bpy.ops.render.render(animation=True)
