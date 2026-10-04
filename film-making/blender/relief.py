"""'The page that isn't flat': the flat page tilts and rises into the real, crumpled sheet surface.

Height = CT-only sheet depth around the zoom target (relief_assets.py), exaggerated EXAG times (stated on screen in
the film); textures = raw CT surface plane (base colour) and the published ink prediction (gold emission). Lit by a
low raking light, as papyrologists light real papyri to see relief.

  blender -b --factory-startup -P relief.py -- --out <dir> [--preview] [--still N | --frames A B]
"""
import argparse
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--preview", action="store_true")
ap.add_argument("--still", type=int, default=None)
ap.add_argument("--frames", nargs=2, type=int, default=[1, 360])
args = ap.parse_args(argv)

EXAG = 6.0
FLAT_END, RISE_END, N = 50, 170, 360
z = np.load(DATA / "relief.npz")
hp = z["height_planes"]
um = float(z["um_per_px"])
R, C = hp.shape
W_MM, H_MM = C * um / 1000, R * um / 1000
zmm = hp * 2.399e-3 * EXAG                                    # exaggerated height in mm

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
xs = (np.arange(C) / (C - 1) - 0.5) * W_MM
ys = (0.5 - np.arange(R) / (R - 1)) * H_MM
X, Y = np.meshgrid(xs, ys)
verts = np.stack([X.ravel(), Y.ravel(), np.zeros(R * C)], 1)
ii = np.arange(R * C).reshape(R, C)
faces = np.stack([ii[:-1, :-1].ravel(), ii[:-1, 1:].ravel(), ii[1:, 1:].ravel(), ii[1:, :-1].ravel()], 1)
mesh = bpy.data.meshes.new("sheet")
mesh.vertices.add(len(verts))
mesh.vertices.foreach_set("co", verts.astype(np.float32).ravel())
mesh.loops.add(faces.size)
mesh.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
mesh.polygons.add(len(faces))
mesh.polygons.foreach_set("loop_start", (np.arange(len(faces)) * 4).astype(np.int32))
mesh.polygons.foreach_set("loop_total", np.full(len(faces), 4, np.int32))
mesh.update()
mesh.validate()
uvl = mesh.uv_layers.new(name="uv")
uv = np.stack([np.tile(np.arange(C) / (C - 1), R), np.repeat(1 - np.arange(R) / (R - 1), C)], 1)
lv = np.empty(len(mesh.loops), int)
mesh.loops.foreach_get("vertex_index", lv)
uvl.data.foreach_set("uv", uv[lv].astype(np.float32).ravel())
mesh.polygons.foreach_set("use_smooth", np.ones(len(faces), bool))
obj = bpy.data.objects.new("sheet", mesh)
scene.collection.objects.link(obj)
base = verts.copy()
target_z = zmm.ravel()

# material: CT as charred papyrus, ink as gold glow
mat = bpy.data.materials.new("papyrus")
mat.use_nodes = True
nt = mat.node_tree
nt.nodes.clear()
out = nt.nodes.new("ShaderNodeOutputMaterial")
bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
ct = nt.nodes.new("ShaderNodeTexImage")
ct.image = bpy.data.images.load(str(DATA / "relief_ct.png"))
ct.image.colorspace_settings.name = "Non-Color"
ink = nt.nodes.new("ShaderNodeTexImage")
ink.image = bpy.data.images.load(str(DATA / "relief_ink.png"))
ink.image.colorspace_settings.name = "Non-Color"
tint = nt.nodes.new("ShaderNodeMixRGB")
tint.blend_type = "MULTIPLY"
tint.inputs["Fac"].default_value = 1.0
tint.inputs["Color2"].default_value = (0.62, 0.43, 0.27, 1)
nt.links.new(ct.outputs["Color"], tint.inputs["Color1"])
nt.links.new(tint.outputs["Color"], bsdf.inputs["Base Color"])
bsdf.inputs["Roughness"].default_value = 0.8
ramp = nt.nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].position = 0.4
ramp.color_ramp.elements[1].position = 0.9
nt.links.new(ink.outputs["Color"], ramp.inputs["Fac"])
mul = nt.nodes.new("ShaderNodeMath")
mul.operation = "MULTIPLY"
mul.inputs[1].default_value = 3.0
nt.links.new(ramp.outputs["Color"], mul.inputs[0])
bsdf.inputs["Emission Color"].default_value = (1.0, 0.62, 0.22, 1)
nt.links.new(mul.outputs[0], bsdf.inputs["Emission Strength"])
nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
obj.data.materials.append(mat)

world = bpy.data.worlds.new("w")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.003, 0.004, 0.008, 1)
# raking light: low sun from the left, plus a faint cool fill
for name, rot, energy, col in (("rake", (math.radians(80), 0, math.radians(-75)), 4.0, (1.0, 0.92, 0.82)),
                               ("fill", (math.radians(35), 0, math.radians(120)), 0.35, (0.7, 0.8, 1.0))):
    ld = bpy.data.lights.new(name, "SUN")
    ld.energy, ld.color, ld.angle = energy, col, math.radians(2)
    lo = bpy.data.objects.new(name, ld)
    lo.rotation_euler = rot
    scene.collection.objects.link(lo)

cam_data = bpy.data.cameras.new("cam")
cam_data.lens = 85
cam_data.dof.use_dof = True
cam_data.dof.aperture_fstop = 2.8
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
target = bpy.data.objects.new("t", None)
scene.collection.objects.link(target)
cam_data.dof.focus_object = target
con = cam.constraints.new("TRACK_TO")
con.target, con.track_axis, con.up_axis = target, "TRACK_NEGATIVE_Z", "UP_Y"


def sm(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def update(sc, *_):
    f = sc.frame_current
    rise = sm((f - FLAT_END) / (RISE_END - FLAT_END))
    co = base.copy()
    co[:, 2] = target_z * rise
    obj.data.vertices.foreach_set("co", co.astype(np.float32).ravel())
    obj.data.update()
    tilt = sm((f - FLAT_END + 20) / (RISE_END - FLAT_END + 20))
    el = math.radians(89.5 - 54 * tilt)                          # straight down -> 35.5 deg
    drift = sm((f - RISE_END) / (N - RISE_END))
    tx = -0.25 * W_MM + 0.5 * W_MM * drift * 0.6
    target.location = Vector((tx * tilt, 0, 0))
    dist = 23.0 - 6.0 * tilt
    az = math.radians(-8 * tilt)
    cam.location = target.location + Vector((dist * math.cos(el) * math.sin(az), -dist * math.cos(el) * math.cos(az), dist * math.sin(el)))


bpy.app.handlers.frame_change_pre.append(update)
scene.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
import os
DEVICE = os.environ.get("BLENDER_DEVICE", "ONEAPI")           # OPTIX on the RunPod 4090, ONEAPI on the laptop's Arc
prefs.compute_device_type = DEVICE
prefs.get_devices()
for dev in prefs.devices:
    dev.use = dev.type == DEVICE
scene.cycles.device = "GPU"
scene.cycles.samples = 24 if args.preview else 160
scene.cycles.use_denoising = True
if DEVICE in ("OPTIX", "CUDA"):
    scene.cycles.denoiser = "OPTIX"                              # GPU denoiser; OIDN fell back to a slow CPU path on the pod
else:
    scene.cycles.denoising_use_gpu = True
if os.environ.get("BLENDER_SAMPLES"):
    scene.cycles.samples = int(os.environ["BLENDER_SAMPLES"])
scene.render.resolution_x, scene.render.resolution_y = (960, 540) if args.preview else (1920, 1080)
scene.render.fps = 30
if os.environ.get("BLENDER_SCALE"):
    scene.render.resolution_percentage = int(100 * float(os.environ["BLENDER_SCALE"]))   # 2 -> 3840 x 2160
scene.view_settings.view_transform = "AgX"
scene.view_settings.look = "AgX - Punchy"
scene.frame_start, scene.frame_end = args.frames
outdir = Path(args.out).resolve()
outdir.mkdir(parents=True, exist_ok=True)
if args.still is not None:
    scene.frame_set(args.still)
    scene.render.filepath = str(outdir / f"still_{args.still:04d}.png")
    bpy.ops.render.render(write_still=True)
else:
    scene.render.filepath = str(outdir / "f_")
    bpy.ops.render.render(animation=True)
