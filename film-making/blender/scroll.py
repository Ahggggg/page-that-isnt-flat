"""Opening shot: PHerc. 139 whole, from its own CT, then cut open at the height of sheet w035.

  1   - 130  orbit the intact scroll (exterior = CT isosurface at 153.6 um, charred material, raking light)
  130 - 210  the top half lifts away, revealing the cross-section face at w035's mid-height
             (level-4 CT slice, 38.4 um/px), with w035's crossing glowing gold
  210 - 300  the camera descends toward the glowing winding
  300 - 360  the whole traced sheet w035 lights up in place (the film then dissolves to unroll.py)
Same coordinate frame throughout (mm; level-0 voxel = 2.399 um). Liberties (disclosed in the film): lighting, materials,
the lift-off animation and the glow.

  blender -b --factory-startup -P scroll.py -- --out <dir> [--preview] [--still N | --frames A B]
Environment: BLENDER_DEVICE (ONEAPI | OPTIX | CUDA), BLENDER_SAMPLES.
"""
import argparse
import math
import os
import sys
from pathlib import Path

import bpy
import numpy as np
import openvdb as vdb
from mathutils import Vector

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
VD = HERE.parent / "app/public/data"
INPUTS = Path(os.environ.get("FILM_INPUTS", HERE.parents[1] / "film-inputs"))   # fetch_film_inputs.py
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--preview", action="store_true")
ap.add_argument("--still", type=int, default=None)
ap.add_argument("--frames", nargs=2, type=int, default=[1, 360])
args = ap.parse_args(argv)
ORBIT, LIFT, DESCEND, N = 130, 210, 300, 360
L0_MM = 2.399e-3
VOX_MM = L0_MM * 32 * 2                                          # level 5, downsampled 2x
ZCUT = 18700 * L0_MM                                             # w035 mid-height (mm)

# ------------------------------------------------------------------------------------------- isosurface source
# Two half-volumes split at ZCUT (the other side zeroed), so each half's isosurface closes itself along the cut plane;
# downsampled 4x (307 um) for a light exterior mesh. No boolean operations.
VOX4 = L0_MM * 32 * 4
halves = {"lower": DATA / "scroll_lower.vdb", "upper": DATA / "scroll_upper.vdb"}
if not all(p.exists() for p in halves.values()):
    vol = np.load(INPUTS / "scroll_L5.npy", mmap_mode="r")
    z, y, x = (n // 4 * 4 for n in vol.shape)
    v = np.asarray(vol[:z, :y, :x], np.float32).reshape(z // 4, 4, y // 4, 4, x // 4, 4).mean(axis=(1, 3, 5)) / 255.0
    kc = int(round(ZCUT / VOX4))
    for name, path in halves.items():
        h = v.copy()
        if name == "lower":
            h[kc:] = 0
        else:
            h[:kc] = 0
        g = vdb.FloatGrid()
        g.copyFromArray(np.ascontiguousarray(h.transpose(2, 1, 0)))
        g.transform = vdb.createLinearTransform(voxelSize=VOX4)
        g.name = "density"
        vdb.write(str(path), grids=[g])
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene


def charred(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    m.node_tree.nodes.clear()
    b = m.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
    o = m.node_tree.nodes.new("ShaderNodeOutputMaterial")
    m.node_tree.links.new(b.outputs["BSDF"], o.inputs["Surface"])
    b.inputs["Base Color"].default_value = (0.05, 0.036, 0.027, 1)
    b.inputs["Roughness"].default_value = 0.55
    b.inputs["Coat Weight"].default_value = 0.15
    return m


def iso_half(name):
    bpy.ops.object.volume_import(filepath=str(halves[name]))
    vo = bpy.context.active_object
    vo.hide_render = True
    ob = bpy.data.objects.new(name, bpy.data.meshes.new(name))
    scene.collection.objects.link(ob)
    mod = ob.modifiers.new("v2m", "VOLUME_TO_MESH")
    mod.object = vo
    mod.threshold = 0.32
    mod.adaptivity = 0.2
    mod.use_smooth_shade = True
    ob.data.materials.append(charred(name + "_mat"))
    # bake: evaluate the volume-to-mesh modifier once and keep a fixed mesh (otherwise it re-meshes every frame)
    dg = bpy.context.evaluated_depsgraph_get()
    baked = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    ob.modifiers.clear()
    ob.data = baked
    ob.data.materials.clear()
    ob.data.materials.append(charred(name + "_mat"))
    bpy.data.objects.remove(vo)
    return ob


lower = iso_half("lower")
upper = iso_half("upper")

# ------------------------------------------------------------------------------------------- the cut face
S = 1657 * 16 * L0_MM                                            # level-4 slice extent (mm)
face = bpy.data.objects.new("face", bpy.data.meshes.new("facem"))
scene.collection.objects.link(face)
zf = ZCUT + 0.05
face.data.from_pydata([(0, 0, zf), (S, 0, zf), (S, S, zf), (0, S, zf)], [], [(0, 1, 2, 3)])
uvl = face.data.uv_layers.new(name="uv")
for loop, uv in zip(uvl.data, [(0, 1), (1, 1), (1, 0), (0, 0)]):      # image row 0 = y 0
    loop.uv = uv
fm = bpy.data.materials.new("face")
fm.use_nodes = True
nt = fm.node_tree
nt.nodes.clear()
fo = nt.nodes.new("ShaderNodeOutputMaterial")
bs = nt.nodes.new("ShaderNodeBsdfPrincipled")
tx = nt.nodes.new("ShaderNodeTexImage")
tx.image = bpy.data.images.load(str(DATA / "slice_L4.png"))
tx.image.colorspace_settings.name = "Non-Color"
al = nt.nodes.new("ShaderNodeTexImage")
al.image = bpy.data.images.load(str(DATA / "slice_alpha.png"))
al.image.colorspace_settings.name = "Non-Color"
gl = nt.nodes.new("ShaderNodeTexImage")
gl.image = bpy.data.images.load(str(DATA / "slice_glow.png"))
gl.image.colorspace_settings.name = "Non-Color"
tint = nt.nodes.new("ShaderNodeMixRGB")
tint.blend_type = "MULTIPLY"
tint.inputs["Fac"].default_value = 1.0
tint.inputs["Color2"].default_value = (0.95, 0.80, 0.62, 1)
nt.links.new(tx.outputs["Color"], tint.inputs["Color1"])
nt.links.new(tint.outputs["Color"], bs.inputs["Base Color"])
nt.links.new(al.outputs["Color"], bs.inputs["Alpha"])
bs.inputs["Roughness"].default_value = 0.6
bs.inputs["Emission Color"].default_value = (1.0, 0.6, 0.18, 1)
gmul = nt.nodes.new("ShaderNodeMath")
gmul.operation = "MULTIPLY"
gmul.inputs[1].default_value = 0.0
nt.links.new(gl.outputs["Color"], gmul.inputs[0])
nt.links.new(gmul.outputs[0], bs.inputs["Emission Strength"])
nt.links.new(bs.outputs["BSDF"], fo.inputs["Surface"])
face.data.materials.append(fm)

# ------------------------------------------------------------------------------------------- the sheet, in place
d = np.load(VD / "sheet.npz")
pos, valid = d["pos"].astype(np.float64), d["valid"]
R, C = valid.shape
idx = -np.ones((R, C), int)
idx[valid] = np.arange(valid.sum())
faces = [(idx[r, c], idx[r, c + 1], idx[r + 1, c + 1], idx[r + 1, c]) for r in range(R - 1) for c in range(C - 1)
         if min(idx[r, c], idx[r, c + 1], idx[r + 1, c + 1], idx[r + 1, c]) >= 0]
sm = bpy.data.meshes.new("w035")
sm.from_pydata([tuple(p) for p in pos[valid]], [], faces)
uvs = sm.uv_layers.new(name="page")
rr, cc = np.nonzero(valid)
uv = np.stack([cc / (C - 1), 1 - rr / (R - 1)], 1)
lv = np.empty(len(sm.loops), int)
sm.loops.foreach_get("vertex_index", lv)
uvs.data.foreach_set("uv", uv[lv].ravel())
sheet = bpy.data.objects.new("w035", sm)
scene.collection.objects.link(sheet)
shm = bpy.data.materials.new("sheet")
shm.use_nodes = True
st = shm.node_tree
st.nodes.clear()
so = st.nodes.new("ShaderNodeOutputMaterial")
em = st.nodes.new("ShaderNodeEmission")
itx = st.nodes.new("ShaderNodeTexImage")
itx.image = bpy.data.images.load(str(VD / "ink_8k.png"))
itx.image.colorspace_settings.name = "Non-Color"
mr = st.nodes.new("ShaderNodeMapRange")
mr.inputs["From Min"].default_value = 0.35
mr.inputs["To Min"].default_value = 0.12
st.links.new(itx.outputs["Color"], mr.inputs["Value"])
smul = st.nodes.new("ShaderNodeMath")
smul.operation = "MULTIPLY"
smul.inputs[1].default_value = 0.0
st.links.new(mr.outputs["Result"], smul.inputs[0])
em.inputs["Color"].default_value = (1.0, 0.62, 0.22, 1)
st.links.new(smul.outputs[0], em.inputs["Strength"])
tr = st.nodes.new("ShaderNodeBsdfTransparent")
mix = st.nodes.new("ShaderNodeMixShader")
mix.inputs["Fac"].default_value = 0.0                             # 0 = invisible, 1 = glowing sheet
st.links.new(tr.outputs["BSDF"], mix.inputs[1])
st.links.new(em.outputs["Emission"], mix.inputs[2])
st.links.new(mix.outputs["Shader"], so.inputs["Surface"])
sheet.data.materials.append(shm)
scroll_c = Vector((S / 2, S / 2, 2405 * 0.0768 / 2))
cut_c = Vector((S / 2, S / 2, ZCUT))
arc = np.load(DATA / "slice_w035_pts.npy").mean(axis=0) * 16 * L0_MM
arc_c = Vector((float(arc[0]), float(arc[1]), ZCUT))

# ------------------------------------------------------------------------------------------- light, camera, timing
world = bpy.data.worlds.new("w")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.004, 0.005, 0.009, 1)
for name, rot, energy, col in (("key", (math.radians(35), 0, math.radians(-30)), 3.4, (1.0, 0.9, 0.78)),
                               ("top", (math.radians(8), 0, math.radians(20)), 1.4, (1.0, 0.95, 0.9)),
                               ("rim", (math.radians(70), 0, math.radians(150)), 1.6, (0.6, 0.75, 1.0))):
    ld = bpy.data.lights.new(name, "SUN")
    ld.energy, ld.color, ld.angle = energy, col, math.radians(3)
    lo = bpy.data.objects.new(name, ld)
    lo.rotation_euler = rot
    scene.collection.objects.link(lo)
cam_data = bpy.data.cameras.new("cam")
cam_data.lens = 50
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
tgt = bpy.data.objects.new("t", None)
scene.collection.objects.link(tgt)
con = cam.constraints.new("TRACK_TO")
con.target, con.track_axis, con.up_axis = tgt, "TRACK_NEGATIVE_Z", "UP_Y"


def s(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def update(sc, *_):
    f = sc.frame_current
    lift = s((f - ORBIT) / (LIFT - ORBIT))
    desc = s((f - LIFT) / (DESCEND - LIFT))
    glow = s((f - LIFT + 20) / 60)
    sheet_on = s((f - DESCEND) / (N - DESCEND))
    upper.location = Vector((0, 0, 160 * lift))
    upper.hide_render = lift > 0.98
    face.hide_render = lift < 0.02
    gmul.inputs[1].default_value = 9.0 * glow
    smul.inputs[1].default_value = 6.0 * sheet_on
    mix.inputs["Fac"].default_value = sheet_on
    sheet.hide_render = sheet_on < 0.005
    az = math.radians(-35 + 70 * s(f / LIFT) + 20 * desc)
    el = math.radians(18 + 40 * lift + 12 * desc)
    target = scroll_c.lerp(cut_c, lift).lerp(arc_c, desc)
    dist = 300 - 120 * lift - 110 * desc
    tgt.location = target
    cam.location = target + Vector((math.cos(el) * math.sin(az), -math.cos(el) * math.cos(az), math.sin(el))) * dist


bpy.app.handlers.frame_change_pre.append(update)
scene.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
DEVICE = os.environ.get("BLENDER_DEVICE", "ONEAPI")
prefs.compute_device_type = DEVICE
prefs.get_devices()
for dev in prefs.devices:
    dev.use = dev.type == DEVICE
scene.cycles.device = "GPU"
scene.cycles.samples = 32 if args.preview else 192
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
