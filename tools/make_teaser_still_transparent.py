"""Artistic teaser still: individual ArticuRiddle objects arranged as a still life.

Each item is "<pkg>:<orig|edit>:<frame>" -- one cabinet out of a pair package, frozen at its own
opening (the package animates closed at frame 15 -> open at 105). Items are sorted by height and
dealt into staggered rows (tallest at the back), each turned towards the camera with a little
jitter, on a seamless studio backdrop.

  blender -b --python make_teaser_still.py -- --items A:edit:90 B:edit:120 ... --out teaser_v3 \
      [--samples 128] [--res 2400 1350] [--seed 3] [--bg 0.80 0.82 0.85]

Colour: Standard view transform, no exposure push -- AgX desaturates the wood textures and made
the cabinets look paler/pinker than in IsaacGym, which shows the texture albedo nearly as-is.
"""
import math
import os
import random
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def args():
    a = sys.argv[sys.argv.index("--") + 1:]
    o = {"items": [], "out": "teaser", "samples": 128, "res": (2400, 1350), "seed": 3,
         "bg": (0.55, 0.58, 0.62), "focal": 85.0, "elev": 20.0, "azim": -18.0, "rows": 0,
         "key": 700.0, "exposure": 0.0, "view": "Standard", "gap": 0.35,
         "save_lib": "", "from_lib": 0, "n": 0, "layout": "rows", "spread": 1.6, "floor_aspect": 2.4,
         "yaw_jitter": 12.0, "open_frames": "", "anim_frames": 0, "closed_frame": 15, "phase_spread": 0.45, "pkg_root": "/mnt/kostas_home/lruiyao/PartManip_su/riddle_pairs_render"}
    i = 0
    while i < len(a):
        k = a[i]
        if k in ("--items",):
            i += 1
            while i < len(a) and not a[i].startswith("--"):
                o["items"].append(a[i]); i += 1
            continue
        if k == "--res":
            o["res"] = (int(a[i + 1]), int(a[i + 2])); i += 3; continue
        if k == "--bg":
            o["bg"] = tuple(float(x) for x in a[i + 1:i + 4]); i += 4; continue
        key = k[2:]
        o[key] = a[i + 1] if isinstance(o[key], str) else type(o[key])(a[i + 1])
        i += 2
    return o


def import_pkg(pkg):
    src = open(os.path.join(HERE, "import_to_blender.py")).read().rsplit("\nmain()", 1)[0]
    argv = sys.argv
    sys.argv = ["blender", "--", pkg, "--no_camera"]
    ns = {"__name__": "imp"}
    exec(compile(src, "import_to_blender.py", "exec"), ns)
    ns["main"]()
    sys.argv = argv
    return bpy.data.collections[os.path.basename(pkg)]


def subtree(ob):
    out = [ob]
    for c in ob.children:
        out += subtree(c)
    return out


def take_item(spec, k):
    """Import a package, keep one actor, freeze it at `frame`, return a root empty over it."""
    pkg, actor, frame = spec.rsplit(":", 2)
    coll = import_pkg(os.path.abspath(pkg))
    sc = bpy.context.scene
    sc.frame_set(int(frame))
    bpy.context.view_layer.update()
    keep, drop = [], []
    for ob in list(coll.objects):
        if ob.parent is None and ob.type == "EMPTY":
            (keep if ob.name.startswith(actor + "/") else drop).append(ob)
    for ob in drop:
        for x in subtree(ob):
            bpy.data.objects.remove(x, do_unlink=True)
    mats = {ob.name: ob.matrix_world.copy() for ob in keep}
    root = bpy.data.objects.new(f"item{k}_{coll.name}_{actor}", None)
    coll.objects.link(root)
    for ob in keep:
        ob.animation_data_clear()
        ob.matrix_world = mats[ob.name]
        ob.parent = root
        ob.matrix_parent_inverse = Matrix.Identity(4)
    bpy.context.view_layer.update()
    meshes = [x for ob in keep for x in subtree(ob) if x.type == "MESH"]
    pts = [m.matrix_world @ Vector(c) for m in meshes for c in m.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return {"root": root, "lo": lo, "hi": hi, "meshes": meshes, "name": coll.name}


def mat(name, rgb, rough=0.9):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1); b.inputs["Roughness"].default_value = rough
    return m


def cyclorama(color, back, y_half=30.0, radius=3.0, height=12.0, front=30.0):
    import bmesh
    prof = [(front, 0.0), (back, 0.0)]
    for k in range(1, 25):
        a = math.pi / 2 * k / 24
        prof.append((back - radius * math.sin(a), radius * (1 - math.cos(a))))
    prof.append((back - radius, height))
    bm = bmesh.new()
    rows = [[bm.verts.new((x, y, z)) for (x, z) in prof] for y in (-y_half, y_half)]
    for i in range(len(prof) - 1):
        bm.faces.new((rows[0][i], rows[0][i + 1], rows[1][i + 1], rows[1][i]))
    me = bpy.data.meshes.new("cyclorama"); bm.to_mesh(me); bm.free()
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new("cyclorama", me)
    bpy.context.scene.collection.objects.link(ob)
    ob.data.materials.append(mat("backdrop", color))


def area(name, loc, target, size, energy, color=(1, 1, 1)):
    ld = bpy.data.lights.new(name, "AREA"); ld.shape = "DISK"; ld.size = size; ld.energy = energy; ld.color = color
    ob = bpy.data.objects.new(name, ld); bpy.context.scene.collection.objects.link(ob)
    ob.location = loc
    ob.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()


def refreeze(items, frames, pkg_root):
    """Re-pose library items at a different opening. The library froze every link at one frame;
    each package's traj.npz still has the pose of every link at EVERY frame, so we just look the
    new frame up (item k gets frames[k % len]). Only the small traj.npz files are read."""
    import re
    import numpy as np
    from mathutils import Quaternion
    for k, it in enumerate(items):
        root = it["root"]
        coll = re.match(r"item\d+_(.+)_(orig|edit)$", root.name).group(1)
        pkg = next(os.path.join(pkg_root, fam, coll) for fam in ("cfH", "cfS")
                   if os.path.isdir(os.path.join(pkg_root, fam, coll)))
        D = np.load(os.path.join(pkg, "traj.npz"))
        names = [str(x) for x in D["body_names"]]
        t = min(frames[k % len(frames)] - 1, D["poses"].shape[0] - 1)   # Blender frame f = step f-1
        for emp in root.children:
            base = re.sub(r"\.\d{3}$", "", emp.name)
            if base not in names:
                continue
            x, y, z, qx, qy, qz, qw = [float(v) for v in D["poses"][t, names.index(base)]]
            emp.matrix_basis = Matrix.Translation((x, y, z)) @ Quaternion((qw, qx, qy, qz)).to_matrix().to_4x4()
        bpy.context.view_layer.update()
        pts = [m.matrix_world @ Vector(c) for m in it["meshes"] for c in m.bound_box]
        it["lo"] = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
        it["hi"] = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
        print(f"[teaser] {coll}: frame {t + 1}")


def animate(items, o, frames_open):
    """Key every link so the whole scene opens from shut to `frames_open` over anim_frames frames.
    Each object starts at its own moment (phases spread by the golden ratio, so neighbours never
    move together) -- render this half and play it forwards+backwards for a seamless open/close loop."""
    import re
    import numpy as np
    from mathutils import Quaternion
    N = o["anim_frames"]
    t0 = o["closed_frame"] - 1
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = 1, N
    for k, it in enumerate(items):
        root = it["root"]
        coll = re.match(r"item\d+_(.+)_(orig|edit)$", root.name).group(1)
        pkg = next(os.path.join(o["pkg_root"], fam, coll) for fam in ("cfH", "cfS")
                   if os.path.isdir(os.path.join(o["pkg_root"], fam, coll)))
        D = np.load(os.path.join(pkg, "traj.npz"))
        names = [str(x) for x in D["body_names"]]
        P = D["poses"]
        t1 = min(frames_open[k % len(frames_open)] - 1, P.shape[0] - 1)
        ph = ((k * 0.6180339887) % 1.0) * o["phase_spread"]
        kids = [(e, names.index(re.sub(r"\.\d{3}$", "", e.name)))
                for e in root.children if re.sub(r"\.\d{3}$", "", e.name) in names]
        for f in range(1, N + 1):
            u = (f - 1) / max(N - 1, 1)
            v = min(max((u - ph) / max(1.0 - o["phase_spread"], 1e-6), 0.0), 1.0)
            v = v * v * (3 - 2 * v)                                  # ease in/out
            t = int(round(t0 + v * (t1 - t0)))
            for emp, j in kids:
                x, y, z, qx, qy, qz, qw = [float(a) for a in P[t, j]]
                emp.matrix_basis = Matrix.Translation((x, y, z)) @ Quaternion((qw, qx, qy, qz)).to_matrix().to_4x4()
                emp.keyframe_insert(data_path="location", frame=f)
                emp.keyframe_insert(data_path="rotation_quaternion", frame=f)
    print(f"[teaser] animation: {N} frames, closed(step {t0}) -> open, phases spread {o['phase_spread']}")


def place_scatter(items, o, rnd, az):
    """Loose still-life: every cabinet gets its own patch of floor. Footprints are circles of
    radius r (half the bbox diagonal); centres keep >= spread*(r_i + r_j)*0.8 apart. Items are
    placed tallest first, each taking the best of ~40 legal candidates: prefer clearance from
    what is already there (even spacing) and a depth that grows with rank (tall ones at the back,
    short ones in front, so nothing hides), plus a little noise so it never looks like a grid."""
    n = len(items)
    R = [0.5 * math.hypot(it["hi"].x - it["lo"].x, it["hi"].y - it["lo"].y) for it in items]
    H = [it["hi"].z - it["lo"].z for it in items]
    area = sum(math.pi * r * r for r in R) * o["spread"] ** 2 * 1.4
    D = math.sqrt(area / o["floor_aspect"]); W = D * o["floor_aspect"]
    ca, sa = math.cos(az), math.sin(az)
    order = sorted(range(n), key=lambda i: -H[i])
    placed = []
    for rank, i in enumerate(order):
        pref = -D / 2 + D * rank / max(n - 1, 1)                 # depth along the view, back -> front
        best, grow = None, 1.0
        while best is None:
            cands = 0
            for _ in range(6000):
                u = rnd.uniform(-D / 2, D / 2) * grow; v = rnd.uniform(-W / 2, W / 2) * grow
                x, y = u * ca - v * sa, u * sa + v * ca            # (depth, lateral) -> world
                clear = min([math.hypot(x - px, y - py) - o["spread"] * 0.8 * (R[i] + pr) for px, py, pr in placed] or [9.0])
                if clear < 0:
                    continue
                score = min(clear, 1.5) - 1.2 * abs(u - pref) / D + rnd.uniform(0, 0.25)
                if best is None or score > best[0]:
                    best = (score, x, y)
                cands += 1
                if cands >= 40:
                    break
            grow *= 1.1
        _, x, y = best
        placed.append((x, y, R[i]))
        it = items[i]
        c = Vector(((it["lo"].x + it["hi"].x) / 2, (it["lo"].y + it["hi"].y) / 2, it["lo"].z))
        yaw = az + math.radians(rnd.uniform(-o["yaw_jitter"], o["yaw_jitter"]))
        it["root"].matrix_world = Matrix.Translation(Vector((x, y, 0.0))) @ Matrix.Rotation(yaw, 4, "Z") @ Matrix.Translation(-c)


def main():
    o = args()
    rnd = random.Random(o["seed"])
    sc = bpy.context.scene
    if o["from_lib"]:
        # the library .blend was opened on the command line: items are the "item*" roots, frozen
        # at their opening, with the bbox stored on them. Keep n of them (library order is already
        # shuffled so any prefix mixes families and sizes), delete the rest.
        roots = sorted((ob for ob in bpy.data.objects if ob.name.startswith("item") and ob.type == "EMPTY"),
                       key=lambda ob: int(ob["order"]))
        n = o["n"] or len(roots)
        for ob in roots[n:]:
            for x in subtree(ob):
                bpy.data.objects.remove(x, do_unlink=True)
        items = [{"root": r, "lo": Vector(r["lo"]), "hi": Vector(r["hi"]), "name": r.name,
                  "meshes": [x for x in subtree(r) if x.type == "MESH"]} for r in roots[:n]]
        if o["open_frames"]:
            refreeze(items, [int(f) for f in o["open_frames"].split(",")], o["pkg_root"])
    else:
        for ob in list(bpy.data.objects):
            bpy.data.objects.remove(ob, do_unlink=True)
        guard = bpy.data.objects.new("guard", bpy.data.lights.new("guard", "POINT"))
        sc.collection.objects.link(guard); guard.data.energy = 0.0
        items = [take_item(s, k) for k, s in enumerate(o["items"])]
        bpy.data.objects.remove(guard, do_unlink=True)
    if o["save_lib"]:
        order = list(range(len(items)))
        rnd.shuffle(order)
        for k, it in zip(order, items):
            it["root"]["lo"], it["root"]["hi"], it["root"]["order"] = list(it["lo"]), list(it["hi"]), k
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(o["save_lib"]))
        print(f"[teaser] library saved: {o['save_lib']}  ({len(items)} objects)")
        return
    # rows grow with the count: 9 -> 3, 15..25 -> 4, 30+ -> 5
    o["rows"] = max(2, math.ceil(0.8 * math.sqrt(len(items)))) if o["rows"] <= 0 else o["rows"]

    # --- still-life layout: tallest at the back, staggered rows, each turned to the camera
    az = math.radians(o["azim"])
    items.sort(key=lambda it: -(it["hi"].z - it["lo"].z))
    if o["layout"] == "scatter":
        place_scatter(items, o, rnd, az)
    else:
        nrow = o["rows"]
        per = [len(items) // nrow + (1 if r < len(items) % nrow else 0) for r in range(nrow)]
        rows, k = [], 0
        for r in range(nrow):
            rows.append(items[k:k + per[r]]); k += per[r]
        x_row = 0.0
        for r, row in enumerate(rows):                            # r = 0 is the back row
            depth = max(it["hi"].x - it["lo"].x for it in row)
            if r > 0:
                prev = max(it["hi"].x - it["lo"].x for it in rows[r - 1])
                x_row += 0.5 * (prev + depth) + 0.45
            widths = [it["hi"].y - it["lo"].y for it in row]
            total = sum(widths) + o["gap"] * (len(row) - 1)
            y = -total / 2 + (0.35 * (widths[0]) * (1 if r % 2 else -1) if r > 0 else 0.0)
            rnd.shuffle(row)
            for it, w in zip(row, [it["hi"].y - it["lo"].y for it in row]):
                c = Vector(((it["lo"].x + it["hi"].x) / 2, (it["lo"].y + it["hi"].y) / 2, it["lo"].z))
                yaw = az + math.radians(rnd.uniform(-12, 12))
                pos = Vector((x_row + rnd.uniform(-0.15, 0.15), y + w / 2, 0.0))
                it["root"].matrix_world = Matrix.Translation(pos) @ Matrix.Rotation(yaw, 4, "Z") @ Matrix.Translation(-c)
                y += w + o["gap"]
    bpy.context.view_layer.update()
    pts = [m.matrix_world @ Vector(c) for it in items for m in it["meshes"] for c in m.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    centre = (lo + hi) / 2

    # --- camera
    cd = bpy.data.cameras.new("cam"); cd.lens = o["focal"]; cd.sensor_fit = "HORIZONTAL"
    cam = bpy.data.objects.new("cam", cd); sc.collection.objects.link(cam); sc.camera = cam
    sc.render.resolution_x, sc.render.resolution_y = o["res"]; sc.render.resolution_percentage = 100
    el = math.radians(o["elev"])
    d = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
    tgt = centre - Vector((0, 0, 0.08 * (hi.z - lo.z)))
    for dist in [2.0 + 0.1 * k for k in range(800)]:
        cam.location = tgt + d * dist
        cam.rotation_euler = (tgt - cam.location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        if all(0.06 < v.x < 0.94 and 0.08 < v.y < 0.92 and v.z > 0
               for v in (world_to_camera_view(sc, cam, p) for p in pts)):
            break

    # --- light: soft warm key high front-left, cool fill right, rim from behind, dim HDRI
    world = bpy.data.worlds.new("w"); sc.world = world; world.use_nodes = True
    nt = world.node_tree
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(os.path.join(os.path.dirname(bpy.app.binary_path),
                                                  f"{bpy.app.version[0]}.{bpy.app.version[1]}",
                                                  "datafiles", "studiolights", "world", "studio.exr"))
    nt.nodes["Background"].inputs["Strength"].default_value = 0.35
    nt.links.new(env.outputs["Color"], nt.nodes["Background"].inputs["Color"])
    span = hi.y - lo.y
    area("key", centre + Vector((3.5, 0.45 * span + 1.5, 6.5)), centre, 5.0, o["key"], (1.0, 0.95, 0.88))
    area("fill", centre + Vector((6.0, -0.5 * span - 2.5, 2.0)), centre, 7.0, 0.25 * o["key"], (0.88, 0.93, 1.0))
    area("rim", centre + Vector((-4.0, -0.2 * span, 5.0)), centre, 4.0, 0.5 * o["key"])
    cyclorama(o["bg"], back=lo.x - 1.5)

    sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"
    sc.cycles.samples = o["samples"]; sc.cycles.use_denoising = True
    sc.view_settings.view_transform = o["view"]
    sc.view_settings.look = "None"
    sc.view_settings.exposure = o["exposure"]
    sc.render.image_settings.file_format = "PNG"
    # transparent variant: the backdrop only catches shadows, the film is transparent
    cyc = bpy.data.objects["cyclorama"]; cyc.is_shadow_catcher = True
    sc.render.film_transparent = True
    sc.render.image_settings.color_mode = "RGBA"
    out = os.path.abspath(o["out"])
    if o["anim_frames"]:
        animate(items, o, [int(f) for f in (o["open_frames"] or "56").split(",")])
        sc.render.filepath = out + "_"
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=out + ".blend")
        bpy.ops.render.render(animation=True)
        print(f"[teaser] wrote {out}_0001..{o['anim_frames']:04d}.png")
        return
    sc.render.filepath = out + ".png"
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=out + ".blend")
    bpy.ops.render.render(write_still=True)
    print(f"[teaser] wrote {out}.png  ({len(items)} objects, view={o['view']})")


main()
