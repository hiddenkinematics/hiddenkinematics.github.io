"""Run before make_teaser_still.py on teaser_lib.blend: give every door that was re-jointed as a
drawer (cfS/*_drawer_*, edit actor) a drawer box behind its panel, parented to the moving link so
it slides with it. Same rule as arti_manip/confuse/drawer_box.py: depth 82 % of the cabinet
depth along the slide, bottom + two sides + back, sides/back 80 % of the panel height.
At q = 0 everything is inside the cabinet, so closed cabinets look unchanged.

    blender -b teaser_lib.blend --python add_drawer_boxes.py --python make_teaser_still.py -- ...
"""
import os, re
import bpy, bmesh
import numpy as np
from mathutils import Matrix, Vector, Quaternion

ROOT = '/mnt/kostas_home/lruiyao/PartManip_su/riddle_pairs_render'
T = 0.012


def subtree(ob):
    out = [ob]
    for c in ob.children:
        out += subtree(c)
    return out


def plank(name, lo, hi, mat, parent):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = Vector(((lo[i] + hi[i]) / 2 + v.co[i] * (hi[i] - lo[i]) for i in range(3)))
    uv = bm.loops.layers.uv.new('UVMap')
    for f in bm.faces:
        ax = max(range(3), key=lambda i: abs(f.normal[i]))
        a, c = [i for i in range(3) if i != ax]
        for l in f.loops:
            l[uv].uv = (l.vert.co[a] * 1.2, l.vert.co[c] * 1.2)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    if mat is not None:
        me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    for c in parent.users_collection:
        c.objects.link(ob)
    ob.parent = parent
    ob.matrix_parent_inverse = Matrix.Identity(4)
    return ob


n_added = 0
for root in [ob for ob in bpy.data.objects if ob.name.startswith('item') and ob.type == 'EMPTY']:
    m = re.match(r"item\d+_(.+)_(orig|edit)$", root.name)
    if not m or m.group(2) != 'edit':
        continue
    coll = m.group(1)
    pkg = os.path.join(ROOT, 'cfS', coll)
    if '_drawer_' not in coll or not os.path.isdir(pkg):
        continue
    target = 'edit/' + coll.split('-')[-1]
    D = np.load(os.path.join(pkg, 'traj.npz'))
    names = [str(x) for x in D['body_names']]
    bi = names.index(target)
    p0, p1 = D['poses'][0, bi].astype(float), D['poses'][105, bi].astype(float)
    R0 = Quaternion((p0[6], p0[3], p0[4], p0[5])).to_matrix()
    dw = Vector(p1[:3] - p0[:3]).normalized()
    d_loc = R0.transposed() @ dw
    up_loc = R0.transposed() @ Vector((0, 0, 1))
    ax = max(range(3), key=lambda i: abs(d_loc[i])); sgn = 1 if d_loc[ax] > 0 else -1
    vax = max(range(3), key=lambda i: abs(up_loc[i])); hax = 3 - ax - vax

    emp = next(e for e in root.children if re.sub(r"\.\d{3}$", "", e.name) == target)
    meshes = [x for x in subtree(emp) if x.type == 'MESH']
    inv = emp.matrix_world.inverted()
    pts = [inv @ (mm.matrix_world @ Vector(c)) for mm in meshes for c in mm.bound_box]
    lo = [min(p[i] for p in pts) for i in range(3)]; hi = [max(p[i] for p in pts) for i in range(3)]

    # cabinet depth along the slide: every mesh of this item not under the moving link, world space
    others = [x for x in subtree(root) if x.type == 'MESH' and x not in meshes and not any(
        x in subtree(e) for e in root.children if re.sub(r"\.\d{3}$", "", e.name).startswith('edit/handle'))]
    wd = emp.matrix_world.to_3x3() @ (d_loc * (1 if True else 1))
    wd.normalize()
    proj = [wd.dot(o.matrix_world @ Vector(c)) for o in others for c in o.bound_box]
    depth = 0.82 * (max(proj) - min(proj))

    back = lo[ax] if sgn > 0 else hi[ax]
    inner = back - sgn * depth
    h0, h1 = lo[vax], hi[vax]; hh = (h1 - h0) * 0.92; hb = h0 + (h1 - h0) * 0.04
    if up_loc[vax] < 0:                      # vertical axis points down in the link frame
        hb = h1 - (h1 - h0) * 0.04; hh = -hh
    w0, w1 = lo[hax] + 0.012, hi[hax] - 0.012
    mat = next((sl.material for mm in meshes for sl in mm.material_slots if sl.material), None)

    def mk(tag, a0, a1, v0, v1, x0, x1):
        L, H = [0, 0, 0], [0, 0, 0]
        L[ax], H[ax] = sorted([a0, a1]); L[vax], H[vax] = sorted([v0, v1]); L[hax], H[hax] = sorted([x0, x1])
        plank(f"{emp.name}_box_{tag}", L, H, mat, emp)

    mk('bottom', back, inner, hb, hb + (T if hh > 0 else -T), w0, w1)
    mk('sideA', back, inner, hb, hb + hh * .8, w0, w0 + T)
    mk('sideB', back, inner, hb, hb + hh * .8, w1 - T, w1)
    mk('back', inner, inner + sgn * T, hb, hb + hh * .8, w0, w1)
    n_added += 1
    print(f"[boxes] {root.name}: depth {depth:.2f} m, panel {h1 - h0:.2f} x {w1 - w0:.2f}")
print(f"[boxes] added drawer boxes to {n_added} items")
