"""Export riddle_pairs_render packages to web assets for the 3D ArticuRiddle viewer.

One GLB per pair (a node per rigid body; meshes stored in the body frame, so node transform =
world pose from traj.npz reproduces the render) plus a keyframes JSON.

    python export_pairs.py OUT_DIR SPEC [SPEC ...]
    SPEC = <family>/<package>            the pair as rendered (orig left, edit right)
         | <family>/<package>:flap       orig drawer on the left; on the right the same drawer
                                         with only its front panel hinged at the top edge,
                                         flipping up (the box stays inside the cabinet)

Extras, both matching arti_manip/confuse:
  * a door re-jointed as a drawer (revolute -> prismatic) gets a drawer box behind its panel
    (bottom, two sides, back; depth 82 % of the cabinet depth, height 92 % of the panel), the same
    rule as drawer_box.py. At q = 0 the box is inside the cabinet, so the closed pair still looks
    identical.
  * the right cabinet is moved GAP metres further out so the two are easier to tell apart.
"""
import json, os, sys
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation

SRC = '/mnt/kostas_home/lruiyao/PartManip_su/riddle_pairs_render'
TEX, GAP, T_BOX = 512, 0.45, 0.012
FRAMES = list(range(15, 106, 2)) + [105]


def shrink(img):
    if img is None:
        return None
    img = img.convert('RGB')
    s = TEX / max(img.size)
    if s < 1:
        img = img.resize((max(1, int(img.size[0] * s)), max(1, int(img.size[1] * s))), Image.LANCZOS)
    return img


def pose_mat(p):
    M = np.eye(4)
    M[:3, :3] = Rotation.from_quat(p[3:7]).as_matrix()
    M[:3, 3] = p[:3]
    return M


def mat_pose(M):
    return np.concatenate([M[:3, 3], Rotation.from_matrix(M[:3, :3]).as_quat()])


def load_visuals(pkg, body):
    out = []
    for v in body['visuals']:
        m = trimesh.load(os.path.join(pkg, v['mesh']), process=False)
        for g in (m.geometry.values() if isinstance(m, trimesh.Scene) else [m]):
            if isinstance(g, trimesh.Trimesh) and len(g.faces):
                g = g.copy()
                g.apply_transform(np.array(v['matrix']))
                mat = getattr(g.visual, 'material', None)
                if mat is not None:
                    if getattr(mat, 'image', None) is not None:
                        mat.image = shrink(mat.image)
                    if getattr(mat, 'baseColorTexture', None) is not None:
                        mat.baseColorTexture = shrink(mat.baseColorTexture)
                out.append(g)
    return out


def plank(lo, hi, material):
    """Axis-aligned box in the body frame with box-projected UVs so a wood texture tiles on it."""
    b = trimesh.creation.box(extents=hi - lo)
    b.apply_translation((lo + hi) / 2)
    b = b.subdivide()  # split faces so each face gets its own vertices for UVs
    b = trimesh.Trimesh(vertices=b.vertices[b.faces].reshape(-1, 3), faces=np.arange(len(b.faces) * 3).reshape(-1, 3), process=False)
    n = np.abs(b.face_normals).argmax(1).repeat(3)
    uv = np.zeros((len(b.vertices), 2))
    for ax in range(3):
        k = n == ax
        a, c = [i for i in range(3) if i != ax]
        uv[k] = b.vertices[k][:, [a, c]] * 1.2
    b.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    return b


def local_box(meshes):
    v = np.concatenate([m.vertices for m in meshes])
    return v.min(0), v.max(0)


def split_panel(meshes, d_loc):
    """Front panel = sub-meshes that are thin along the motion direction and sit at its front."""
    ax = int(np.abs(d_loc).argmax()); sgn = np.sign(d_loc[ax])
    lo, hi = local_box(meshes)
    front = hi[ax] if sgn > 0 else lo[ax]
    panel, box = [], []
    for m in meshes:
        a, b = m.vertices[:, ax].min(), m.vertices[:, ax].max()
        near = min(abs(a - front), abs(b - front)) < 0.02
        (panel if (b - a) < 0.05 and near else box).append(m)
    return panel, box, ax, sgn


def export(spec, out):
    rel, _, mode = spec.partition(':')
    pkg = os.path.join(SRC, rel)
    d = json.load(open(os.path.join(pkg, 'scene.json')))
    t = np.load(os.path.join(pkg, 'traj.npz'))
    poses = t['poses'].astype(np.float64)
    bodies = d['bodies']
    names = [f"{b['actor']}/{b['name']}" for b in bodies]
    assert names == [str(x) for x in t['body_names']], rel
    target = rel.split('-')[-1]                        # e.g. link_2
    info = {'joint_orig': d['joint_orig'], 'joint_edit': d['joint_edit'], 'family': d['family'], 'category': d['category']}

    if mode in ('flap', 'flapdown', 'slide'):
        # right cabinet = a copy of the left one; only the panel (and its handles) gets the new joint
        keep = [i for i, b in enumerate(bodies) if b['actor'] == 'orig']
        dy = poses[0, [i for i, b in enumerate(bodies) if b['actor'] == 'edit'], 1].mean() - poses[0, keep, 1].mean()
        bodies = [bodies[i] for i in keep] + [dict(bodies[i], actor='edit') for i in keep]
        extra = poses[:, keep].copy(); extra[..., 1] += dy
        poses = np.concatenate([poses[:, keep], extra], 1)
        info['joint_edit'] = {'flap': {'type': 'revolute', 'note': 'front panel hinged at its top edge, flips up'},
                              'flapdown': {'type': 'revolute', 'note': 'front panel hinged at its bottom edge, drops down'},
                              'slide': {'type': 'prismatic', 'note': 'door slides sideways along the cabinet front'}}[mode]
        info['family'] = mode

    B = len(bodies)
    edit = [i for i, b in enumerate(bodies) if b['actor'] == 'edit']
    poses[:, edit, 1] += GAP
    rest = poses[0].copy()
    frames = poses[FRAMES].copy()

    geoms = {i: load_visuals(pkg, b) for i, b in enumerate(bodies)}
    extra_nodes = {}   # name -> (meshes, rest pose)

    def moving_of(actor):
        return [i for i, b in enumerate(bodies) if b['actor'] == actor and np.abs(poses[:, i] - poses[0, i]).max() > 1e-5]

    link = next(i for i, b in enumerate(bodies) if b['actor'] == 'edit' and b['name'] == target)
    P0 = pose_mat(rest[link]); P1 = pose_mat(poses[105, link])
    d_world = P1[:3, 3] - P0[:3, 3]

    if mode == '' and d['joint_edit']['type'] == 'prismatic' and d['joint_orig']['type'] == 'revolute':
        # door re-jointed as a drawer: add the box behind the panel
        d_loc = P0[:3, :3].T @ (d_world / np.linalg.norm(d_world))
        ax = int(np.abs(d_loc).argmax()); sgn = np.sign(d_loc[ax])
        lo, hi = local_box(geoms[link])
        up_loc = P0[:3, :3].T @ np.array([0, 0, 1.0]); vax = int(np.abs(up_loc).argmax())
        hax = 3 - ax - vax
        cab = [m.copy().apply_transform(pose_mat(rest[i])) for i, b in enumerate(bodies)
               if b['actor'] == 'edit' and i not in moving_of('edit') for m in geoms[i]]
        cv = np.concatenate([m.vertices for m in cab]); dw = d_world / np.linalg.norm(d_world)
        depth = 0.82 * np.ptp(cv @ dw)
        back = lo[ax] if sgn > 0 else hi[ax]           # inner face of the panel
        h0, h1 = lo[vax], hi[vax]; hh = (h1 - h0) * 0.92; hb = h0 + (h1 - h0) * 0.04
        w0, w1 = lo[hax] + 0.012, hi[hax] - 0.012
        mat = geoms[link][0].visual.material if hasattr(geoms[link][0].visual, 'material') else None
        def span(axis_lo, axis_hi):
            return axis_lo, axis_hi
        planks = []
        def mk(a0, a1, v0, v1, w_0, w_1):
            L, H = np.zeros(3), np.zeros(3)
            L[ax], H[ax] = sorted([a0, a1]); L[vax], H[vax] = v0, v1; L[hax], H[hax] = w_0, w_1
            planks.append(plank(L, H, mat))
        inner = back - sgn * depth
        mk(back, inner, hb, hb + T_BOX, w0, w1)                          # bottom
        mk(back, inner, hb, hb + hh * .8, w0, w0 + T_BOX)               # side
        mk(back, inner, hb, hb + hh * .8, w1 - T_BOX, w1)               # side
        mk(inner, inner + sgn * T_BOX, hb, hb + hh * .8, w0, w1)        # back
        geoms[link] = geoms[link] + planks
        info['box'] = True

    OUT, UP = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])      # cabinet fronts face +x, z is up

    def progress(f):
        # fraction of the original motion reached at frame f (translation or rotation angle)
        if d['joint_orig']['type'] == 'prismatic':
            return np.linalg.norm(poses[f, link, :3] - rest[link, :3]) / max(1e-9, np.linalg.norm(d_world))
        ang = lambda q: Rotation.from_quat(q).magnitude()
        r0 = Rotation.from_quat(rest[link, 3:])
        full = (Rotation.from_quat(poses[105, link, 3:]) * r0.inv()).magnitude()
        return (Rotation.from_quat(poses[f, link, 3:]) * r0.inv()).magnitude() / max(1e-9, full)

    if mode == 'slide':
        # hinged door -> sliding door: step out 2.5 cm, then slide along the front toward the cabinet centre
        pv = np.concatenate([(pose_mat(rest[link]) @ np.c_[m.vertices, np.ones(len(m.vertices))].T).T[:, :3] for m in geoms[link]])
        cab = np.concatenate([(pose_mat(rest[i]) @ np.c_[m.vertices, np.ones(len(m.vertices))].T).T[:, :3]
                              for i, b in enumerate(bodies) if b['actor'] == 'edit' and i not in moving_of('edit') for m in geoms[i]])
        width = np.ptp(pv[:, 1])
        side = np.sign(cab[:, 1].mean() - pv[:, 1].mean()) or 1.0
        hinged = moving_of('edit'); closed = poses[0].copy()
        for fi, f in enumerate(FRAMES):
            fr = np.clip(progress(f), 0, 1)
            off = OUT * 0.025 * min(1, fr / 0.2) + np.array([0, side, 0]) * 0.85 * width * max(0, (fr - 0.2) / 0.8)
            for i in hinged:
                M = pose_mat(closed[i]); M[:3, 3] += off
                frames[fi, i] = mat_pose(M)

    if mode in ('flap', 'flapdown'):
        d_loc = P0[:3, :3].T @ (d_world / np.linalg.norm(d_world))
        panel, box, ax, sgn = split_panel(geoms[link], d_loc)
        up_loc = P0[:3, :3].T @ np.array([0, 0, 1.0]); vax = int(np.abs(up_loc).argmax()); hax = 3 - ax - vax
        geoms[link] = panel
        extra_nodes['edit__boxstatic'] = (box, rest[link].copy())
        # hinge line: top (flap) or bottom (flapdown) edge of the panel's front face, along its width
        plo, phi = local_box(panel)
        want_top = mode == 'flap'
        top = (phi[vax] if up_loc[vax] > 0 else plo[vax]) if want_top else (plo[vax] if up_loc[vax] > 0 else phi[vax])
        frontc = phi[ax] if sgn > 0 else plo[ax]
        pt = np.zeros(3); pt[ax] = frontc; pt[vax] = top; pt[hax] = (plo[hax] + phi[hax]) / 2
        hinge_p = (P0 @ np.append(pt, 1))[:3]
        e = np.zeros(3); e[hax] = 1; hinge_u = P0[:3, :3] @ e
        dw = d_world / np.linalg.norm(d_world)
        free_edge = -UP if want_top else UP                                # the edge that swings out
        if np.dot(np.cross(hinge_u, free_edge), dw) < 0:
            hinge_u = -hinge_u
        swing = np.radians(95 if want_top else 88)
        hinged = [i for i in moving_of('edit')]                          # panel + handles
        closed = poses[0].copy()
        for fi, f in enumerate(FRAMES):
            # same timing as the original drawer: fraction of its travel -> fraction of 95 deg
            frac = progress(f)
            R = np.eye(4); R[:3, :3] = Rotation.from_rotvec(hinge_u * swing * frac).as_matrix()
            T = np.eye(4); T[:3, 3] = hinge_p; Ti = np.eye(4); Ti[:3, 3] = -hinge_p
            for i in hinged:
                frames[fi, i] = mat_pose(T @ R @ Ti @ pose_mat(closed[i]))

    moving = [i for i in range(B) if np.abs(frames[:, i] - rest[i]).max() > 1e-6]
    scene = trimesh.Scene()
    nodes = [f"{b['actor']}__{b['name']}" for b in bodies]
    for i, node in enumerate(nodes):
        scene.graph.update(frame_from=scene.graph.base_frame, frame_to=node, matrix=np.eye(4))
        for k, g in enumerate(geoms[i]):
            scene.add_geometry(g, node_name=f"{node}__{k}", geom_name=f"{node}__{k}", parent_node_name=node)
    rest_list = [rest[i] for i in range(B)]
    for node, (ms, p) in extra_nodes.items():
        scene.graph.update(frame_from=scene.graph.base_frame, frame_to=node, matrix=np.eye(4))
        for k, g in enumerate(ms):
            scene.add_geometry(g, node_name=f"{node}__{k}", geom_name=f"{node}__{k}", parent_node_name=node)
        nodes.append(node); rest_list.append(p)

    slug = rel.replace('/', '_') + (f'_{mode}' if mode else '')
    open(os.path.join(out, slug + '.glb'), 'wb').write(scene.export(file_type='glb'))
    kf = dict(info, bodies=nodes, rest=np.round(rest_list, 5).tolist(), moving=moving,
              frames=[np.round(fr[moving], 5).tolist() for fr in frames])
    json.dump(kf, open(os.path.join(out, slug + '.json'), 'w'), separators=(',', ':'))
    print(slug, f"{os.path.getsize(os.path.join(out, slug + '.glb'))/1e6:.2f}MB", 'moving', [nodes[i] for i in moving],
          info['joint_orig']['type'], '->', info['joint_edit']['type'], 'box' if info.get('box') else '', flush=True)


if __name__ == '__main__':
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    for spec in sys.argv[2:]:
        export(spec, out)
