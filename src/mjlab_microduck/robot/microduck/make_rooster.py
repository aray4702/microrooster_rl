#!/usr/bin/env python3
"""Generate the Micro Rooster phase-1 model from the Microduck walk model.

    uv run python src/mjlab_microduck/robot/microduck/make_rooster.py

Writes robot_rooster.xml + scene_rooster.xml next to this file (so the duck's
mesh assets resolve) and prints the mass budget and standing height.

What changes vs robot_walk.xml (rough sim prototype, not a CAD export):
  - 12 servos instead of 14: head_yaw and head_roll are welded (joint +
    actuator removed). The jaw servo, unmodelled as a joint in the walk model,
    is dropped too. Neck keeps neck_pitch + head_pitch (enough to peck and bob).
  - Servo mass: every kept XL330 (~18 g) becomes a Feetech STS3215 (~55 g);
    removed servos subtract their 18 g. Added as point masses at the servo
    housing positions, so body CoM/inertia move the right way.
  - Legs: thigh and shin link offsets stretched by LEG_SCALE (meshes are not
    stretched — expect visual gaps at the knee and ankle).
  - Rooster parts: comb, wattle and tail as visual-only geoms whose mass is
    folded into the parent body's inertia.

Every number below is a phase-1 guess to be replaced by CAD + scale readings.
"""

from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
SRC_XML = HERE / "robot_walk.xml"
OUT_ROBOT = HERE / "robot_rooster.xml"
OUT_SCENE = HERE / "scene_rooster.xml"

LEG_SCALE = 1.15
XL330_MASS = 0.018
STS3215_MASS = 0.055
REMOVED_JOINTS = ("head_yaw", "head_roll")
# Bodies whose XL330 housings belong to removed servos (head_yaw servo,
# head_roll servo, jaw servo).
REMOVED_SERVO_BODIES = ("yaw_roll_motion", "jaw_soft")
# Thigh and shin links: the child body's offset is the link vector.
LEG_LINK_BODIES = ("leg", "ankle_left", "leg_2", "ankle_right")

COMB_MASS = 0.012
WATTLE_MASS = 0.004
TAIL_MASS = 0.025
RED = (0.80, 0.08, 0.06, 1.0)
TAIL_RGBA = (0.10, 0.22, 0.12, 1.0)

# Same STAND joint angles as the duck (scene_walk.xml), minus the removed joints.
STAND_JOINTS = {
    "left_hip_yaw": 0.0, "left_hip_roll": -0.0873, "left_hip_pitch": -0.4579,
    "left_knee": -0.0049, "left_ankle": 0.4530,
    "neck_pitch": 0.3491, "head_pitch": 0.3491,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0873, "right_hip_pitch": 0.4579,
    "right_knee": 0.0049, "right_ankle": -0.4530,
}


def box_inertia(mass, size):
    """Diagonal inertia of a solid box given half-sizes."""
    x, y, z = (2 * s for s in size)
    return mass / 12.0 * np.array([y * y + z * z, x * x + z * z, x * x + y * y])


def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


def mat_to_quat(r):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, r.flatten())
    return q


def add_masses(model, body_id, parts):
    """Fold extra masses into a body's explicit inertial.

    parts: list of (mass, pos_in_body_frame, diag_inertia_in_body_axes).
    Negative masses remove material (point mass, no rotational term).
    Returns (mass, ipos, iquat, diag_inertia).
    """
    m0 = model.body_mass[body_id]
    c0 = model.body_ipos[body_id].copy()
    r0 = quat_to_mat(model.body_iquat[body_id])
    i0 = r0 @ np.diag(model.body_inertia[body_id]) @ r0.T

    masses = [m0] + [p[0] for p in parts]
    coms = [c0] + [np.asarray(p[1], float) for p in parts]
    inertias = [i0] + [np.diag(p[2]) for p in parts]

    total = sum(masses)
    assert total > 0, f"body {body_id} mass went non-positive"
    com = sum(m * c for m, c in zip(masses, coms)) / total
    tensor = np.zeros((3, 3))
    for m, c, i in zip(masses, coms, inertias):
        d = c - com
        tensor += i + m * (d @ d * np.eye(3) - np.outer(d, d))
    evals, evecs = np.linalg.eigh(tensor)
    assert np.all(evals > 0), f"body {body_id} inertia not positive definite: {evals}"
    if np.linalg.det(evecs) < 0:
        evecs[:, 0] *= -1
    return total, com, mat_to_quat(evecs), evals


def servo_positions(model, body_id):
    """Body-frame positions of the visual XL330 housings in a body."""
    out = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body_id or model.geom_dataid[g] < 0:
            continue
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_contype[g] != 0:
            continue
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[g]) == "xl330":
            out.append(model.geom_pos[g].copy())
    return out


def subtree_visual_aabb(model, data, root_name, own_only=False):
    """World AABB of the visual geoms in a subtree (or one body), from mesh vertices."""
    root = model.body(root_name).id
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(model.ngeom):
        b = model.geom_bodyid[g]
        while b not in (0, root) and not own_only:
            b = model.body_parentid[b]
        if b != root or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[g]
        start, n = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        verts = model.mesh_vert[start:start + n]
        world = verts @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        lo, hi = np.minimum(lo, world.min(0)), np.maximum(hi, world.max(0))
    return lo, hi


def world_to_body(data, body_id, p):
    r = data.xmat[body_id].reshape(3, 3)
    return r.T @ (np.asarray(p) - data.xpos[body_id])


def stand_qpos(model, z):
    qpos = np.zeros(model.nq)
    qpos[2], qpos[3] = z, 1.0
    for name, val in STAND_JOINTS.items():
        qpos[model.jnt_qposadr[model.joint(name).id]] = val
    return qpos


def foot_bottom_z(model, data):
    lo, _ = subtree_visual_aabb(model, data, "ankle_left")
    lo2, _ = subtree_visual_aabb(model, data, "ankle_right")
    return min(lo[2], lo2[2])


def main():
    spec = mujoco.MjSpec.from_file(str(SRC_XML))
    duck = spec.compile()
    duck_mass = duck.body_subtreemass[1]

    # 1. 12 servos: weld head_yaw / head_roll.
    for name in REMOVED_JOINTS:
        spec.delete(spec.actuator(name))
        spec.delete(spec.joint(name))

    # 2. Longer legs.
    for name in LEG_LINK_BODIES:
        body = spec.body(name)
        body.pos = np.asarray(body.pos) * LEG_SCALE

    # Compile once to measure geometry at the STAND pose.
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[:] = stand_qpos(model, 0.2)
    mujoco.mj_forward(model, data)

    head_body = model.body("jaw_soft").id
    trunk_body = model.body("trunk_base").id
    head_lo, head_hi = subtree_visual_aabb(model, data, "yaw_roll_motion")
    trunk_lo, trunk_hi = subtree_visual_aabb(model, data, "trunk_base", own_only=True)

    # 3. Rooster parts, positioned from the measured head and trunk boxes
    #    (+x is forward, +z up at STAND).
    head_len = head_hi[0] - head_lo[0]
    comb_parts = []
    for i, (dx, h) in enumerate(((0.30, 0.016), (0.50, 0.022), (0.70, 0.018))):
        p_world = [head_lo[0] + dx * head_len, 0.0, head_hi[2] + h * 0.5]
        comb_parts.append((f"comb_{i}", "ellipsoid", (0.009, 0.003, h * 0.6),
                           world_to_body(data, head_body, p_world), COMB_MASS / 3))
    wattle_world = [head_hi[0] - 0.25 * head_len, 0.0, head_lo[2] + 0.004]
    wattle = ("wattle", "ellipsoid", (0.006, 0.004, 0.011),
              world_to_body(data, head_body, wattle_world), WATTLE_MASS)

    tail_parts = []
    angles = np.deg2rad([-25, -12, 0, 12, 25])
    root = np.array([trunk_lo[0] + 0.005, 0.0, trunk_hi[2] - 0.01])
    feather_half = np.array([0.003, 0.006, 0.040])  # thin, tall feather
    for i, a in enumerate(angles):
        # Fan in the y-z plane, raked back 30 degrees.
        direction = np.array([-np.sin(np.deg2rad(30)), np.sin(a), np.cos(a)])
        direction /= np.linalg.norm(direction)
        center = root + direction * feather_half[2]
        tail_parts.append((f"tail_{i}", "box", tuple(feather_half),
                           world_to_body(data, trunk_body, center), TAIL_MASS / len(angles),
                           direction))

    # Visual geoms (no collision, no mass of their own: mass is folded below).
    def add_geom(body_name, name, gtype, size, pos, rgba, axis=None):
        geom = spec.body(body_name).add_geom()
        geom.name = f"rooster_{name}"
        geom.type = getattr(mujoco.mjtGeom, f"mjGEOM_{gtype.upper()}")
        geom.size = list(size)
        geom.pos = list(pos)
        geom.rgba = list(rgba)
        geom.contype = geom.conaffinity = 0
        geom.group = 2
        geom.mass = 0.0
        # Sizes are meant in world axes at STAND (z up), but the exported body
        # frames are rotated, so orient each geom explicitly.
        body_r = data.xmat[model.body(body_name).id].reshape(3, 3)
        q = np.zeros(4)
        if axis is None:
            q = mat_to_quat(body_r.T)
        else:
            mujoco.mju_quatZ2Vec(q, body_r.T @ axis)
        geom.quat = list(q)

    for name, gtype, size, pos, _ in comb_parts + [wattle]:
        add_geom("jaw_soft", name, gtype, size, pos, RED)
    for name, gtype, size, pos, _, axis in tail_parts:
        add_geom("trunk_base", name, gtype, size, pos, TAIL_RGBA, axis)

    # 4. Mass bookkeeping per body: servo swap + rooster parts.
    extra = {}
    for b in range(1, model.nbody):
        name = model.body(b).name
        dm = -XL330_MASS if name in REMOVED_SERVO_BODIES else STS3215_MASS - XL330_MASS
        for p in servo_positions(model, b):
            extra.setdefault(name, []).append((dm, p, np.zeros(3)))
    for _, _, size, pos, m in comb_parts + [wattle]:
        extra.setdefault("jaw_soft", []).append((m, pos, box_inertia(m, size)))
    for _, _, size, pos, m, _ in tail_parts:
        extra.setdefault("trunk_base", []).append((m, pos, box_inertia(m, size)))

    for name, parts in extra.items():
        mass, ipos, iquat, inertia = add_masses(model, model.body(name).id, parts)
        body = spec.body(name)
        body.explicitinertial = True
        body.fullinertia = [np.nan] * 6  # export uses fullinertia; we write the diagonal form
        body.mass, body.ipos, body.iquat, body.inertia = mass, ipos, iquat, inertia

    rooster = spec.compile()
    n_servo_kept = sum(len(servo_positions(model, b)) for b in range(1, model.nbody)
                       if model.body(b).name not in REMOVED_SERVO_BODIES)

    # 5. Standing height: feet flat on the floor at STAND.
    data = mujoco.MjData(rooster)
    data.qpos[:] = stand_qpos(rooster, 0.2)
    mujoco.mj_forward(rooster, data)
    stand_z = 0.2 - foot_bottom_z(rooster, data) + 0.001
    data.qpos[:] = stand_qpos(rooster, stand_z)
    mujoco.mj_forward(rooster, data)
    com = data.subtree_com[1]
    head_top = subtree_visual_aabb(rooster, data, "yaw_roll_motion")[1][2]

    OUT_ROBOT.write_text(spec.to_xml())
    write_scene(rooster, stand_z)

    print(f"duck mass        {duck_mass * 1000:7.1f} g   (14 XL330)")
    print(f"rooster mass     {rooster.body_subtreemass[1] * 1000:7.1f} g   "
          f"({rooster.nu} actuators, {n_servo_kept} STS3215 housings)")
    print(f"actuated joints  {[rooster.actuator(i).name for i in range(rooster.nu)]}")
    print(f"stand trunk z    {stand_z * 1000:7.1f} mm  (duck: 120.0)")
    print(f"CoM height       {com[2] * 1000:7.1f} mm  (duck: 141.7)")
    print(f"top of comb      {max(head_top, 0) * 1000:7.1f} mm")
    print(f"wrote {OUT_ROBOT.name}, {OUT_SCENE.name}")


def write_scene(model, stand_z):
    qpos = " ".join(f"{v:.6g}" for v in stand_qpos(model, stand_z))
    ctrl = " ".join(f"{STAND_JOINTS[model.actuator(i).name]:.6g}" for i in range(model.nu))
    OUT_SCENE.write_text(f"""<mujoco model="rooster_scene">
    <include file="{OUT_ROBOT.name}" />

    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />
        <rgba haze="0.15 0.25 0.35 1" />
        <global azimuth="160" elevation="-20" />
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512"
            height="3072" />
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
            rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300" />
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5"
            reflectance="0.2" />
    </asset>

    <worldbody>
        <light pos="0 0 3.5" dir="0 0 -1" directional="true" />
        <geom name="floor" size="0 0 0.05" pos="0 0 0" type="plane" material="groundplane" />
    </worldbody>

    <keyframe>
        <key name="STAND" qpos="{qpos}" ctrl="{ctrl}" />
    </keyframe>
</mujoco>
""")


if __name__ == "__main__":
    main()
