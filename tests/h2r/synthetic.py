"""A small sealed task written with the vendored loader's sealing functions: a toy hand with the
Sharpa link names, a free box with a hinged lid and a table. No dataset is needed."""

import json

import mujoco
import numpy as np

from maniptrans_h2r.h2r_io import demo as demo_io, robot as robot_io, task as task_io
from maniptrans_h2r.robot import FINGERS, tracked_links

NOMINAL_DT_S = 1/300
BOX = "".join(f"v {x} {y} {z}\n" for x in (-.02, .02) for y in (-.02, .02) for z in (-.02, .02)) + \
    "f 1 2 4 3\nf 5 7 8 6\nf 1 5 6 2\nf 3 4 8 7\nf 1 3 7 5\nf 2 6 8 4\n"
WRIST = (("x", "slide", "1 0 0", "-1 1"), ("y", "slide", "0 1 0", "-1 1"), ("z", "slide", "0 0 1", "-1 1"),
         ("roll", "hinge", "1 0 0", "-6.2 6.2"), ("pitch", "hinge", "0 1 0", "-6.2 6.2"), ("yaw", "hinge", "0 0 1", "-6.2 6.2"))


def hand_xml(side, offset):
    """Six virtual wrist joints, a palm and five three-joint fingers ending in a fixed fingertip."""
    links, fingers = tracked_links("sharpa_wave"), ""
    for index, finger in enumerate(FINGERS):
        chain = f'<body name="{side}_{finger}_fingertip" pos=".02 0 0"><geom size=".004" mass=".001"/></body>'
        for depth, link in reversed(list(enumerate(links[finger][:3]))):
            chain = (f'<body name="{side}_{finger}_{link}" pos="{.03 if depth else .05} {.02*(index-2) if not depth else 0} 0">'
                     f'<joint name="{side}_{finger}_{link}_joint" axis="0 1 0" range="-.2 1.5"/>'
                     f'<geom name="{side}_{finger}_{link}_geom" type="capsule" size=".005" fromto="0 0 0 .02 0 0" mass=".01"/>{chain}</body>')
        fingers += chain
    body = f'<body name="{side}_palm"><geom name="{side}_palm_geom" type="box" size=".04 .04 .01" mass=".2"/>{fingers}</body>'
    for name, kind, axis, limits in reversed(WRIST):
        body = (f'<body name="{side}_root_{name}"><inertial pos="0 0 0" mass=".01" diaginertia="1e-5 1e-5 1e-5"/>'
                f'<joint name="{side}_{name}_joint" type="{kind}" axis="{axis}" range="{limits}"/>{body}</body>')
    return f'<body name="{side}_base" pos="0 {offset} .3">{body}</body>'


def joints(side):
    wrist = [f"{side}_{name}_joint" for name, *_ in WRIST]
    links = tracked_links("sharpa_wave")
    return wrist, [f"{side}_{finger}_{link}_joint" for finger in FINGERS for link in links[finger][:3]]


def scene_xml(sides, mesh):
    names = [name for side in sides for group in joints(side) for name in group]
    return f'''<mujoco>
  <option timestep="{NOMINAL_DT_S!r}"/>
  <asset><mesh name="box" file="{mesh}"/></asset>
  <worldbody>
    <geom name="table" type="box" size=".5 .5 .02" pos="0 0 -.02"/>
    {"".join(hand_xml(side, .3 if side == "left" else -.3) for side in sides)}
    <body name="object_bottom" pos="0 0 .03"><freejoint name="object_bottom_free"/>
      <geom name="bottom_visual" type="mesh" mesh="box" contype="0" conaffinity="0" mass="0"/>
      <geom name="bottom_collision" type="box" size=".02 .02 .02" mass=".1"/>
      <body name="object_top" pos="0 0 .04"><joint name="object_lid" axis="0 1 0" range="0 1.5"/>
        <geom name="top_visual" type="mesh" mesh="box" contype="0" conaffinity="0" mass="0"/>
        <geom name="top_collision" type="box" size=".02 .02 .01" mass=".05"/></body></body>
  </worldbody>
  <actuator>{"".join(f'<position name="{name}_ctrl" joint="{name}" kp="20" kv="1"/>' for name in names)}</actuator>
</mujoco>'''


def write_task(root, sides=("left", "right"), robot_name="sharpa_wave", frames=31):
    """A standalone task directory; returns its path."""
    demo, robot = root / "demo", root / "robot"
    (demo / "meshes").mkdir(parents=True)
    (demo / "meshes/box.obj").write_text(BOX)
    (demo / "scene.xml").write_text("<mujoco/>")
    unit = np.array([1., 0., 0., 0.])
    positions = np.zeros((frames, 2, 3))
    positions[:, 0, 2], positions[:, 1, 2] = .03, .07
    arrays = {"time_s": np.arange(frames)/30., "source_time_s": np.arange(frames)/30.,
              "part_pos_m": positions, "part_quat_wxyz": np.tile(unit, (frames, 2, 1)),
              "joint_lid_rad": np.linspace(0., 1., frames)}
    for side in sides:
        arrays.update({f"{side}_joints_m": np.zeros((frames, 21, 3)), f"{side}_vertices_m": np.zeros((frames, 778, 3)),
                       f"{side}_faces": np.zeros((2, 3), dtype=np.int32), f"{side}_mano_quat": np.tile(unit, (frames, 16, 1)),
                       f"{side}_mano_trans": np.zeros((frames, 3)), f"{side}_mano_shape": np.zeros(10)})
    np.savez_compressed(demo / "reference.npz", **arrays)
    (demo / "reference.json").write_text(json.dumps({
        "dataset": "synthetic", "profile": "clip", "task": {"hand_sides": list(sides)},
        "reference_hz": 30., "duration_s": (frames-1)/30., "mano": {"convention": "h2r-mano-v1"},
        "quaternion_convention": "wxyz-unit-continuous-v1", "object_geometry": {"part_order": ["bottom", "top"]}}))
    (demo / "scene.json").write_text(json.dumps({
        "interaction_objects": {"box": {
            "root": {"part": "bottom", "body": "object_bottom", "joint": "object_bottom_free", "motion": "free"},
            "parts": {"bottom": "object_bottom", "top": "object_top"},
            "joints": {"lid": {"type": "hinge", "parent": "bottom", "child": "top", "axis": [0, 1, 0],
                               "mjcf_name": "object_lid", "reference_key": "joint_lid_rad"}}}},
        "scoring_parts": ["bottom", "top"], "environment": {"table": {"shape": "box"}},
        "settings": {"timestep_s": NOMINAL_DT_S, "gravity_m_s2": [0., 0., -9.81]}}))
    demo_io.seal_demo(demo, input_key="recipe")

    (robot / "assets").mkdir(parents=True)
    for name in ("robot.xml", "assets/robot.urdf"):
        (robot / name).write_text("<robot/>")
    names = [name for side in sides for group in joints(side) for name in group]
    model = mujoco.MjModel.from_xml_string(scene_xml(sides, str(demo / "meshes/box.obj")))
    robot_bodies = [model.body(i).name for i in range(1, model.nbody) if not model.body(i).name.startswith("object_")]
    (robot / "semantics.json").write_text(json.dumps({
        "joint_names": names, "joint_units": ["m" if "_x_" in n or "_y_" in n or "_z_" in n else "rad" for n in names],
        "actuator_names": [f"{n}_ctrl" for n in names], "actuator_joint_names": names, "body_names": robot_bodies,
        "hands": {side: {"wrist_body": f"{side}_palm", "wrist_translation_joints": joints(side)[0][:3],
                         "wrist_rotation_joints": joints(side)[0][3:], "finger_joints": joints(side)[1]} for side in sides}}))
    (robot / "compiled.json").write_text("{}")
    robot_io.seal_robot(robot, robot_name, sides, mjcf="robot.xml", urdf="assets/robot.urdf",
                        hand_urdfs={side: "assets/robot.urdf" for side in sides})

    (root / "scene.xml").write_text(scene_xml(sides, "demo/meshes/box.obj"))
    # Nothing beyond the solver options is audited in this toy scene.
    (root / "compiled.json").write_text(json.dumps({
        "robot": {"joints": {}, "bodies": {}, "geoms": {}},
        "base_scene": {"options": {"timestep": NOMINAL_DT_S}, "geoms": {}, "parts": {}, "joints": {}}}))
    task_io.seal_task(root, "demo", "robot")
    return root
