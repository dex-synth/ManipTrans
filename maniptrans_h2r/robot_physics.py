"""Single-hand models and named compatibility signatures from sealed physics."""

from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from .io import digest, sha256
from .physics import OPTION_FIELDS, PHYSICS_FIELDS, equal, verify_sealed

SIGNATURE_SCHEMA = "maniptrans-hand-physics-v2"
REFERENCE_SCHEMA = "hand-reference-v3"


def require_compatibility(value):
    if value.get("schema") != SIGNATURE_SCHEMA or value.get("reference_schema") != REFERENCE_SCHEMA:
        raise ValueError("Re-prepare references and retrain bases with the current wrist/physics schema")


def check_hand_constraints(model, ids):
    """Reject hand constraints not yet covered by the Sharpa signature.

    Object-only constraints do not affect the reusable hand's identity.
    """
    import mujoco
    sites = set(np.flatnonzero(np.isin(model.site_bodyid, ids["body"])))
    hand_tendons = set()
    for tendon in range(model.ntendon):
        start, count = int(model.tendon_adr[tendon]), int(model.tendon_num[tendon])
        for k in range(start, start+count):
            kind, obj = model.wrap_type[k], int(model.wrap_objid[k])
            if ((kind == mujoco.mjtWrap.mjWRAP_JOINT and obj in ids["jnt"])
                    or (kind == mujoco.mjtWrap.mjWRAP_SITE and obj in sites)
                    or (kind in (mujoco.mjtWrap.mjWRAP_SPHERE, mujoco.mjtWrap.mjWRAP_CYLINDER) and obj in ids["geom"])):
                hand_tendons.add(tendon)
    for kind, a, b in zip(model.eq_type, model.eq_obj1id, model.eq_obj2id):
        if kind == mujoco.mjtEq.mjEQ_JOINT:
            selected = ids["jnt"]
        elif kind == mujoco.mjtEq.mjEQ_TENDON:
            selected = hand_tendons
        else:
            selected = ids["body"]
        if a in selected or b in selected:
            raise ValueError("Unsupported hand equality constraint in physics signature")
    if hand_tendons:
        raise ValueError("Unsupported hand tendon in physics signature")
    if any(a in ids["geom"] or b in ids["geom"] for a, b in zip(model.pair_geom1, model.pair_geom2)):
        raise ValueError("Unsupported explicit hand contact pair in physics signature")


def require_clock(task):
    if not np.isclose(task["dt_s"], 1/120, rtol=0, atol=1e-12) or task["control_ticks"] != 2:
        raise ValueError("v2 requires a new sealed task with --physics-hz 120 --control-hz 60")


def hand_ids(model, robot, side):
    root = int(model.joint(robot["hands"][side]["joints"][0]).bodyid[0])
    descendants = {root}
    for i in range(root+1, model.nbody):
        if int(model.body_parentid[i]) in descendants:
            descendants.add(i)
    keep = set(descendants)
    parent = int(model.body_parentid[root])
    while parent:
        if model.body_jntnum[parent]:
            raise ValueError("Shared movable ancestor is unsupported")
        keep.add(parent)
        parent = int(model.body_parentid[parent])
    joints = [model.joint(n).id for n in robot["hands"][side]["joints"]]
    return dict(body=sorted(keep), jnt=joints, dof=[int(model.jnt_dofadr[j]) for j in joints],
                geom=[g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in keep],
                actuator=[a for a in robot["actuator_indices"] if int(model.actuator_trnid[a, 0]) in joints])


def signature(model, robot, side):
    import mujoco
    ids = hand_ids(model, robot, side)
    check_hand_constraints(model, ids)
    object_types = dict(body=mujoco.mjtObj.mjOBJ_BODY, jnt=mujoco.mjtObj.mjOBJ_JOINT,
                        geom=mujoco.mjtObj.mjOBJ_GEOM, actuator=mujoco.mjtObj.mjOBJ_ACTUATOR)
    fields = {}
    for prefix, selected in ids.items():
        for field in PHYSICS_FIELDS:
            if not field.startswith(prefix+"_"):
                continue
            value = np.asarray(getattr(model, field))[selected].copy()
            if field in {"body_parentid", "jnt_bodyid", "geom_bodyid"}:
                fields[field] = [model.body(int(v)).name for v in value]
            elif field == "actuator_trnid":
                fields[field] = [[model.joint(int(v[0])).name, int(v[1])] for v in value]
            else:
                fields[field] = value.tolist()
        if prefix in object_types:
            fields[prefix+"_names"] = [mujoco.mj_id2name(model, object_types[prefix], i) for i in selected]
    fields["qpos_spring"] = [float(model.qpos_spring[int(model.jnt_qposadr[j])]) for j in ids["jnt"]]
    fields["meshes"] = {}
    for g in ids["geom"]:
        m = int(model.geom_dataid[g])
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            v, n = int(model.mesh_vertadr[m]), int(model.mesh_vertnum[m])
            f, nf = int(model.mesh_faceadr[m]), int(model.mesh_facenum[m])
            fields["meshes"][model.geom(g).name] = digest(dict(vertices=model.mesh_vert[v:v+n].tolist(), faces=model.mesh_face[f:f+nf].tolist()))
    exclusions = []
    for pair in model.exclude_signature:
        a, b = int(pair) >> 16, int(pair) & 65535
        if a in ids["body"] and b in ids["body"]:
            exclusions.append(sorted([model.body(a).name, model.body(b).name]))
    fields["exclude"] = sorted(exclusions)
    # Include CPU options even when MJWarp compiles them into kernels or rejects
    # unsupported nondefaults. Different sealed settings must not share a base.
    option_names = (*OPTION_FIELDS, "jacobian", "disableactuator", "noslip_iterations", "noslip_tolerance",
                    "o_margin", "o_solref", "o_solimp", "o_friction", "sdf_initpoints", "sdf_iterations", "sleep_tolerance")
    options = {n: np.asarray(getattr(model.opt, n)).tolist() for n in option_names}
    return dict(schema=SIGNATURE_SCHEMA, robot_physics_id=digest(fields), physics_options_id=digest(options),
                physics_dt_s=1/120, control_dt_s=1/60, decimation=2,
                reference_schema=REFERENCE_SCHEMA, control_schema="virtual-wrist-position-v2",
                feature_schema="maniptrans-h2r-hand-v2", fields=fields, options=options)


def extract_hand(task_path, task, robot, side, output):
    """Expand includes, retain fixed ancestors, and materialize actuator gains."""
    import mujoco
    from .io import read_json, write_json
    require_clock(task)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = task_path / task["scene"]
    model = mujoco.MjModel.from_xml_path(str(source))
    verify_sealed(model, read_json(task_path / "scene/compiled.json"))
    ids = hand_ids(model, robot, side)
    body_names = {model.body(i).name for i in ids["body"]}
    joint_names = {model.joint(i).name for i in ids["jnt"]}
    with tempfile.TemporaryDirectory() as temp:
        expanded = Path(temp) / "expanded.xml"
        mujoco.mj_saveLastXML(str(expanded), model)
        tree = ET.parse(expanded)
    root = tree.getroot()
    root.find("option").set("timestep", repr(task["dt_s"]))
    for name in ("keyframe", "sensor", "size"):
        for child in root.findall(name):
            root.remove(child)
    def prune(parent):
        for child in list(parent):
            if child.tag == "body":
                if child.get("name") not in body_names:
                    parent.remove(child)
                else:
                    prune(child)
            elif parent.tag == "worldbody":
                parent.remove(child)
    prune(root.find("worldbody"))
    # References wholly outside the hand disappear; mixed constraints fail explicitly.
    kept = {e.get("name") for e in root.find("worldbody").iter() if e.get("name")}
    for section in ("tendon", "equality", "contact"):
        parent = root.find(section)
        if parent is None:
            continue
        for node in list(parent):
            refs = [v for e in node.iter() for k, v in e.attrib.items()
                    if k in {"joint", "joint1", "joint2", "body1", "body2", "site", "site1", "site2", "geom", "geom1", "geom2", "tendon1", "tendon2"}]
            included = [r in kept for r in refs]
            if any(included) and not all(included):
                raise ValueError(f"Cross-hand constraint is unsupported: {ET.tostring(node)}")
            if not refs or not any(included):
                parent.remove(node)
            elif node.get("name"):
                kept.add(node.get("name"))
    actuators = root.find("actuator")
    for node in list(actuators):
        name = node.get("name")
        act = model.actuator(name)
        if act.id not in ids["actuator"]:
            actuators.remove(node)
            continue
        if int(act.trntype[0]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            raise ValueError("Current Sharpa mapping requires joint actuators")
        values = dict(name=name, joint=model.joint(int(act.trnid[0])).name,
            gaintype="fixed", biastype="affine", dyntype="none",
            gainprm=" ".join(map(repr, act.gainprm.tolist())), biasprm=" ".join(map(repr, act.biasprm.tolist())),
            dynprm=" ".join(map(repr, act.dynprm.tolist())), gear=" ".join(map(repr, act.gear.tolist())),
            ctrllimited=str(bool(act.ctrllimited[0])).lower(), ctrlrange=" ".join(map(repr, act.ctrlrange.tolist())),
            forcelimited=str(bool(act.forcelimited[0])).lower(), forcerange=" ".join(map(repr, act.forcerange.tolist())))
        if int(act.gaintype[0]) != 0 or int(act.biastype[0]) != 1 or int(act.dyntype[0]) != 0:
            raise ValueError("Unsupported non-affine task actuator")
        node.tag, node.attrib = "general", values
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    for node in root.iter():
        if "file" not in node.attrib:
            continue
        path = Path(node.get("file"))
        path = path if path.is_absolute() else source.parent / path
        if not path.exists():
            raise FileNotFoundError(path)
        name = sha256(path) + path.suffix
        shutil.copy2(path, assets / name)
        node.set("file", f"assets/{name}")
    target = output / "robot_scene.xml"
    tree.write(target, encoding="unicode")
    child_model = mujoco.MjModel.from_xml_path(str(target))
    # Named comparisons tolerate only the serializer's decimal precision.
    new_robot = deepcopy(robot)
    names = robot["hands"][side]["joints"]
    new_robot.update(joint_names=names, sides=[side], hands={side: deepcopy(robot["hands"][side])})
    new_robot["hands"][side]["indices"] = list(range(len(names)))
    for key, field in (("qpos_indices", "qposadr"), ("qvel_indices", "dofadr")):
        new_robot[key] = [int(getattr(child_model.joint(n), field)[0]) for n in names]
    new_robot["lower"] = [float(child_model.joint(n).range[0]) for n in names]
    new_robot["upper"] = [float(child_model.joint(n).range[1]) for n in names]
    new_robot["actuator_indices"] = list(range(child_model.nu))
    new_robot["ctrl_joint_indices"] = [names.index(child_model.joint(int(a[0])).name) for a in child_model.actuator_trnid]
    original = signature(model, robot, side)
    actual = signature(child_model, new_robot, side)
    for field, value in original["fields"].items():
        if field.endswith("_names") or field in {"meshes", "exclude", "body_parentid", "jnt_bodyid", "geom_bodyid", "actuator_trnid"}:
            if value != actual["fields"][field]:
                raise ValueError(f"Single-hand structure differs: {field}")
        else:
            equal(actual["fields"][field], value, field)
    write_json(output / "robot_mapping.json", new_robot)
    write_json(output / "compatibility.json", original)
    return child_model, new_robot, original
