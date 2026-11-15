"""Load task physics unchanged, adding only observation sensors."""

import xml.etree.ElementTree as ET

import numpy as np

from .clock import PHYSICS_DT_S
from .io import sha256

CAPACITY = 64
# The method's own pre-check of reset and initial poses. The evaluation's admission threshold is
# an evaluation setting and is not given to methods.
MAX_PENETRATION_M = 0.02
PHYSICS_FIELDS = (
    "body_parentid", "body_pos", "body_quat", "body_mass", "body_ipos", "body_iquat", "body_inertia", "body_gravcomp",
    "jnt_type", "jnt_bodyid", "jnt_axis", "jnt_pos", "jnt_range", "jnt_limited", "jnt_actfrclimited",
    "jnt_actfrcrange", "jnt_actgravcomp", "dof_damping", "dof_armature", "dof_frictionloss",
    "jnt_solref", "jnt_solimp", "jnt_margin", "dof_solref", "dof_solimp",
    "geom_type", "geom_bodyid", "geom_pos", "geom_quat", "geom_size", "geom_contype", "geom_conaffinity",
    "geom_friction", "geom_condim", "geom_solref", "geom_solimp", "geom_margin", "geom_gap", "geom_priority", "geom_solmix",
    "actuator_trnid", "actuator_trntype", "actuator_gaintype", "actuator_biastype", "actuator_dyntype",
    "actuator_gainprm", "actuator_biasprm", "actuator_dynprm", "actuator_gear",
    "actuator_ctrllimited", "actuator_ctrlrange", "actuator_forcelimited", "actuator_forcerange",
    "actuator_actrange", "actuator_actlimited", "actuator_actearly", "actuator_lengthrange", "geom_fluid",
    "qpos_spring", "jnt_stiffness", "exclude_signature",
)
OPTION_FIELDS = ("timestep", "gravity", "integrator", "solver", "cone", "iterations", "ls_iterations",
                 "tolerance", "ls_tolerance", "impratio", "disableflags", "enableflags",
                 "ccd_iterations", "ccd_tolerance", "density", "viscosity", "wind", "magnetic")


def equal(actual, expected, name, rtol=2e-5, atol=1e-9):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise ValueError(f"Task physics mismatch: {name} ({actual.shape} vs {expected.shape})")


def load_model(task, robot):
    """The sealed task scene with observation sensors, stepped at the method's training step."""
    import mujoco

    source = task.scene_xml
    nominal = mujoco.MjModel.from_xml_path(str(source))
    verify_sealed(nominal, task.compiled)
    tree = ET.parse(source)
    for node in tree.iter():
        if "file" in node.attrib:
            node.set("file", str((source.parent / node.get("file")).resolve()))
    sensors = tree.find("sensor")
    if sensors is None:
        sensors = ET.SubElement(tree.getroot(), "sensor")
    bodies = list(dict.fromkeys([b for h in robot["hands"].values() for b in h["bodies"]]
                               + list(task.object_binding["parts"].values())))
    for i, body in enumerate(bodies):
        for tag in ("framepos", "framequat", "framelinvel", "frameangvel"):
            ET.SubElement(sensors, tag, name=f"mt_{tag}_{i}", objtype="xbody", objname=body)
    pairs = []
    for side in robot["sides"]:
        for tip, body in enumerate(robot["hands"][side]["contact_bodies"]):
            # Net force on each fingertip from every partner, including the robot itself.
            # Exact geom selectors avoid including a fingertip body's descendants.
            for geom in range(nominal.ngeom):
                if (nominal.geom_bodyid[geom] != nominal.body(body).id
                        or not (nominal.geom_contype[geom] or nominal.geom_conaffinity[geom])):
                    continue
                name = f"mt_contact_{len(pairs)}"
                ET.SubElement(sensors, "contact", name=name, geom1=nominal.geom(geom).name,
                              reduce="none", num=str(CAPACITY + 1), data="found force dist pos normal tangent")
                pairs.append((side, tip, name))
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding="unicode"))
    for name in PHYSICS_FIELDS:
        equal(getattr(model, name), getattr(nominal, name), name)
    for name in OPTION_FIELDS:
        equal(getattr(model.opt, name), getattr(nominal.opt, name), f"option.{name}")
    # Everything above is the sealed task; only the step is the method's own (see clock.py).
    model.opt.timestep = PHYSICS_DT_S
    return model, bodies, pairs, sha256(source)


def gpu_audit(cpu, gpu, task_id, scene_hash):
    import importlib.metadata

    values, options, missing = {}, {}, []
    for field in PHYSICS_FIELDS:
        if not hasattr(gpu, field):
            missing.append(field)
            continue
        expected = np.asarray(getattr(cpu, field))
        value = getattr(gpu, field)
        value = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
        if value.ndim == expected.ndim+1 and value.shape[1:] == expected.shape:
            equal(value, np.broadcast_to(expected, value.shape), f"mjwarp.{field}", atol=1e-7)
            value = value[0]
        equal(value, expected, f"mjwarp.{field}", atol=1e-7)
        values[field] = value.tolist()
    for field in OPTION_FIELDS:
        value = getattr(gpu.opt, "impratio_invsqrt" if field == "impratio" else field)
        value = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
        if field == "impratio":
            value = 1 / value**2
        expected = np.asarray(getattr(cpu.opt, field))
        if value.shape == (1, *expected.shape):
            value = value[0]
        equal(value, expected, f"mjwarp.option.{field}", atol=1e-7)
        options[field] = value.tolist()
    # Some structural arrays are compiled into kernels rather than exposed in MJWarp.
    structural = {"exclude_signature"}
    if set(missing) - structural:
        raise ValueError(f"Cannot verify essential MJWarp physics fields: {sorted(set(missing)-structural)}")
    return {"task_id": task_id, "scene_sha256": scene_hash, "fields": values, "options": options,
            "cpu_only_structural_fields": missing,
            "versions": {p: importlib.metadata.version(p) for p in ("mujoco", "mujoco-warp", "warp-lang", "torch")}}


def verify_sealed(model, compiled):
    """Check version-independent physical values against the archived compilation."""
    robot = compiled["robot"]
    for name, expected in robot["joints"].items():
        j = model.joint(name)
        a = model.actuator(expected["actuator"])
        dof = int(j.dofadr[0])
        actual = dict(type=int(j.type[0]), axis=j.axis, pos=j.pos, limited=bool(j.limited[0]), range=j.range,
            ctrl_limited=bool(a.ctrllimited[0]), ctrlrange=a.ctrlrange, force_limited=bool(a.forcelimited[0]),
            forcerange=a.forcerange, joint_force_limited=model.jnt_actfrclimited[j.id],
            joint_forcerange=model.jnt_actfrcrange[j.id], actgravcomp=model.jnt_actgravcomp[j.id],
            kp=a.gainprm[0], kv=-a.biasprm[2], friction=model.dof_frictionloss[dof],
            damping=model.dof_damping[dof], armature=model.dof_armature[dof])
        for field, value in actual.items():
            equal(value, expected[field], f"sealed.{name}.{field}")
    for name, expected in robot["bodies"].items():
        body = model.body(name)
        for field in ("pos", "quat", "mass", "ipos", "iquat", "inertia", "gravcomp"):
            equal(getattr(model, f"body_{field}")[body.id], expected[field], f"sealed.{name}.{field}")
    for key, expected in robot["geoms"].items():
        body, local_index = key.rsplit("/", 1)
        geom_id = np.flatnonzero(model.geom_bodyid == model.body(body).id)[int(local_index)]
        g = model.geom(int(geom_id))
        for field in ("type", "pos", "quat", "size", "contype", "conaffinity", "condim", "friction",
                      "solref", "solimp", "margin", "gap", "priority", "solmix"):
            equal(getattr(model, f"geom_{field}")[g.id], expected[field], f"sealed.{g.name}.{field}")
    base = compiled["base_scene"]
    for field, value in base["options"].items():
        equal(getattr(model.opt, field), value, f"sealed.option.{field}")
    for name, values in base["geoms"].items():
        for field, value in values.items():
            equal(getattr(model, f"geom_{field}")[model.geom(name).id], value, f"sealed.{name}.{field}")
    for name, values in base["parts"].items():
        body = model.body(f"object_{name}")
        import mujoco
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, body.iquat)
        rotation = rotation.reshape(3, 3)
        equal(body.mass[0], values["mass_kg"], f"sealed.{name}.mass")
        equal(body.ipos, values["com_m"], f"sealed.{name}.com")
        equal(rotation @ np.diag(body.inertia) @ rotation.T, values["inertia_kg_m2"], f"sealed.{name}.inertia")
    for name, values in base["joints"].items():
        joint = model.joint(f"object_{name}")
        dof = int(joint.dofadr[0])
        actual = dict(type=joint.type[0], limited=joint.limited[0], limits_rad=joint.range,
            axis=joint.axis, origin_m=joint.pos, damping=model.dof_damping[dof],
            friction=model.dof_frictionloss[dof], armature=model.dof_armature[dof],
            solimpfriction=model.dof_solimp[dof], solreffriction=model.dof_solref[dof],
            solimplimit=joint.solimp, solreflimit=joint.solref, stiffness=joint.stiffness[0],
            springref_rad=model.qpos_spring[int(joint.qposadr[0])])
        for field, value in values.items():
            equal(actual[field], value, f"sealed.{name}.{field}")


def sensor_columns(model, name):
    sensor = model.sensor(name)
    return slice(int(sensor.adr[0]), int(sensor.adr[0] + sensor.dim[0]))


def check_initial_collision(model, qpos, threshold=MAX_PENETRATION_M):
    import mujoco

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    collisions = []
    for c in data.contact:
        if -float(c.dist) > threshold:
            collisions.append({"geom1": model.geom(int(c.geom1)).name,
                               "geom2": model.geom(int(c.geom2)).name, "depth_m": -float(c.dist)})
    return collisions
