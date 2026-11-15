"""The one input: a task directory read through the vendored loader, and the v2 evaluation handshake."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from maniptrans_h2r.clock import CONTROL_DT_S, PHYSICS_DT_S
from maniptrans_h2r.control import CONTROL
from maniptrans_h2r.evaluate import Controller
from maniptrans_h2r.h2r_io import client, load_task
from maniptrans_h2r.io import read_json
from maniptrans_h2r.physics import load_model
from maniptrans_h2r.robot import command_limits, initial_qpos, mapping
from maniptrans_h2r.robot_physics import extract_hand, same_signature, signature
from synthetic import NOMINAL_DT_S, write_task


def load(tmp_path, **options):
    task = load_task(write_task(tmp_path / "task", **options))
    return task, mujoco.MjModel.from_xml_path(str(task.scene_xml))


@pytest.mark.parametrize("sides", [("left", "right"), ("right",), ("left",)])
def test_mapping_follows_the_hands_of_the_task(tmp_path, sides):
    task, model = load(tmp_path, sides=sides)
    robot = mapping(model, task)
    assert robot["sides"] == [side for side in ("right", "left") if side in sides]  # Bimanual policy order.
    assert robot["joint_names"] == task.robot.semantics["joint_names"]
    for side in sides:
        hand = robot["hands"][side]
        assert hand["joints"][:6] == [f"{side}_{axis}_joint" for axis in ("x", "y", "z", "roll", "pitch", "yaw")]
        assert len(hand["bodies"]) == 21 and hand["bodies"][0] == f"{side}_palm"
    qpos = initial_qpos(model, task)
    root = int(model.joint("object_bottom_free").qposadr[0])
    np.testing.assert_allclose(qpos[root:root+7], [0, 0, .03, 1, 0, 0, 0])
    assert qpos[int(model.joint("object_lid").qposadr[0])] == 0  # Reference frame 0.


def test_a_robot_without_keypoint_correspondence_is_rejected_by_name(tmp_path):
    task, model = load(tmp_path, robot_name="other_hand")
    with pytest.raises(ValueError, match="No ManipTrans keypoint correspondence for robot 'other_hand'"):
        mapping(model, task)


def test_training_models_carry_the_method_step_not_the_task_step(tmp_path):
    task, model = load(tmp_path)
    assert task.timestep_s == model.opt.timestep == NOMINAL_DT_S != PHYSICS_DT_S
    robot = command_limits(model, mapping(model, task))
    scene, bodies, pairs, _ = load_model(task, robot)
    assert scene.opt.timestep == PHYSICS_DT_S
    assert set(task.object_binding["parts"].values()) <= set(bodies)
    assert {side for side, _, _ in pairs} == {"left", "right"}  # Fingertip contacts with every partner.
    hand, hand_robot, compatibility = extract_hand(task, robot, "right", tmp_path / "right")
    assert hand.opt.timestep == PHYSICS_DT_S and hand.nu == 21 and hand_robot["sides"] == ["right"]
    assert compatibility["options"]["timestep"] == compatibility["physics_dt_s"] == PHYSICS_DT_S
    assert compatibility["control_dt_s"] == CONTROL_DT_S
    assert read_json(tmp_path / "right/compatibility.json") == compatibility
    with pytest.raises(ValueError, match="training physics step"):
        from maniptrans_h2r.mjlab_env import MjlabVecEnv
        MjlabVecEnv(model, robot, [], [], 1, "cuda:0", {}, "residual")


def test_a_bimanual_base_serves_the_single_hand_task(tmp_path):
    both, model = load(tmp_path / "both")
    single, single_model = load(tmp_path / "single", sides=("right",))
    base = signature(model, mapping(model, both), "right")
    assert same_signature(base, signature(single_model, mapping(single_model, single), "right"))
    assert not same_signature(base, signature(model, mapping(model, both), "left"))


@pytest.fixture
def controller():
    result = Controller.__new__(Controller)
    names = [f"q{i}" for i in range(7)]
    result.robot = dict(joint_names=names, sides=["right"], lower=[-1.]*7, upper=[1.]*7, command_lower=[-1.]*7,
                        command_upper=[1.]*7, hands={"right": dict(indices=list(range(7)), bodies=["hand"], contact_bodies=["hand"])})
    result.meta = dict(object_bodies=["object"], object_joint_names=["lid"])
    result.manifest = dict(task_id="task", method="maniptrans-mjlab", controller=CONTROL["schema"], id="deployment",
                           checkpoint="model.pth", control=CONTROL, residual_mean=True, action_dim=14,
                           training=dict(training_seed=42, training_epochs=3, training_transitions=96))
    result.control_dt_s, result.device, result.model = CONTROL_DT_S, "cpu", torch.nn.Identity()
    result.initial = torch.zeros(1, 7)
    result.controls, result.next_ticks = {}, {}
    return result


def hello(**changes):
    result = dict(type="hello", protocol=client.PROTOCOL, task_id="task", demo_id="demo", robot_id="robot",
                  dt_s=1/300, end_tick=300, joint_names=[f"q{i}" for i in reversed(range(7))],
                  object=dict(joints={"lid": "object_lid"}),
                  observation=dict(bodies=["hand", "object", "other"], contacts=True, contact_coverage="robot-all-v1"),
                  episodes=[dict(episode_id="e0", seed=0, controller_seed=5)])
    return {**result, **changes}


def test_start_declares_the_policy_period_on_the_announced_physics_step(controller, tmp_path):
    controller.root = tmp_path
    (tmp_path / "model.pth").write_bytes(b"weights")
    start = controller.start(hello())
    assert start["requirements"] == dict(bodies=["hand", "object"], contacts=True, control_dt_s=CONTROL_DT_S)
    assert start["task_id"] == "task" and start["binding"]["method"] == "maniptrans-mjlab"
    assert {"demo_id": "demo", "robot_id": "robot"}.items() <= start["binding"].items()
    assert start["binding"]["controller"]["training_epochs"] == 3 and len(start["binding"]["controller"]["checkpoint_sha256"]) == 64
    assert set(start["initial_robot"]["e0"]) == set(controller.robot["joint_names"])
    assert controller.ticks == 5 and controller.orders == dict(robot=list(reversed(range(7))), object=[0])
    controller.start(hello(dt_s=1/120, end_tick=120))  # The evaluation step is the evaluator's choice.
    assert controller.ticks == 2
    with pytest.raises(ValueError, match="not a whole number"):
        controller.start(hello(dt_s=1/100, end_tick=100))


@pytest.mark.parametrize("changes,message", [
    (dict(protocol="h2r-evaluation-v1", control_ticks=2), "Unsupported evaluation protocol 'h2r-evaluation-v1'"),
    (dict(task_id="other"), "different task"),
    (dict(observation=dict(bodies=["hand", "object"], contacts=True, contact_coverage="robot-object-environment")),
     "contact_sensing.include_robot_robot=true"),
    (dict(observation=dict(bodies=["hand", "object"], contacts=False)), "needs robot-all-v1 contact observations"),
])
def test_start_rejects_other_protocols_tasks_and_reduced_contacts(controller, changes, message):
    with pytest.raises(ValueError, match=message):
        controller.start(hello(**changes))


def test_suite_driver_picks_bases_by_signature_and_keeps_failed_cases(tmp_path, monkeypatch):
    from maniptrans_h2r import suite
    cases = [SimpleNamespace(case_id="set/a++b/hand", task_id="1", task=tmp_path / "first"),
             SimpleNamespace(case_id="set/taco/hand", task_id="2", task=tmp_path / "second")]
    monkeypatch.setattr(suite, "load_suite", lambda path: SimpleNamespace(cases=cases))
    bases = {"right": dict(side="right", compatibility=dict(friction=2.)), "left": dict(side="left", compatibility=dict(friction=2.)),
             "taco-right": dict(side="right", compatibility=dict(friction=.8))}
    prepared = {"set-a__b-hand": dict(right=dict(friction=2.), left=dict(friction=2.)),
                "set-taco-hand": dict(right=dict(friction=.8), left=dict(friction=.8))}
    monkeypatch.setattr(suite, "load_artifact", lambda path, kind: (None, bases[path] if kind == "imitator" else
                                                                    dict(compatibility=prepared[path.parent.name])))
    calls = []

    def run(module, arguments, log):
        calls.append((module, arguments))
        return 0

    monkeypatch.setattr(suite, "run", run)
    monkeypatch.setattr("sys.argv", ["suite", "--suite", "suite", "--output", str(tmp_path / "out"), "--imitator", *bases,
                                     "--iterations", "7", "--", "--epochs", "3"])
    with pytest.raises(SystemExit) as stopped:
        suite.main()
    assert stopped.value.code == 1  # The second case has no left base with its contact parameters.
    assert [module for module, _ in calls] == ["prepare", "train", "prepare"]
    prepare, train = calls[0][1], calls[1][1]
    assert prepare[prepare.index("--task")+1] == str(tmp_path / "first") and prepare[prepare.index("--iterations")+1] == "7"
    assert train[train.index("--imitator-right")+1] == "right" and train[train.index("--imitator-left")+1] == "left"
    assert train[-2:] == ["--epochs", "3"]
    results = [read_json(tmp_path / "out" / name / "result.json") for name in prepared]
    assert [result["status"] for result in results] == ["completed", "failed"]
    assert "no given base matches the left hand physics" in results[1]["reason"]
