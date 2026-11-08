from copy import deepcopy
import socket
import struct
import subprocess
import sys

import numpy as np
import pytest
import torch

from maniptrans_h2r.control import PositionControl
from maniptrans_h2r.geometry import bps_basis, encode_bps
from maniptrans_h2r.io import load_artifact, read_json, seal
from maniptrans_h2r.observations import Reference, protocol_state, state_difference
from maniptrans_h2r.policy import Policy, build, parameters
from maniptrans_h2r.protocol import receive, send


def test_sealed_inputs_reject_tampering_and_nonfinite(tmp_path):
    (tmp_path / "value").write_text("original")
    seal(tmp_path, "prepared.json", {"format": "test"})
    load_artifact(tmp_path)
    (tmp_path / "value").write_text("changed")
    with pytest.raises(ValueError, match="Changed dependency"):
        load_artifact(tmp_path)
    for content in ('{"a": 1, "a": 2}', '{"a": 1e999}', '{"a": NaN}'):
        (tmp_path / "bad.json").write_text(content)
        with pytest.raises(ValueError):
            read_json(tmp_path / "bad.json")


def test_bps_uses_unsquared_euclidean_distance_and_fixed_basis():
    basis = bps_basis()
    assert basis.shape == (128, 3)
    np.testing.assert_allclose(encode_bps(np.zeros((1, 3))), np.linalg.norm(basis, axis=1), atol=1e-7)
    np.testing.assert_allclose(encode_bps(basis), 0, atol=1e-7)
    assert np.linalg.norm(basis, axis=1).max() <= .2000001


def test_reference_clock_and_quaternion_sign():
    states = np.zeros((2, 1, 13), np.float32)
    states[:, 0, 3] = [1, -1]
    states[1, :, 0] = 2
    reference = Reference({"time_s": np.array([0., 2.]), "object_state": states}, "cpu")
    sampled = reference.sample(torch.tensor([1., 3.]))["object_state"]
    torch.testing.assert_close(sampled[:, 0, 0], torch.tensor([1., 2.]))
    torch.testing.assert_close(sampled[..., 3].abs(), torch.ones(2, 1))
    changed = sampled.clone()
    changed[..., 3:7] *= -1
    torch.testing.assert_close(state_difference(sampled, sampled), state_difference(changed, sampled))


def test_wrist_position_residual_and_episode_history():
    robot = dict(joint_names=list(range(8)), lower=[-1]*8, upper=[1]*8,
                 hands={"right": {"indices": list(range(8))}})
    control = PositionControl(robot, 2, "cpu", dict(wrist_translation_m=.1, wrist_rotation_rad=.2, moving_average=.5))
    initial = torch.zeros(2, 8)
    control.reset([0, 1], initial)
    target = initial.clone()
    target[:, :6] = .25
    action = torch.zeros(2, 16)
    action[0, :8] = .5
    action[0, 8:] = .25
    result = control.decode(action, target, ["right"], True)
    torch.testing.assert_close(result[0], torch.tensor([.175]*3+[.225]*3+[.5]*2))
    previous = result[0].clone()
    control.reset([1], torch.zeros(1, 8))
    torch.testing.assert_close(control.previous[0], previous)


def test_contact_force_sign_and_missing_contact_rejection():
    robot = {"sides": ["right"], "hands": {"right": {"bodies": ["tip"], "contact_bodies": ["tip"]}}}
    task = {"object": {"parts": {"part": "object"}}}
    frame = dict(robot_q=[0], robot_dq=[0], object_q=[], object_dq=[],
                 bodies={"tip": [0]*13, "object": [0]*13},
                 contacts={"records": [{"body1": "tip", "body2": "object", "force_on_body2_world": [1, 2, 3]}]})
    state = protocol_state([frame], robot, task, "cpu")
    torch.testing.assert_close(state["tip_force"]["right"][0, 0], torch.tensor([-1., -2., -3.]))
    frame["contacts"] = None
    with pytest.raises(ValueError, match="missing"):
        protocol_state([frame], robot, task, "cpu")


def test_native_policy_episode_rng_and_conditional_base_actions(tmp_path):
    torch.set_num_threads(2)
    dimensions = dict(privileged=4, proprioception=5, target=6)
    base = parameters(dimensions, dimensions, ["right"], {}, 8, True)
    model = build(base, dimensions, 8, "cpu")
    path = tmp_path / "base.pth"
    torch.save({"model": model.state_dict()}, path)
    residual = parameters(dimensions, dimensions, ["right"], {"right": str(path)}, 8, True)
    model = build(residual, dimensions, 8, "cpu").eval()
    single = Policy(deepcopy(model), 8, {"a": 123}, "cpu")
    batch = Policy(deepcopy(model), 8, {"a": 123, "b": 456}, "cpu")
    observation = {k: torch.randn(2, n) for k, n in dimensions.items()}
    for _ in range(3):
        a = single({k: v[:1] for k, v in observation.items()}, ["a"])
        ab = batch(observation, ["a", "b"])
        torch.testing.assert_close(a, ab[:1], rtol=1e-5, atol=1e-6)
    base_actions = torch.randn(2, 8)
    def forward():
        return model({"is_train": True, "prev_actions": torch.zeros(2, 8),
                      "obs": dict(observation), "base_actions": base_actions})
    torch.testing.assert_close(forward()["mus"], forward()["mus"], rtol=0, atol=0)
    assert all(not p.requires_grad for p in model.a2c_network.base_model.parameters())


def test_protocol_rejects_duplicate_and_overflow_numbers():
    for payload in (b'{"a":1,"a":2}', b'{"a":1e999}'):
        left, right = socket.socketpair()
        with left, right:
            left.sendall(struct.pack("!I", len(payload))+payload)
            with pytest.raises(ValueError):
                receive(right)
    left, right = socket.socketpair()
    with left, right:
        send(left, {"type": "step", "commands": {}})
        assert receive(right) == {"type": "step", "commands": {}}


def test_online_import_does_not_load_a_physics_runtime():
    subprocess.run([sys.executable, "-c", "import sys; import maniptrans_h2r.evaluate; "
                    "assert not any(n in sys.modules for n in ('mujoco', 'mujoco_warp', 'warp', 'isaacgym'))"], check=True)
