from copy import deepcopy
import subprocess
import sys

import numpy as np
import pytest
import torch

from maniptrans_h2r.geometry import bps_basis, encode_bps
from maniptrans_h2r.io import digest, load_artifact, read_json, seal, write_json
from maniptrans_h2r.observations import Reference, protocol_state, state_difference
from maniptrans_h2r.policy import Policy, build, parameters


def test_sealed_inputs_reject_tampering_and_nonfinite(tmp_path):
    (tmp_path / "value").write_text("original")
    seal(tmp_path, "prepared", {})
    load_artifact(tmp_path, "prepared")
    with pytest.raises(ValueError, match="reads 'maniptrans-h2r-pretrain-v2'"):
        load_artifact(tmp_path, "pretrain")  # The same file name, another kind of artifact.
    (tmp_path / "value").write_text("changed")
    with pytest.raises(ValueError, match="Changed dependency"):
        load_artifact(tmp_path, "prepared")
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


@pytest.mark.parametrize("kind,name,old", [
    ("prepared", "prepared.json", "maniptrans-h2r-prepared-v2"), ("pretrain", "prepared.json", "maniptrans-h2r-pretrain-v1"),
    ("imitator", "imitator.json", "maniptrans-h2r-imitator-v1"), ("deployment", "deployment.json", "maniptrans-h2r-deployment-v2")])
def test_artifacts_of_the_former_task_format_are_rejected(tmp_path, kind, name, old):
    # A consistent artifact as the previous version sealed it: bound to a bundle, 120 Hz task ticks.
    metadata = dict(format=old, task_id="t", bundle_id="b", robot_manifest_sha256="r", files={})
    write_json(tmp_path / name, dict(metadata, id=digest(metadata)))
    with pytest.raises(ValueError, match="earlier task format cannot be reused: prepare and train again"):
        load_artifact(tmp_path, kind)


def test_contact_force_sign_joint_order_and_missing_contact_rejection():
    robot = {"sides": ["right"], "hands": {"right": {"bodies": ["tip"], "contact_bodies": ["tip"]}}}
    frame = dict(robot_q=[1, 2], robot_dq=[3, 4], object_q=[5], object_dq=[6],
                 bodies={"tip": [0]*13, "object": [0]*13},
                 contacts={"records": [{"body1": "tip", "body2": "object", "force_on_body2_world": [1, 2, 3]},
                                       {"body1": "other", "body2": "tip", "force_on_body2_world": [0, 0, 10]}]})
    state = protocol_state([frame], robot, ["object"], dict(robot=[1, 0], object=[0]), "cpu")
    torch.testing.assert_close(state["tip_force"]["right"][0, 0], torch.tensor([-1., -2., 7.]))
    assert state["robot_q"].tolist() == [[2, 1]] and state["robot_dq"].tolist() == [[4, 3]]
    frame["contacts"] = None
    with pytest.raises(ValueError, match="missing"):
        protocol_state([frame], robot, ["object"], dict(robot=[1, 0], object=[0]), "cpu")


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


def test_online_import_does_not_load_a_physics_runtime():
    subprocess.run([sys.executable, "-c", "import sys; import maniptrans_h2r.evaluate; "
                    "assert not any(n in sys.modules for n in ('mujoco', 'mujoco_warp', 'warp', 'isaacgym'))"], check=True)
