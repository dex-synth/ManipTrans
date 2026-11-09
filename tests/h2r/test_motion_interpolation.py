"""Timestamp subtraction must not exclude an otherwise valid last frame."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from maniptrans_h2r import hand_reference


@pytest.fixture
def motion(monkeypatch):
    # Actual failing HOT3D duration, after subtracting absolute timestamps.
    times = np.array([0., 2.099999999998545])
    quat = Rotation.from_rotvec([[0., 0., 0.], [.5, 0., 0.]]).as_quat(scalar_first=True)
    points = np.zeros((2, 21, 3))
    points[1, :, 0] = 1.5
    arrays = dict(time_s=times, wrist_quat_wxyz=quat,
                  local_quat_wxyz=np.repeat(quat[:, None], 15, axis=1), joints_m=points, betas=np.zeros(10))
    metadata = dict(source_side='left', hot3d_shapedirs_fix=True, tip_vertex_ids=list(range(5)))

    def model(*, global_orient, hand_pose, betas):
        count = len(hand_pose)
        return SimpleNamespace(joints=torch.cat([torch.zeros(count, 1, 3), hand_pose.reshape(count, 15, 3)], dim=1),
                               vertices=torch.zeros(count, 5, 3))

    monkeypatch.setattr(hand_reference, 'mano_model', lambda *args: model)
    return arrays, metadata


@pytest.mark.parametrize('synthetic', [False, True])
def test_endpoint_roundoff_preserves_last_pose_and_input_times(motion, synthetic):
    arrays, metadata = motion
    times = np.array([-1e-12, 1., 2.1])
    original = times.copy()
    endpoints = [0., 2.1] if synthetic else None
    actual = hand_reference.interpolate_motion(arrays, metadata, times, '', endpoints=endpoints)
    exact = np.array([0., 1., arrays['time_s'][-1]])
    expected = hand_reference.interpolate_motion(arrays, metadata, exact, '',
        endpoints=arrays['time_s'] if synthetic else None)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(times, original)


@pytest.mark.parametrize('times,endpoints', [([0., 2.100001], None), ([-1e-6, 1.], None),
                                            ([0., 1.], [0., 2.100001])])
def test_real_extrapolation_still_fails(motion, times, endpoints):
    arrays, metadata = motion
    with pytest.raises(ValueError, match='outside source interval'):
        hand_reference.interpolate_motion(arrays, metadata, times, '', endpoints=endpoints)
