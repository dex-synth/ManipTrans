"""Limited wrist references preserve physical targets and every transition."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from maniptrans_h2r import hand_reference
from maniptrans_h2r.hand_reference import wrist_windows


def test_multiple_turns_fit_only_across_episode_boundaries():
    path = np.column_stack([np.full(7000, .1), np.full(7000, .2), np.linspace(0, 8*np.pi, 7000)])
    segments = wrist_windows(path, np.full(3, -6.2), np.full(3, 6.2))
    covered = []
    previous = None
    for start, angles in segments:
        assert 2 <= len(angles) <= 2000
        assert np.abs(angles).max() <= 6.2+1e-7
        assert np.abs(np.diff(angles, axis=0)).max() < .01
        np.testing.assert_allclose(Rotation.from_euler('XYZ', angles).as_matrix(),
                                   Rotation.from_euler('XYZ', path[start:start+len(angles)]).as_matrix(), atol=1e-12)
        if previous is not None:
            assert start == previous
        previous = start+len(angles)-1
        covered.extend(range(start, previous))
    assert covered == list(range(len(path)-1))


def test_limit_boundaries_split_before_fixed_window_length():
    path = np.column_stack([np.zeros(900), np.zeros(900), np.linspace(0, 6*np.pi, 900)])
    segments = wrist_windows(path, np.full(3, -6.2), np.full(3, 6.2))
    assert len(segments) > 1
    assert sum(len(q)-1 for _, q in segments) == 899


@pytest.mark.parametrize('count', [31, 2000, 2001, 4000, 4020, 10003])
def test_short_tail_balances_without_losing_samples(count):
    segments = wrist_windows(np.zeros((count, 3)), np.full(3, -6.2), np.full(3, 6.2))
    assert sum(len(q)-1 for _, q in segments) == count-1
    assert all(31 <= len(q) <= 2000 for _, q in segments)


def test_unrepresentable_transition_is_rejected_instead_of_removed():
    path = np.array([[0., 0., 0.], [0., 0., 20.]])
    with pytest.raises(ValueError, match='transition 0:2'):
        wrist_windows(path, np.full(3, -6.2), np.full(3, 6.2))


def test_windowing_preserves_whole_clip_velocities(monkeypatch):
    count = 2100
    times = np.arange(count)/60
    yaw = np.linspace(0, 6*np.pi, count)
    rotations = Rotation.from_euler('XYZ', np.column_stack([yaw*0, yaw*0, yaw])).as_matrix()
    points = np.zeros((count, 21, 3))
    points[:, :, 0] = (np.sin(times)*.1)[:, None]
    class Wrist:
        lower, upper = np.full(6, -6.2), np.full(6, 6.2)
        parent_rotation = local_rotation = np.eye(3)
        def __init__(self, *args):
            pass
        def rotations_from_points(self, human):
            return rotations
        def translation(self, positions):
            return positions
    monkeypatch.setattr(hand_reference, 'WristReference', Wrist)
    reference, segments = hand_reference.pretraining_reference(None, None, 'right', points, times)
    assert len(segments) > 1
    quat = hand_reference.quaternion_signs(Rotation.from_matrix(rotations).as_quat(scalar_first=True))
    linear, angular = hand_reference.velocities(points[:, 0], quat, times)
    np.testing.assert_allclose(reference['right_wrist'][:, 7:10], linear, atol=1e-8)
    np.testing.assert_allclose(reference['right_wrist'][:, 10:13], angular, atol=1e-7)
    for start, angles in segments:
        assert np.isfinite(reference['right_wrist'][start:start+len(angles)]).all()
