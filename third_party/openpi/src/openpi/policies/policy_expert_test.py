import numpy as np
import pytest

from openpi.policies.policy_expert import _canonicalize_revolute_joints
from vjepa2.FK import PandaFK


def test_canonicalizes_single_turn_alias():
    raw = np.array(
        [
            0.06961487,
            0.23888142,
            -0.0467384,
            -2.61087298,
            0.05135126,
            7.27308083,
            1.5428164,
        ]
    )

    canonical, turns = _canonicalize_revolute_joints(raw, PandaFK._JOINT_LIMITS)

    np.testing.assert_allclose(canonical[5], raw[5] - 2 * np.pi, atol=1e-6)
    np.testing.assert_array_equal(turns, [0, 0, 0, 0, 0, 1, 0])


def test_canonicalizes_multiple_turn_aliases():
    canonical_expected = np.array(
        [
            -0.22900453,
            1.05702317,
            2.43265152,
            -2.08048439,
            -1.96017823,
            0.44400389,
            1.56337325,
        ]
    )
    turns = np.array([0, 0, 0, 0, 6, -1, -9])
    raw = canonical_expected + turns * (2 * np.pi)

    canonical, observed_turns = _canonicalize_revolute_joints(
        raw, PandaFK._JOINT_LIMITS
    )

    np.testing.assert_allclose(canonical, canonical_expected, atol=1e-5)
    np.testing.assert_array_equal(observed_turns, turns)


def test_rejects_state_without_limit_equivalent():
    raw = np.array([0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0])

    with pytest.raises(RuntimeError, match="Simulator joint state diverged"):
        _canonicalize_revolute_joints(raw, PandaFK._JOINT_LIMITS)


def test_rejects_extreme_turn_count():
    raw = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 65 * 2 * np.pi + 1.0, 0.0])

    with pytest.raises(RuntimeError, match="Simulator joint state diverged"):
        _canonicalize_revolute_joints(raw, PandaFK._JOINT_LIMITS)


def test_clips_small_limit_overshoot():
    raw = np.array([0.0, 0.0, 0.0, -1.0, -3.0, 1.0, 0.0])

    canonical, turns = _canonicalize_revolute_joints(raw, PandaFK._JOINT_LIMITS)

    np.testing.assert_allclose(canonical[4], PandaFK._JOINT_LIMITS[4, 0])
    np.testing.assert_array_equal(turns, np.zeros(7, dtype=np.int32))
