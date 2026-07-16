import numpy as np
import pytest

from openpi.policies.policy_expert import Policy
from openpi.policies.policy_expert import RecoverableExpertError
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


def test_rejects_reported_joint_six_divergence():
    raw = np.array(
        [
            -0.50939953,
            0.2298124,
            0.10433729,
            -2.07029057,
            -0.01091915,
            4.94146442,
            -1.73471487,
        ]
    )

    with pytest.raises(RecoverableExpertError, match="Simulator joint state diverged"):
        _canonicalize_revolute_joints(raw, PandaFK._JOINT_LIMITS)


def test_infer_returns_abort_for_recoverable_failure(monkeypatch):
    policy = Policy.__new__(Policy)
    policy._open = 0.0

    def fail_infer(obs):
        raise RecoverableExpertError("simulator diverged")

    monkeypatch.setattr(policy, "_infer_impl", fail_infer)

    response = policy.infer({})

    assert response["abort_episode"] is True
    assert response["abort_reason"] == "simulator diverged"
    assert response["actions"].shape == (16, 8)
    assert np.all(np.isfinite(response["actions"]))


@pytest.mark.parametrize("error", [RuntimeError("bug"), TimeoutError("bug")])
def test_infer_does_not_hide_unexpected_errors(monkeypatch, error):
    policy = Policy.__new__(Policy)

    def fail_infer(obs):
        raise error

    monkeypatch.setattr(policy, "_infer_impl", fail_infer)

    with pytest.raises(type(error), match="bug"):
        policy.infer({})


def test_clips_small_limit_overshoot():
    raw = np.array([0.0, 0.0, 0.0, -1.0, -3.0, 1.0, 0.0])

    canonical, turns = _canonicalize_revolute_joints(raw, PandaFK._JOINT_LIMITS)

    np.testing.assert_allclose(canonical[4], PandaFK._JOINT_LIMITS[4, 0])
    np.testing.assert_array_equal(turns, np.zeros(7, dtype=np.int32))
