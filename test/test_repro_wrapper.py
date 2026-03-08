"""
Reproducibility tests — RLLibSingleAgentWrapper.

Verifies that the wrapper produces bit-identical observations, rewards, and
termination signals when initialised and stepped with the same seed and actions.

Covers:
- reset → identical flat observation vectors
- step  → identical (obs, reward, terminated, truncated) tuples
- multi-step sequence → identical trajectory
- cross-seed divergence
"""

from __future__ import annotations

import numpy as np
import pytest

from rllib_single_agent_wrapper import RLLibSingleAgentWrapper

from repro_helpers import make_small_env, make_wrapper_with_policies


# ===========================================================================
# Wrapper reproducibility
# ===========================================================================

class TestWrapperResetReproducibility:
    """Wrapper reset with same seed → identical flat observations."""

    def test_reset_identical(self):
        w1 = make_wrapper_with_policies(seed=42)
        w2 = make_wrapper_with_policies(seed=42)
        obs1, _ = w1.reset(seed=42)
        obs2, _ = w2.reset(seed=42)
        np.testing.assert_allclose(obs1, obs2, rtol=0, atol=0)

    def test_different_seed_diverges(self):
        w1 = make_wrapper_with_policies(seed=42)
        w2 = make_wrapper_with_policies(seed=99)
        obs1, _ = w1.reset(seed=42)
        obs2, _ = w2.reset(seed=99)
        assert not np.allclose(obs1, obs2), (
            "Different seeds should give different obs"
        )


class TestWrapperStepReproducibility:
    """Wrapper step with identical actions → identical results."""

    def test_single_step_identical(self):
        w1 = make_wrapper_with_policies(seed=42)
        w2 = make_wrapper_with_policies(seed=42)
        w1.reset(seed=42)
        w2.reset(seed=42)

        action = 0
        obs1, r1, t1, tr1, _ = w1.step(action)
        obs2, r2, t2, tr2, _ = w2.step(action)

        np.testing.assert_allclose(
            obs1, obs2, rtol=0, atol=0, err_msg="Obs after step differ",
        )
        assert r1 == pytest.approx(r2), "Rewards after step differ"
        assert t1 == t2, "Terminated differs"
        assert tr1 == tr2, "Truncated differs"

    @pytest.mark.integration
    def test_multi_step_identical(self):
        w1 = make_wrapper_with_policies(seed=42)
        w2 = make_wrapper_with_policies(seed=42)
        w1.reset(seed=42)
        w2.reset(seed=42)

        for action in [0, 1, 2, 0, 3, 1]:
            o1, r1, t1, tr1, _ = w1.step(action)
            o2, r2, t2, tr2, _ = w2.step(action)
            np.testing.assert_allclose(o1, o2, rtol=0, atol=0)
            assert r1 == pytest.approx(r2)
            assert t1 == t2
            assert tr1 == tr2
            if t1 or tr1:
                break


class TestWrapperCrossSeedDivergence:
    """Wrappers from bare env with different seeds must diverge."""

    def test_wrapper_obs_differ(self):
        env1 = make_small_env()
        env2 = make_small_env()
        w1 = RLLibSingleAgentWrapper(
            env1, controlled_agent="agent_0", force_episode_horizon=30,
        )
        w2 = RLLibSingleAgentWrapper(
            env2, controlled_agent="agent_0", force_episode_horizon=30,
        )
        obs1, _ = w1.reset(seed=1)
        obs2, _ = w2.reset(seed=2)
        assert not np.allclose(obs1, obs2)

