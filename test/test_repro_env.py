"""
Reproducibility tests — Environment layer.

Verifies that ``PeerGroupEnvironment.reset()`` and ``.step()`` produce
bit-identical results when invoked with the same seed and actions.

Covers:
- reset → identical observations, action masks, agent ages, peer groups
- step  → identical rewards, terminations, next observations
- full episode with deterministic policies → identical trajectories
- different seed → divergent outcomes (sanity check)
"""

from __future__ import annotations

import numpy as np
import pytest

from agent_policies import do_nothing_policy

from repro_helpers import (
    assert_nested_obs_equal,
    flatten_obs_deterministic,
    make_small_env,
    run_deterministic_episode,
)


# ===========================================================================
# reset() reproducibility
# ===========================================================================

class TestEnvResetReproducibility:
    """``reset(seed=X)`` called twice must yield identical initial states."""

    def test_observations_identical(self):
        env = make_small_env()
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)

        for agent in env.possible_agents:
            assert_nested_obs_equal(
                obs1[agent]["observation"],
                obs2[agent]["observation"],
                label=agent,
            )

    def test_action_masks_identical(self):
        env = make_small_env()
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)

        for agent in env.possible_agents:
            for key in sorted(obs1[agent]["action_mask"].keys()):
                np.testing.assert_array_equal(
                    obs1[agent]["action_mask"][key],
                    obs2[agent]["action_mask"][key],
                    err_msg=f"Mask mismatch for {agent}/{key}",
                )

    def test_agent_ages_identical(self):
        env = make_small_env()
        env.reset(seed=42)
        ages1 = env.agent_ages.copy()
        env.reset(seed=42)
        ages2 = env.agent_ages.copy()
        np.testing.assert_allclose(ages1, ages2, rtol=0, atol=0)

    def test_peer_groups_identical(self):
        env = make_small_env()
        env.reset(seed=42)
        pg1 = [sorted(g) for g in env.peer_groups]
        env.reset(seed=42)
        pg2 = [sorted(g) for g in env.peer_groups]
        assert pg1 == pg2

    def test_different_seed_produces_different_ages(self):
        env = make_small_env()
        env.reset(seed=42)
        ages1 = env.agent_ages.copy()
        env.reset(seed=99)
        ages2 = env.agent_ages.copy()
        assert not np.allclose(ages1, ages2), (
            "Different seeds should produce different agent ages"
        )

    def test_full_flat_obs_identical(self):
        """Flatten every agent's obs deterministically and compare full arrays."""
        env = make_small_env()
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)

        for agent in sorted(env.possible_agents):
            flat1 = flatten_obs_deterministic(obs1[agent]["observation"])
            flat2 = flatten_obs_deterministic(obs2[agent]["observation"])
            np.testing.assert_allclose(
                flat1, flat2, rtol=0, atol=0,
                err_msg=f"Flat obs mismatch for {agent}",
            )


# ===========================================================================
# step() reproducibility
# ===========================================================================

class TestEnvStepReproducibility:
    """``step()`` with identical actions must produce identical results."""

    def test_single_step_deterministic(self):
        env = make_small_env()

        # --- Run 1 ---
        obs, _ = env.reset(seed=42)
        actions = {
            a: do_nothing_policy(obs[a]["observation"], obs[a]["action_mask"])
            for a in env.agents
        }
        obs1, r1, t1, tr1, _ = env.step(actions)

        # --- Run 2 (identical) ---
        obs, _ = env.reset(seed=42)
        actions = {
            a: do_nothing_policy(obs[a]["observation"], obs[a]["action_mask"])
            for a in env.agents
        }
        obs2, r2, t2, tr2, _ = env.step(actions)

        for agent in env.possible_agents:
            assert r1[agent] == pytest.approx(r2[agent]), (
                f"Reward mismatch for {agent}"
            )
            assert t1[agent] == t2[agent], f"Termination mismatch for {agent}"
            assert_nested_obs_equal(
                obs1[agent]["observation"],
                obs2[agent]["observation"],
                label=f"{agent} post-step",
            )


# ===========================================================================
# Full episode reproducibility
# ===========================================================================

class TestFullEpisodeReproducibility:
    """Full episodes with same seed + deterministic policies → identical trajectories."""

    @pytest.mark.integration
    def test_two_runs_identical(self):
        env = make_small_env()
        snap1 = run_deterministic_episode(env, seed=42, n_steps=15)
        snap2 = run_deterministic_episode(env, seed=42, n_steps=15)

        # Rewards: exact match (deterministic policies, same RNG state)
        assert snap1["rewards"] == pytest.approx(snap2["rewards"]), (
            "Reward trajectories differ"
        )

        # Full observation arrays: element-wise match
        assert len(snap1["obs_flat"]) == len(snap2["obs_flat"])
        for step_idx, (f1, f2) in enumerate(
            zip(snap1["obs_flat"], snap2["obs_flat"])
        ):
            np.testing.assert_allclose(
                f1, f2, rtol=0, atol=0,
                err_msg=f"Obs mismatch at step {step_idx}",
            )

        np.testing.assert_allclose(
            snap1["agent_ages"], snap2["agent_ages"], rtol=0, atol=0,
        )
        np.testing.assert_array_equal(snap1["completed"], snap2["completed"])
        np.testing.assert_array_equal(snap1["active"], snap2["active"])
        assert snap1["peer_groups"] == snap2["peer_groups"]

    @pytest.mark.integration
    def test_different_seed_diverges(self):
        env = make_small_env()
        snap1 = run_deterministic_episode(env, seed=42, n_steps=15)
        snap2 = run_deterministic_episode(env, seed=99, n_steps=15)

        ages_differ = not np.allclose(snap1["agent_ages"], snap2["agent_ages"])
        rewards_differ = snap1["rewards"] != pytest.approx(snap2["rewards"])
        obs_differ = any(
            not np.allclose(f1, f2)
            for f1, f2 in zip(snap1["obs_flat"], snap2["obs_flat"])
        )
        assert ages_differ or rewards_differ or obs_differ, (
            "Different seeds must produce divergent results"
        )

    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 123, 999])
    def test_multiple_seeds_reproducible(self, seed):
        env = make_small_env()
        snap1 = run_deterministic_episode(env, seed=seed, n_steps=10)
        snap2 = run_deterministic_episode(env, seed=seed, n_steps=10)

        assert snap1["rewards"] == pytest.approx(snap2["rewards"])
        for f1, f2 in zip(snap1["obs_flat"], snap2["obs_flat"]):
            np.testing.assert_allclose(f1, f2, rtol=0, atol=0)

    def test_env_observations_cross_seed_differ(self):
        """Flattened observations with different seeds must diverge."""
        env = make_small_env()
        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=2)

        agent = env.possible_agents[0]
        flat1 = flatten_obs_deterministic(obs1[agent]["observation"])
        flat2 = flatten_obs_deterministic(obs2[agent]["observation"])
        assert not np.allclose(flat1, flat2), (
            "Observations with different seeds should differ"
        )

