"""
Tests for the PeerGroupEnvironment.

Verifies core mechanics:
- peer group connection adds cross-group members
- action mask structure is valid for active/inactive agents
- environment reset is reproducible with the same seed
- environment step produces valid outputs
"""

import numpy as np
import pytest

from env.peer_group_environment import PeerGroupEnvironment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_env():
    """A small, deterministic environment for unit tests."""
    env = PeerGroupEnvironment(
        start_agents=4,
        max_agents=8,
        max_steps=50,
        n_groups=2,
        max_peer_group_size=4,
        n_projects_per_step=1,
        max_projects_per_agent=2,
        max_agent_age=100,
    )
    env.reset(seed=42)
    return env


# ---------------------------------------------------------------------------
# Peer group connection
# ---------------------------------------------------------------------------

class TestConnectPeerGroups:
    def test_connect_does_not_crash(self, small_env):
        """_connect_peer_groups must not crash when there are active agents
        in disjoint groups."""
        env = small_env

        # Manually set up two clean groups with known members
        env.peer_groups = [[0, 1], [2, 3]]
        env.agent_peer_idx = [0, 0, 1, 1] + [0] * (env.n_agents - 4)
        env.active_agents = np.array([1, 1, 1, 1] + [0] * (env.n_agents - 4), dtype=np.int8)

        # Should not raise
        env._connect_peer_groups()

    @pytest.mark.xfail(
        reason=(
            "Known bug: _connect_peer_groups selects from active-status values "
            "(0/1) instead of agent indices, so cross-group insertion may "
            "add already-present members → no growth."
        ),
        strict=False,
    )
    def test_adds_cross_group_agents(self, small_env):
        """Connecting peer groups should add at least one cross-group member
        to each group when active candidates are available.

        NOTE: This test documents a known issue in _connect_peer_groups where
        ``self.active_agents[list(group - other)]`` returns *status values*
        (0 or 1) rather than *agent indices*. ``rng.choice`` then picks from
        those values, which may coincide with agents already in the group,
        causing no net growth.
        """
        env = small_env

        env.peer_groups = [[0, 1], [2, 3]]
        env.agent_peer_idx = [0, 0, 1, 1] + [0] * (env.n_agents - 4)
        env.active_agents = np.array([1, 1, 1, 1] + [0] * (env.n_agents - 4), dtype=np.int8)

        len0_before = len(env.peer_groups[0])
        len1_before = len(env.peer_groups[1])

        env._connect_peer_groups()

        assert len(env.peer_groups[0]) > len0_before, (
            "Group 0 should have gained a cross-group member"
        )
        assert len(env.peer_groups[1]) > len1_before, (
            "Group 1 should have gained a cross-group member"
        )

    def test_respects_max_peer_group_size(self, small_env):
        """Groups at max capacity should not gain new members."""
        env = small_env

        # Fill group 0 to max, leave group 1 small
        env.peer_groups = [list(range(env.max_peer_group_size)), [env.max_peer_group_size]]
        env.agent_peer_idx = [0] * env.max_peer_group_size + [1] + [1] * (env.n_agents - env.max_peer_group_size - 1)
        env.active_agents[:env.max_peer_group_size + 1] = 1

        len0_before = len(env.peer_groups[0])
        env._connect_peer_groups()

        # Group 0 was already at max, should not grow
        assert len(env.peer_groups[0]) == len0_before


# ---------------------------------------------------------------------------
# Action mask
# ---------------------------------------------------------------------------

class TestGetActionMask:
    def test_returns_valid_structure(self, small_env):
        """Action mask should contain all expected keys with correct shapes."""
        env = small_env
        mask = env._get_action_mask("agent_0")

        assert isinstance(mask, dict)
        assert "choose_project" in mask
        assert "collaborate_with" in mask
        assert "put_effort" in mask

        assert mask["choose_project"].shape == (env.n_projects_per_step + 1,)
        assert mask["collaborate_with"].shape == (env.max_peer_group_size,)
        assert mask["put_effort"].shape == (env.max_projects_per_agent + 1,)

    def test_inactive_agent_gets_zero_mask(self, small_env):
        """An inactive agent should receive an all-zero action mask."""
        env = small_env

        # Deactivate agent_0
        idx = env.agent_to_id["agent_0"]
        env.active_agents[idx] = 0

        mask = env._get_action_mask("agent_0")

        assert np.all(mask["choose_project"] == 0)
        assert np.all(mask["collaborate_with"] == 0)
        assert np.all(mask["put_effort"] == 0)

    def test_no_project_always_allowed(self, small_env):
        """The 'do nothing' action (index 0) for put_effort should always be allowed."""
        env = small_env
        mask = env._get_action_mask("agent_0")
        assert mask["put_effort"][0] == 1


# ---------------------------------------------------------------------------
# Reset reproducibility
# ---------------------------------------------------------------------------

class TestResetReproducibility:
    def test_same_seed_produces_identical_observations(self):
        """Resetting with the same seed must produce identical observations."""
        env = PeerGroupEnvironment(
            start_agents=8, max_agents=16, max_steps=50,
            n_groups=4, max_peer_group_size=4,
        )

        obs1, _ = env.reset(seed=42)
        rep1 = obs1["agent_0"]["observation"]["peer_reputation"].copy()
        proj1 = obs1["agent_0"]["observation"]["project_opportunities"]

        obs2, _ = env.reset(seed=42)
        rep2 = obs2["agent_0"]["observation"]["peer_reputation"].copy()
        proj2 = obs2["agent_0"]["observation"]["project_opportunities"]

        np.testing.assert_array_equal(rep1, rep2)
        for k in proj1:
            for kk in proj1[k]:
                np.testing.assert_array_equal(
                    np.asarray(proj1[k][kk]), np.asarray(proj2[k][kk])
                )

    def test_different_seed_produces_different_observations(self):
        """Resetting with a different seed should (almost certainly) differ."""
        env = PeerGroupEnvironment(
            start_agents=8, max_agents=16, max_steps=50,
            n_groups=4, max_peer_group_size=4,
        )
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=99)

        # Project opportunities should differ for different seeds
        p1 = obs1["agent_0"]["observation"]["project_opportunities"]["project_0"]
        p2 = obs2["agent_0"]["observation"]["project_opportunities"]["project_0"]
        any_diff = any(
            not np.array_equal(np.asarray(p1[k]), np.asarray(p2[k])) for k in p1
        )
        assert any_diff, "Different seeds should produce different project opportunities"

    def test_rng_is_generator_after_reset(self):
        """After reset, env.rng should be a proper np.random.Generator."""
        env = PeerGroupEnvironment(
            start_agents=4, max_agents=8, max_steps=10,
            n_groups=2, max_peer_group_size=4,
        )
        env.reset(seed=123)
        assert isinstance(env.rng, np.random.Generator)


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_returns_valid_structure(self, small_env):
        """A single step should return observations, rewards, terminations,
        truncations, and infos for all agents."""
        env = small_env
        obs, _ = env.reset(seed=42)

        # Build do-nothing actions for all agents
        actions = {}
        for agent in env.agents:
            mask = obs[agent]["action_mask"]
            actions[agent] = {
                "choose_project": 0,
                "collaborate_with": np.zeros_like(mask["collaborate_with"]),
                "put_effort": 0,
            }

        obs2, rewards, terminations, truncations, infos = env.step(actions)

        assert set(obs2.keys()) == set(env.agents)
        assert set(rewards.keys()) == set(env.agents)
        assert set(terminations.keys()) == set(env.agents)
        assert set(truncations.keys()) == set(env.agents)
        assert set(infos.keys()) == set(env.agents)

    def test_multi_step_trajectory_is_deterministic(self):
        """Running N steps with the same seed and actions must be reproducible."""
        def run(seed, n_steps=10):
            env = PeerGroupEnvironment(
                start_agents=8, max_agents=16, max_steps=100,
                n_groups=4, max_peer_group_size=4,
            )
            obs, _ = env.reset(seed=seed)
            fingerprint = []
            for _ in range(n_steps):
                actions = {}
                for agent in env.agents:
                    mask = obs[agent]["action_mask"]
                    actions[agent] = {
                        "choose_project": 0,
                        "collaborate_with": np.zeros_like(mask["collaborate_with"]),
                        "put_effort": 0,
                    }
                obs, rewards, terms, truncs, infos = env.step(actions)
                fingerprint.append(sum(rewards.values()))
            return fingerprint

        t1 = run(seed=42)
        t2 = run(seed=42)
        assert t1 == t2, "Same seed should produce identical reward trajectories"
