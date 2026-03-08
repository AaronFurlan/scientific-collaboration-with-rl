"""
Tests for the RLLibSingleAgentWrapper.

Verifies:
- wrapper construction from a mock multi-agent env
- reset() returns a flat numpy observation vector
- step() returns (obs_vec, reward, terminated, truncated, info)
- other-agent policies are called during step()
- action decoding produces valid env-action dicts
- wrapper reset is reproducible with the same seed
"""

import numpy as np
import pytest

from gymnasium.spaces import Box, Dict as GymDict, Discrete, MultiBinary
from rllib_single_agent_wrapper import RLLibSingleAgentWrapper


# ---------------------------------------------------------------------------
# Mock environment that satisfies all wrapper requirements
# ---------------------------------------------------------------------------

class MockParallelEnv:
    """Minimal PettingZoo-ParallelEnv-like mock with all attributes the
    RLLibSingleAgentWrapper inspects during __init__.
    """

    # --- Attributes required by the wrapper's __init__ ---
    n_projects_per_step = 1
    max_projects_per_agent = 2
    max_peer_group_size = 4
    n_steps = 10

    def __init__(self):
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = list(self.possible_agents)

    # --- Spaces ---

    def observation_space(self, agent):
        return GymDict({
            "vec": Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        })

    def action_space(self, agent):
        return GymDict({
            "choose_project": Discrete(self.n_projects_per_step + 1),
            "collaborate_with": MultiBinary(self.max_peer_group_size),
            "put_effort": Discrete(self.max_projects_per_agent + 1),
        })

    # --- Gymnasium API ---

    def reset(self, seed=None, options=None):
        self.agents = list(self.possible_agents)
        obs = {}
        for a in self.agents:
            obs[a] = {
                "observation": {
                    "vec": np.zeros(3, dtype=np.float32) if a == "agent_0"
                           else np.ones(3, dtype=np.float32),
                },
                "action_mask": {
                    "choose_project": np.ones(self.n_projects_per_step + 1, dtype=np.int8),
                    "collaborate_with": np.ones(self.max_peer_group_size, dtype=np.int8),
                    "put_effort": np.ones(self.max_projects_per_agent + 1, dtype=np.int8),
                },
            }
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions):
        # Simple reward: agent_0 gets 1.0 if it chose project 1, else 0.0
        a0_action = actions.get("agent_0", {})
        cp = a0_action.get("choose_project", 0) if isinstance(a0_action, dict) else 0
        reward0 = 1.0 if cp == 1 else 0.0
        rewards = {"agent_0": reward0, "agent_1": 0.0}

        obs = {}
        for a in self.agents:
            obs[a] = {
                "observation": {
                    "vec": np.full(3, 0.5, dtype=np.float32),
                },
                "action_mask": {
                    "choose_project": np.ones(self.n_projects_per_step + 1, dtype=np.int8),
                    "collaborate_with": np.ones(self.max_peer_group_size, dtype=np.int8),
                    "put_effort": np.ones(self.max_projects_per_agent + 1, dtype=np.int8),
                },
            }

        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, rewards, terminations, truncations, infos

    def render(self, mode="human"):
        return None

    def close(self):
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_env():
    return MockParallelEnv()


@pytest.fixture
def wrapper(mock_env):
    return RLLibSingleAgentWrapper(
        mock_env,
        controlled_agent="agent_0",
        other_policies={},
        force_episode_horizon=10,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestWrapperConstruction:
    def test_creates_discrete_action_space(self, wrapper):
        """Wrapper must expose a single Discrete action space (macro-action)."""
        assert isinstance(wrapper.action_space, Discrete)

    def test_creates_box_observation_space(self, wrapper):
        """Wrapper must expose a flat Box observation space."""
        assert isinstance(wrapper.observation_space, Box)
        assert len(wrapper.observation_space.shape) == 1

    def test_action_space_size_matches_encoding(self, wrapper):
        """Action space size = (n_projects+1) * (max_projects+1) * 2^max_peer_group_size."""
        expected = (
            (MockParallelEnv.n_projects_per_step + 1)
            * (MockParallelEnv.max_projects_per_agent + 1)
            * (1 << MockParallelEnv.max_peer_group_size)
        )
        assert wrapper.action_space.n == expected


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_returns_flat_numpy_array(self, wrapper):
        """reset() must return (obs_vec, info) where obs_vec is a 1D float32 array."""
        obs, info = wrapper.reset()

        assert isinstance(obs, np.ndarray), f"Expected ndarray, got {type(obs)}"
        assert obs.ndim == 1
        assert obs.dtype == np.float32

    def test_obs_matches_observation_space(self, wrapper):
        """The returned obs vector must have the same length as observation_space."""
        obs, _ = wrapper.reset()
        assert obs.shape == wrapper.observation_space.shape

    def test_info_is_dict(self, wrapper):
        """Info should be a dict."""
        _, info = wrapper.reset()
        assert isinstance(info, dict)

    def test_sets_current_controlled_agent(self, wrapper):
        """After reset, wrapper.current_controlled must be the requested agent."""
        wrapper.reset()
        assert wrapper.current_controlled == "agent_0"


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

class TestStep:
    def test_returns_correct_tuple(self, wrapper):
        """step() must return (obs, reward, terminated, truncated, info)."""
        wrapper.reset()
        result = wrapper.step(0)

        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert obs.ndim == 1
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_obs_shape_consistent_across_steps(self, wrapper):
        """Observation shape must stay the same after stepping."""
        obs0, _ = wrapper.reset()
        obs1, _, _, _, _ = wrapper.step(0)
        assert obs0.shape == obs1.shape

    def test_force_horizon_truncates(self, wrapper):
        """Wrapper should truncate after force_episode_horizon steps."""
        wrapper.reset()
        truncated = False
        for step in range(20):
            _, _, terminated, truncated, _ = wrapper.step(0)
            if truncated or terminated:
                break
        assert truncated, "Wrapper should have truncated the episode"
        assert step + 1 == 10, f"Expected truncation at step 10, got {step + 1}"


# ---------------------------------------------------------------------------
# Other-agent policy integration
# ---------------------------------------------------------------------------

class TestOtherPolicy:
    def test_other_policy_is_called(self, mock_env):
        """The wrapper must call the other-agent policy during step()."""
        called = {"count": 0}

        def other_policy(nested_obs):
            called["count"] += 1
            # Return a valid env-action dict
            return {
                "choose_project": 0,
                "collaborate_with": np.zeros(mock_env.max_peer_group_size, dtype=np.int8),
                "put_effort": 0,
            }

        w = RLLibSingleAgentWrapper(
            mock_env,
            controlled_agent="agent_0",
            other_policies={"agent_1": other_policy},
            force_episode_horizon=10,
        )
        w.reset()
        w.step(0)
        assert called["count"] >= 1, "Other-agent policy was never called"

    def test_other_policy_receives_nested_obs(self, mock_env):
        """The other-agent policy must receive the raw nested observation dict
        with 'observation' and 'action_mask' keys."""
        received = {}

        def other_policy(nested_obs):
            received["obs"] = nested_obs
            return {
                "choose_project": 0,
                "collaborate_with": np.zeros(mock_env.max_peer_group_size, dtype=np.int8),
                "put_effort": 0,
            }

        w = RLLibSingleAgentWrapper(
            mock_env,
            controlled_agent="agent_0",
            other_policies={"agent_1": other_policy},
            force_episode_horizon=10,
        )
        w.reset()
        w.step(0)

        assert "obs" in received, "Policy was never called"
        obs = received["obs"]
        assert isinstance(obs, dict)
        assert "observation" in obs
        assert "action_mask" in obs


# ---------------------------------------------------------------------------
# Action decoding
# ---------------------------------------------------------------------------

class TestActionDecoding:
    def test_action_0_is_do_nothing(self, wrapper):
        """Macro-action 0 should decode to choose_project=0, put_effort=0,
        collaborate_with all zeros (encoding: 0 = 0*... + 0*... + 0)."""
        decoded = wrapper.decode_action_id(0)
        assert decoded["choose_project"] == 0
        assert decoded["put_effort"] == 0
        assert sum(decoded["collaborate_with"]) == 0

    def test_roundtrip_encode_decode(self, wrapper):
        """Every valid action id should decode without error."""
        # Test a sample of action ids (full space may be very large)
        n = wrapper.action_space.n
        sample_ids = [0, 1, n // 2, n - 1]
        for aid in sample_ids:
            decoded = wrapper.decode_action_id(aid)
            assert 0 <= decoded["choose_project"] <= MockParallelEnv.n_projects_per_step
            assert 0 <= decoded["put_effort"] <= MockParallelEnv.max_projects_per_agent
            assert len(decoded["collaborate_with"]) == MockParallelEnv.max_peer_group_size

    def test_invalid_action_raises(self, wrapper):
        """Out-of-range action ids should raise ValueError."""
        with pytest.raises(ValueError):
            wrapper.decode_action_id(-1)
        with pytest.raises(ValueError):
            wrapper.decode_action_id(wrapper.action_space.n)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_obs(self, mock_env):
        """Resetting with the same seed should produce identical observations."""
        w = RLLibSingleAgentWrapper(
            mock_env,
            controlled_agent="agent_0",
            other_policies={},
            force_episode_horizon=10,
        )

        obs1, _ = w.reset(seed=42)
        obs2, _ = w.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)
