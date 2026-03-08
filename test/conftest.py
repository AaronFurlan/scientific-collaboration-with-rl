"""
Shared fixtures and markers for the test suite.

Registers:
  - ``integration`` marker for long-running / multi-step tests.

Provides fixtures:
  - ``small_env``       — a reset-ready PeerGroupEnvironment.
  - ``sample_obs_mask`` — (observation, action_mask) tuple from agent_0.
  - ``policy_fns``      — {archetype: callable} for the three heuristic policies.
"""

from __future__ import annotations

import pytest

from repro_helpers import build_policy_fns, make_small_env


# ---------------------------------------------------------------------------
# Custom markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration / long-running (deselect with '-m \"not integration\"')",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_env():
    """A small, deterministic PeerGroupEnvironment (reset with seed=42)."""
    env = make_small_env()
    env.reset(seed=42)
    return env


@pytest.fixture
def sample_obs_mask(small_env):
    """(observation_dict, action_mask_dict) for agent_0 after reset(seed=42)."""
    env = make_small_env()
    obs, _ = env.reset(seed=42)
    agent = env.possible_agents[0]
    return obs[agent]["observation"], obs[agent]["action_mask"]


@pytest.fixture
def policy_fns():
    """Dict mapping archetype names to their policy functions."""
    return build_policy_fns()

