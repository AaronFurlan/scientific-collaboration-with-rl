"""
Shared test utilities for reproducibility tests.

Provides deterministic helpers used across all ``test_repro_*.py`` modules.
Imported via ``from test.repro_helpers import …`` or via conftest fixtures.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from agent_policies import (
    create_mixed_policy_population,
    do_nothing_policy,
    get_policy_function,
)
from env.peer_group_environment import PeerGroupEnvironment
from rllib_single_agent_wrapper import RLLibSingleAgentWrapper


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default policy distribution used across reproducibility tests.
BALANCED_DIST: Dict[str, float] = {
    "careerist": 1 / 3,
    "orthodox_scientist": 1 / 3,
    "mass_producer": 1 / 3,
}

#: Default heuristic thresholds matching the training defaults.
THRESHOLDS: Dict[str, float] = {
    "careerist": 0.2,
    "orthodox_scientist": 0.8,
    "mass_producer": 22,
}

#: Default small-env parameters for fast, deterministic tests.
SMALL_ENV_DEFAULTS: Dict[str, Any] = dict(
    start_agents=6,
    max_agents=8,
    max_steps=30,
    n_groups=2,
    max_peer_group_size=4,
    n_projects_per_step=1,
    max_projects_per_agent=2,
    max_agent_age=100,
    max_rewardless_steps=30,
    acceptance_threshold=0.5,
    reward_mode="by_effort",
)


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def flatten_obs_deterministic(obs_dict: Dict[str, Any]) -> np.ndarray:
    """Recursively flatten an observation dict into a 1-D float32 array.

    Keys are iterated in **sorted** order at every nesting level to guarantee
    a deterministic, reproducible byte layout regardless of Python dict
    insertion order.
    """
    parts: List[np.ndarray] = []
    for key in sorted(obs_dict.keys()):
        v = obs_dict[key]
        if isinstance(v, dict):
            parts.append(flatten_obs_deterministic(v))
        else:
            parts.append(np.asarray(v, dtype=np.float32).ravel())
    return np.concatenate(parts) if parts else np.array([], dtype=np.float32)


def assert_nested_obs_equal(obs_a: Dict, obs_b: Dict, *, label: str = "") -> None:
    """Assert two nested observation dicts are element-wise identical.

    Compares each leaf array with ``np.testing.assert_array_equal`` for
    discrete/integer arrays and ``np.testing.assert_allclose`` for floats,
    iterating in sorted-key order for determinism.
    """
    prefix = f"{label}: " if label else ""
    for key in sorted(obs_a.keys()):
        va, vb = obs_a[key], obs_b[key]
        if isinstance(va, dict):
            assert_nested_obs_equal(va, vb, label=f"{prefix}{key}")
        else:
            arr_a = np.asarray(va)
            arr_b = np.asarray(vb)
            if np.issubdtype(arr_a.dtype, np.integer):
                np.testing.assert_array_equal(
                    arr_a, arr_b, err_msg=f"{prefix}{key} (int)"
                )
            else:
                np.testing.assert_allclose(
                    arr_a, arr_b, rtol=0, atol=0,
                    err_msg=f"{prefix}{key} (float)",
                )


# ---------------------------------------------------------------------------
# Action comparison
# ---------------------------------------------------------------------------

def actions_equal(a1: Dict, a2: Dict) -> bool:
    """Compare two action dicts that may contain numpy arrays."""
    if a1.keys() != a2.keys():
        return False
    for k in a1:
        v1, v2 = a1[k], a2[k]
        if isinstance(v1, np.ndarray):
            if not np.array_equal(v1, v2):
                return False
        elif v1 != v2:
            return False
    return True


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_small_env(**overrides) -> PeerGroupEnvironment:
    """Create a small, fast ``PeerGroupEnvironment`` for unit tests."""
    params = {**SMALL_ENV_DEFAULTS, **overrides}
    return PeerGroupEnvironment(**params)


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------

def apply_heuristic_action(
    policy_fns: Dict[str, Any],
    archetype: str,
    obs: Dict,
    mask: Dict,
) -> Dict:
    """Apply the correct heuristic policy for *archetype* to ``(obs, mask)``.

    Uses the default thresholds from :data:`THRESHOLDS`.
    """
    fn = policy_fns[archetype]
    return fn(obs, mask, THRESHOLDS[archetype])


def build_policy_fns() -> Dict[str, Any]:
    """Return ``{archetype_name: policy_function}`` for the three archetypes."""
    return {
        "careerist": get_policy_function("careerist"),
        "orthodox_scientist": get_policy_function("orthodox_scientist"),
        "mass_producer": get_policy_function("mass_producer"),
    }


# ---------------------------------------------------------------------------
# Wrapper factory
# ---------------------------------------------------------------------------

def make_wrapper_with_policies(seed: int) -> RLLibSingleAgentWrapper:
    """Build a ``RLLibSingleAgentWrapper`` with heuristic other-agent policies.

    Uses :func:`make_small_env` defaults and a balanced archetype distribution.
    """
    env = make_small_env()
    archetypes = create_mixed_policy_population(env.n_agents, BALANCED_DIST, seed=seed)
    fns = build_policy_fns()

    other_policies: Dict[str, Any] = {}
    for agent_id in env.possible_agents:
        if agent_id == "agent_0":
            continue
        idx = env.agent_to_id[agent_id]
        arch = archetypes[idx]
        fn = fns[arch]
        threshold = THRESHOLDS[arch]

        # Eagerly capture *fn* and *threshold* to avoid late-binding closure bugs.
        def _make(f=fn, t=threshold):
            return lambda nested: f(
                nested["observation"], nested["action_mask"], t
            )

        other_policies[agent_id] = _make()

    return RLLibSingleAgentWrapper(
        env,
        controlled_agent="agent_0",
        other_policies=other_policies,
        force_episode_horizon=30,
        topk_collab=3,
        topk_apply_to_all_agents=True,
    )


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_deterministic_episode(
    env: PeerGroupEnvironment,
    seed: int,
    n_steps: int = 15,
) -> Dict[str, Any]:
    """Run a short episode with deterministic heuristic policies.

    Returns a snapshot dict containing:
    - ``rewards``: list of per-step total rewards
    - ``obs_flat``: list of per-step full flattened observation arrays
    - ``agent_ages``, ``completed``, ``active``, ``peer_groups``: final env state
    """
    obs, _ = env.reset(seed=seed)

    policy_fns = build_policy_fns()
    archetypes = create_mixed_policy_population(
        env.n_agents, BALANCED_DIST, seed=seed,
    )

    all_rewards: List[float] = []
    all_obs_flat: List[np.ndarray] = []

    for _ in range(n_steps):
        actions = {}
        for agent in env.agents:
            idx = env.agent_to_id[agent]
            o = obs[agent]["observation"]
            m = obs[agent]["action_mask"]
            if env.active_agents[idx] == 0:
                actions[agent] = do_nothing_policy(o, m)
            else:
                actions[agent] = apply_heuristic_action(
                    policy_fns, archetypes[idx], o, m,
                )
        obs, rewards, terms, truncs, infos = env.step(actions)
        all_rewards.append(sum(rewards.values()))

        # Full deterministic fingerprint: flatten every agent's observation
        flat_parts = []
        for agent_id in sorted(obs.keys()):
            flat_parts.append(
                flatten_obs_deterministic(obs[agent_id]["observation"])
            )
        all_obs_flat.append(np.concatenate(flat_parts))

    return {
        "rewards": all_rewards,
        "obs_flat": all_obs_flat,
        "agent_ages": env.agent_ages.copy(),
        "completed": env.agent_completed_projects.copy(),
        "active": env.active_agents.copy(),
        "peer_groups": [list(g) for g in env.peer_groups],
    }


def run_simulation_pipeline(seed: int, n_steps: int = 20) -> Dict[str, Any]:
    """Run a full mini-simulation matching ``run_policy_simulation.py`` logic.

    Returns a dict of final aggregate metrics suitable for equality comparison.
    """
    env = make_small_env(max_steps=n_steps)
    archetypes = create_mixed_policy_population(env.n_agents, BALANCED_DIST, seed=seed)
    policy_fns = build_policy_fns()

    obs, _ = env.reset(seed=seed)
    cumulative_rewards = {a: 0.0 for a in env.possible_agents}

    for _ in range(n_steps):
        actions = {}
        for agent in env.agents:
            idx = env.agent_to_id[agent]
            o = obs[agent]["observation"]
            m = obs[agent]["action_mask"]
            if env.active_agents[idx] == 0:
                actions[agent] = do_nothing_policy(o, m)
            else:
                actions[agent] = apply_heuristic_action(
                    policy_fns, archetypes[idx], o, m,
                )

        obs, rewards, terms, truncs, infos = env.step(actions)
        for a, r in rewards.items():
            cumulative_rewards[a] += r

        if all(terms.values()):
            break

    return {
        "cumulative_rewards": cumulative_rewards,
        "completed": env.agent_completed_projects.tolist(),
        "active": env.active_agents.tolist(),
        "agent_ages": env.agent_ages.tolist(),
        "h_indexes": env.agent_h_indexes.tolist(),
        "n_projects": len(env.projects),
    }

