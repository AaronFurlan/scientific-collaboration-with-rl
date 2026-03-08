"""
Reproducibility tests — Policy population generation & heuristic determinism.

Covers:
- create_mixed_policy_population: deterministic with same seed
- create_per_group_policy_population: inherently deterministic
- Heuristic policies (careerist, orthodox, mass_producer, do_nothing):
  same observation → same action
- Cross-seed divergence for population assignment
"""

from __future__ import annotations

import pytest

from agent_policies import (
    careerist_policy,
    create_mixed_policy_population,
    create_per_group_policy_population,
    do_nothing_policy,
    mass_producer_policy,
    orthodox_scientist_policy,
)

from repro_helpers import BALANCED_DIST, THRESHOLDS, actions_equal, make_small_env


# ===========================================================================
# Population assignment
# ===========================================================================

class TestPolicyPopulationReproducibility:
    def test_mixed_population_deterministic(self):
        p1 = create_mixed_policy_population(20, BALANCED_DIST, seed=42)
        p2 = create_mixed_policy_population(20, BALANCED_DIST, seed=42)
        assert p1 == p2

    def test_mixed_population_different_seed(self):
        p1 = create_mixed_policy_population(20, BALANCED_DIST, seed=42)
        p2 = create_mixed_policy_population(20, BALANCED_DIST, seed=99)
        assert p1 != p2, "Different seeds should produce different shuffles"

    def test_per_group_population_deterministic(self):
        p1 = create_per_group_policy_population(20, BALANCED_DIST)
        p2 = create_per_group_policy_population(20, BALANCED_DIST)
        assert p1 == p2

    def test_population_length(self):
        dist = {"careerist": 0.5, "orthodox_scientist": 0.3, "mass_producer": 0.2}
        p = create_mixed_policy_population(50, dist, seed=0)
        assert len(p) == 50

    def test_mixed_population_contains_only_valid_archetypes(self):
        valid = {"careerist", "orthodox_scientist", "mass_producer"}
        p = create_mixed_policy_population(30, BALANCED_DIST, seed=42)
        assert set(p).issubset(valid)

    def test_mixed_population_cross_seed_differ(self):
        p1 = create_mixed_policy_population(30, BALANCED_DIST, seed=1)
        p2 = create_mixed_policy_population(30, BALANCED_DIST, seed=2)
        assert p1 != p2


# ===========================================================================
# Heuristic policy determinism
# ===========================================================================

class TestHeuristicPolicyDeterminism:
    """All heuristic policies must be deterministic given the same observation."""

    @pytest.fixture
    def sample_obs_and_mask(self):
        env = make_small_env()
        obs, _ = env.reset(seed=42)
        agent = env.possible_agents[0]
        return obs[agent]["observation"], obs[agent]["action_mask"]

    def test_careerist_deterministic(self, sample_obs_and_mask):
        o, m = sample_obs_and_mask
        a1 = careerist_policy(o, m, THRESHOLDS["careerist"])
        a2 = careerist_policy(o, m, THRESHOLDS["careerist"])
        assert actions_equal(a1, a2)

    def test_orthodox_deterministic(self, sample_obs_and_mask):
        o, m = sample_obs_and_mask
        a1 = orthodox_scientist_policy(o, m, THRESHOLDS["orthodox_scientist"])
        a2 = orthodox_scientist_policy(o, m, THRESHOLDS["orthodox_scientist"])
        assert actions_equal(a1, a2)

    def test_mass_producer_deterministic(self, sample_obs_and_mask):
        o, m = sample_obs_and_mask
        a1 = mass_producer_policy(o, m, THRESHOLDS["mass_producer"])
        a2 = mass_producer_policy(o, m, THRESHOLDS["mass_producer"])
        assert actions_equal(a1, a2)

    def test_do_nothing_deterministic(self, sample_obs_and_mask):
        o, m = sample_obs_and_mask
        a1 = do_nothing_policy(o, m)
        a2 = do_nothing_policy(o, m)
        assert actions_equal(a1, a2)

    def test_all_policies_return_expected_keys(self, sample_obs_and_mask):
        """Every heuristic policy must return choose_project, put_effort, collaborate_with."""
        o, m = sample_obs_and_mask
        expected_keys = {"choose_project", "put_effort", "collaborate_with"}
        for fn, threshold in [
            (careerist_policy, THRESHOLDS["careerist"]),
            (orthodox_scientist_policy, THRESHOLDS["orthodox_scientist"]),
            (mass_producer_policy, THRESHOLDS["mass_producer"]),
        ]:
            action = fn(o, m, threshold)
            assert set(action.keys()) >= expected_keys

