"""
Reproducibility tests — Full simulation pipeline.

Verifies that a complete simulation run (environment + population assignment +
heuristic policies + stepping) produces identical aggregate results when
invoked with the same seed.

Covers:
- Two runs with same seed → identical final stats
- Different seed → divergent stats
- Parametric seeds → all reproducible
- Long episode stress test
"""

from __future__ import annotations

import pytest

from repro_helpers import run_simulation_pipeline


# ===========================================================================
# Full pipeline reproducibility
# ===========================================================================

class TestFullSimulationReproducibility:
    """Two complete simulation runs with the same seed → identical final stats."""

    @pytest.mark.integration
    def test_two_runs_identical(self):
        r1 = run_simulation_pipeline(seed=42)
        r2 = run_simulation_pipeline(seed=42)
        assert r1 == r2

    @pytest.mark.integration
    def test_different_seed_different_results(self):
        r1 = run_simulation_pipeline(seed=42)
        r2 = run_simulation_pipeline(seed=99)
        assert r1 != r2

    @pytest.mark.parametrize("seed", [0, 7, 42, 100, 2024])
    def test_parametric_seeds_reproducible(self, seed):
        r1 = run_simulation_pipeline(seed=seed, n_steps=15)
        r2 = run_simulation_pipeline(seed=seed, n_steps=15)
        assert r1 == r2

    @pytest.mark.integration
    def test_long_episode_reproducible(self):
        """Longer episode to stress-test RNG state consistency."""
        r1 = run_simulation_pipeline(seed=42, n_steps=50)
        r2 = run_simulation_pipeline(seed=42, n_steps=50)
        assert r1 == r2

