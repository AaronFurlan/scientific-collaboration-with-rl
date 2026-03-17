"""
Reproducibility tests — Utility classes.

Verifies that ``Area``, ``GaussianMixture``, and ``Project`` behave
deterministically when initialised with the same RNG seed.

Covers:
- Area.random_point / random_gaussian_point: identical with same rng
- GaussianMixture.sample: identical with same rng
- Project: identical internal rng state with same seed
- Cross-seed divergence for all three classes
"""

from __future__ import annotations

import numpy as np
import pytest

from env.area import Area
from env.project import Project
from env.utils import GaussianMixture


# ===========================================================================
# Area
# ===========================================================================

class TestAreaReproducibility:
    def test_random_points_identical(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        a1 = Area(xlim=(-1, 1), ylim=(-1, 1), rng=rng1)
        a2 = Area(xlim=(-1, 1), ylim=(-1, 1), rng=rng2)

        for i in range(20):
            p1 = a1.random_point()
            p2 = a2.random_point()
            assert p1 == pytest.approx(p2), f"Diverged at iteration {i}"

    def test_gaussian_points_identical(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        a1 = Area(xlim=(-1, 1), ylim=(-1, 1), rng=rng1)
        a2 = Area(xlim=(-1, 1), ylim=(-1, 1), rng=rng2)

        for i in range(20):
            p1 = a1.random_gaussian_point()
            p2 = a2.random_gaussian_point()
            assert p1 == pytest.approx(p2), f"Diverged at iteration {i}"

    def test_different_rng_diverges(self):
        a1 = Area(xlim=(-1, 1), ylim=(-1, 1), rng=np.random.default_rng(1))
        a2 = Area(xlim=(-1, 1), ylim=(-1, 1), rng=np.random.default_rng(2))
        p1 = a1.random_point()
        p2 = a2.random_point()
        assert p1 != pytest.approx(p2)

    def test_many_gaussian_points_cross_seed_differ(self):
        """Gaussian points from different seeds must diverge."""
        a1 = Area(xlim=(-1, 1), ylim=(-1, 1), rng=np.random.default_rng(10))
        a2 = Area(xlim=(-1, 1), ylim=(-1, 1), rng=np.random.default_rng(11))
        pts1 = [a1.random_gaussian_point() for _ in range(10)]
        pts2 = [a2.random_gaussian_point() for _ in range(10)]
        assert pts1 != pts2


# ===========================================================================
# GaussianMixture
# ===========================================================================

class TestGaussianMixtureReproducibility:
    def test_samples_identical(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        gm1 = GaussianMixture([0.5, 0.5], [0, 10], [1, 2], rng=rng1)
        gm2 = GaussianMixture([0.5, 0.5], [0, 10], [1, 2], rng=rng2)

        s1 = gm1.sample(50)
        s2 = gm2.sample(50)
        np.testing.assert_allclose(s1, s2, rtol=0, atol=0)

    def test_different_rng_diverges(self):
        gm1 = GaussianMixture(
            [0.5, 0.5], [0, 10], [1, 2], rng=np.random.default_rng(1),
        )
        gm2 = GaussianMixture(
            [0.5, 0.5], [0, 10], [1, 2], rng=np.random.default_rng(2),
        )
        s1 = gm1.sample(50)
        s2 = gm2.sample(50)
        assert not np.allclose(s1, s2)

    def test_single_component_deterministic(self):
        """A single-component mixture must still be reproducible."""
        gm1 = GaussianMixture([1.0], [0], [1], rng=np.random.default_rng(42))
        gm2 = GaussianMixture([1.0], [0], [1], rng=np.random.default_rng(42))
        np.testing.assert_allclose(gm1.sample(100), gm2.sample(100), rtol=0, atol=0)

    def test_single_component_cross_seed_differ(self):
        gm1 = GaussianMixture([1.0], [0], [1], rng=np.random.default_rng(1))
        gm2 = GaussianMixture([1.0], [0], [1], rng=np.random.default_rng(2))
        assert not np.allclose(gm1.sample(100), gm2.sample(100))


# ===========================================================================
# Project
# ===========================================================================

class TestProjectReproducibility:
    def test_project_same_rng_identical_state(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        p1 = Project(
            "p1", required_effort=10, prestige=0.5, time_window=50,
            peer_fit=np.ones(4), rng=rng1,
        )
        p2 = Project(
            "p2", required_effort=10, prestige=0.5, time_window=50,
            peer_fit=np.ones(4), rng=rng2,
        )
        # Both rngs should be in identical state after construction
        assert p1.rng.random() == pytest.approx(p2.rng.random())

    def test_project_different_rng_diverges(self):
        p1 = Project(
            "p1", required_effort=10, prestige=0.5, time_window=50,
            peer_fit=np.ones(4), rng=np.random.default_rng(1),
        )
        p2 = Project(
            "p2", required_effort=10, prestige=0.5, time_window=50,
            peer_fit=np.ones(4), rng=np.random.default_rng(2),
        )
        assert p1.rng.random() != pytest.approx(p2.rng.random())

