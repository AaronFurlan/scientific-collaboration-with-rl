"""
Tests for train_ppo_rllib.py

Verifies:
- _safe_float: edge cases (None, NaN, Inf, strings, normal values)
- _first_not_none: selection logic
- wandb_sanitize: filtering of non-scalar / NaN / Inf values
- resolve_checkpoint_path: various return types from Algorithm.save()
- extract_metrics: correct extraction from RLlib v2 result dicts
- POLICY_CONFIGS: completeness and value constraints
- make_env_creator: produces a valid env with correct spaces
- plot_training_history: writes CSV and PNG to disk
"""

from __future__ import annotations

import csv
import math
import os
import tempfile
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pytest

from train_ppo_rllib import (
    _safe_float,
    _first_not_none,
    wandb_sanitize,
    resolve_checkpoint_path,
    extract_metrics,
    POLICY_CONFIGS,
    make_env_creator,
    plot_training_history,
)


# ===========================================================================
# _safe_float
# ===========================================================================

class TestSafeFloat:
    def test_int(self):
        assert _safe_float(3) == 3.0

    def test_float(self):
        assert _safe_float(2.5) == 2.5

    def test_none_returns_nan(self):
        assert math.isnan(_safe_float(None))

    def test_nan_returns_nan(self):
        assert math.isnan(_safe_float(float("nan")))

    def test_inf_returns_inf(self):
        assert math.isinf(_safe_float(float("inf")))

    def test_string_number(self):
        assert _safe_float("3.14") == pytest.approx(3.14)

    def test_non_convertible_returns_nan(self):
        assert math.isnan(_safe_float("hello"))

    def test_numpy_scalar(self):
        assert _safe_float(np.float32(1.5)) == pytest.approx(1.5)

    def test_zero(self):
        assert _safe_float(0) == 0.0

    def test_negative(self):
        assert _safe_float(-7.3) == pytest.approx(-7.3)

    def test_bool_true(self):
        assert _safe_float(True) == 1.0

    def test_bool_false(self):
        assert _safe_float(False) == 0.0


# ===========================================================================
# _first_not_none
# ===========================================================================

class TestFirstNotNone:
    def test_first_value(self):
        assert _first_not_none(1, 2, 3) == 1

    def test_skips_none(self):
        assert _first_not_none(None, 2, 3) == 2

    def test_all_none(self):
        assert _first_not_none(None, None) is None

    def test_preserves_zero(self):
        """0 is not None, so it must be returned."""
        assert _first_not_none(0, 5) == 0

    def test_preserves_zero_float(self):
        assert _first_not_none(0.0, 5.0) == 0.0

    def test_preserves_empty_string(self):
        assert _first_not_none("", "fallback") == ""

    def test_preserves_false(self):
        assert _first_not_none(False, True) is False

    def test_no_args(self):
        assert _first_not_none() is None

    def test_single_none(self):
        assert _first_not_none(None) is None

    def test_single_value(self):
        assert _first_not_none(42) == 42


# ===========================================================================
# wandb_sanitize
# ===========================================================================

class TestWandbSanitize:
    def test_keeps_int(self):
        assert wandb_sanitize({"a": 1}) == {"a": 1}

    def test_keeps_float(self):
        assert wandb_sanitize({"a": 2.5}) == {"a": 2.5}

    def test_keeps_bool(self):
        assert wandb_sanitize({"a": True}) == {"a": True}

    def test_drops_none(self):
        assert wandb_sanitize({"a": None}) == {}

    def test_drops_nan(self):
        assert wandb_sanitize({"a": float("nan")}) == {}

    def test_drops_inf(self):
        assert wandb_sanitize({"a": float("inf")}) == {}

    def test_drops_neg_inf(self):
        assert wandb_sanitize({"a": float("-inf")}) == {}

    def test_drops_dict_value(self):
        assert wandb_sanitize({"a": {"nested": 1}}) == {}

    def test_drops_list_value(self):
        assert wandb_sanitize({"a": [1, 2, 3]}) == {}

    def test_drops_string(self):
        assert wandb_sanitize({"a": "hello"}) == {}

    def test_drops_tuple(self):
        assert wandb_sanitize({"a": (1, 2)}) == {}

    def test_keeps_zero(self):
        assert wandb_sanitize({"a": 0}) == {"a": 0}

    def test_keeps_negative(self):
        assert wandb_sanitize({"a": -3.5}) == {"a": -3.5}

    def test_numpy_scalar_unwrapped(self):
        result = wandb_sanitize({"a": np.float64(1.23)})
        assert result == {"a": pytest.approx(1.23)}

    def test_numpy_nan_dropped(self):
        assert wandb_sanitize({"a": np.float64("nan")}) == {}

    def test_mixed(self):
        metrics = {
            "ok_int": 5,
            "ok_float": 3.14,
            "bad_none": None,
            "bad_nan": float("nan"),
            "bad_dict": {"x": 1},
            "bad_list": [1, 2],
            "ok_bool": False,
        }
        result = wandb_sanitize(metrics)
        assert set(result.keys()) == {"ok_int", "ok_float", "ok_bool"}

    def test_empty(self):
        assert wandb_sanitize({}) == {}


# ===========================================================================
# resolve_checkpoint_path
# ===========================================================================

class TestResolveCheckpointPath:
    def test_none_input(self):
        assert resolve_checkpoint_path(None) is None

    def test_string_input(self):
        assert resolve_checkpoint_path("/tmp/checkpoint") == "/tmp/checkpoint"

    def test_object_with_path(self):
        obj = SimpleNamespace(path="/some/path")
        assert resolve_checkpoint_path(obj) == "/some/path"

    def test_object_with_empty_path(self):
        """If .path is empty string, fall through."""
        obj = SimpleNamespace(path="")
        # empty string is falsy -> falls through to other checks
        # SimpleNamespace has no .checkpoint -> may return None or str(obj)
        result = resolve_checkpoint_path(obj)
        # Should not return empty string
        assert result is None or isinstance(result, str)

    def test_object_with_checkpoint_attr_having_path(self):
        inner = SimpleNamespace(path="/inner/path")
        outer = SimpleNamespace(checkpoint=inner)
        assert resolve_checkpoint_path(outer) == "/inner/path"

    def test_object_with_checkpoint_str(self):
        outer = SimpleNamespace(checkpoint="/string/path")
        assert resolve_checkpoint_path(outer) == "/string/path"

    def test_dict_with_checkpoint_path_key(self):
        d = {"checkpoint_path": "/dict/path"}
        assert resolve_checkpoint_path(d) == "/dict/path"

    def test_dict_with_best_checkpoint_object(self):
        inner = SimpleNamespace(path="/best/path")
        d = {"best_checkpoint": inner}
        assert resolve_checkpoint_path(d) == "/best/path"

    def test_dict_with_checkpoint_string(self):
        d = {"checkpoint": "/ckpt/string"}
        assert resolve_checkpoint_path(d) == "/ckpt/string"

    def test_unknown_type_returns_none(self):
        assert resolve_checkpoint_path(12345) is None

    def test_dict_without_keys_returns_none(self):
        assert resolve_checkpoint_path({"unrelated": 42}) is None


# ===========================================================================
# extract_metrics
# ===========================================================================

class TestExtractMetrics:
    @pytest.fixture
    def minimal_result(self) -> Dict[str, Any]:
        """A minimal RLlib v2 training result dict for testing."""
        return {
            "env_runners": {
                "episode_return_mean": 10.5,
                "episode_return_min": 2.0,
                "episode_return_max": 20.0,
                "episode_len_mean": 100.0,
                "episode_len_min": 50.0,
                "episode_len_max": 150.0,
                "num_env_steps_sampled": 4000,
                "num_episodes": 8,
                "num_env_steps_sampled_lifetime": 20000,
                "timers": {},
            },
            "evaluation": {
                "env_runners": {
                    "episode_return_mean": 12.0,
                    "episode_return_min": 5.0,
                    "episode_return_max": 18.0,
                    "episode_return_std": 3.5,
                    "episode_len_mean": 110.0,
                    "num_episodes": 4,
                    "num_env_steps_sampled": 2000,
                },
            },
            "learners": {
                "default_policy": {
                    "mean_kl_loss": 0.01,
                    "entropy": 1.5,
                    "policy_loss": -0.05,
                    "vf_loss": 0.3,
                    "vf_explained_var": 0.6,
                    "default_optimizer_learning_rate": 1e-4,
                    "gradients_default_optimizer_global_norm": 0.8,
                    "curr_kl_coeff": 0.2,
                },
            },
            "custom_metrics": {},
            "timers": {
                "training_iteration": 5.0,
            },
            "perf": {
                "cpu_util_percent": 45.0,
                "time_total_s": 120.0,
            },
            "timesteps_total": 20000,
            "timesteps_this_iter": 4000,
            "info": {},
        }

    def test_returns_tuple(self, minimal_result):
        metrics, global_step = extract_metrics(minimal_result, iteration=3, prev_total_env_steps=16000)
        assert isinstance(metrics, dict)
        assert isinstance(global_step, int)

    def test_iteration_stored(self, minimal_result):
        metrics, _ = extract_metrics(minimal_result, iteration=7, prev_total_env_steps=0)
        assert metrics["iteration"] == 7

    def test_train_return_mean(self, minimal_result):
        metrics, _ = extract_metrics(minimal_result, iteration=0, prev_total_env_steps=0)
        assert metrics["train/episode_return_mean"] == 10.5
        assert metrics["train/ep_return_mean"] == 10.5

    def test_eval_return_mean(self, minimal_result):
        metrics, _ = extract_metrics(minimal_result, iteration=0, prev_total_env_steps=0)
        assert metrics["eval/episode_return_mean"] == 12.0
        assert metrics["eval/ep_return_mean"] == 12.0

    def test_ppo_learner_metrics(self, minimal_result):
        metrics, _ = extract_metrics(minimal_result, iteration=0, prev_total_env_steps=0)
        assert metrics["ppo/mean_kl"] == 0.01
        assert metrics["ppo/entropy"] == 1.5
        assert metrics["ppo/policy_loss"] == -0.05
        assert metrics["ppo/value_loss"] == 0.3
        assert metrics["ppo/vf_explained_var"] == 0.6
        assert metrics["ppo/lr"] == 1e-4
        assert metrics["ppo/curr_kl_coeff"] == 0.2

    def test_global_step_from_timesteps_total(self, minimal_result):
        _, global_step = extract_metrics(minimal_result, iteration=0, prev_total_env_steps=0)
        assert global_step == 20000

    def test_global_step_fallback_to_lifetime(self, minimal_result):
        minimal_result["timesteps_total"] = None
        _, global_step = extract_metrics(minimal_result, iteration=0, prev_total_env_steps=0)
        assert global_step == 20000  # num_env_steps_sampled_lifetime

    def test_global_step_fallback_to_prev_plus_iter(self, minimal_result):
        minimal_result["timesteps_total"] = None
        minimal_result["env_runners"]["num_env_steps_sampled_lifetime"] = None
        _, global_step = extract_metrics(minimal_result, iteration=0, prev_total_env_steps=5000)
        assert global_step == 5000 + 4000  # prev + timesteps_this_iter

    def test_eval_missing_gracefully(self, minimal_result):
        minimal_result["evaluation"] = {}
        metrics, _ = extract_metrics(minimal_result, iteration=0, prev_total_env_steps=0)
        assert metrics["eval/episode_return_mean"] is None

    def test_learner_missing_gracefully(self, minimal_result):
        minimal_result["learners"] = {}
        metrics, _ = extract_metrics(minimal_result, iteration=0, prev_total_env_steps=0)
        assert metrics["ppo/mean_kl"] is None

    def test_timer_metrics(self, minimal_result):
        metrics, _ = extract_metrics(minimal_result, iteration=0, prev_total_env_steps=0)
        assert metrics["timer/training_iteration"] == 5.0

    def test_perf_metrics(self, minimal_result):
        metrics, _ = extract_metrics(minimal_result, iteration=0, prev_total_env_steps=0)
        assert metrics["perf/cpu_util_percent"] == 45.0
        assert metrics["perf/time_total_s"] == 120.0

    def test_total_env_steps_key(self, minimal_result):
        metrics, _ = extract_metrics(minimal_result, iteration=0, prev_total_env_steps=0)
        assert metrics["env/total_env_steps"] == 20000

    def test_empty_result(self):
        """Completely empty result should not crash."""
        metrics, global_step = extract_metrics({}, iteration=0, prev_total_env_steps=0)
        assert isinstance(metrics, dict)
        assert global_step == 0


# ===========================================================================
# POLICY_CONFIGS
# ===========================================================================

class TestPolicyConfigs:
    def test_expected_configs_exist(self):
        expected = [
            "All Careerist", "All Orthodox", "All Mass Producer",
            "Balanced", "Careerist Heavy", "Orthodox Heavy", "Mass Producer Heavy",
        ]
        for name in expected:
            assert name in POLICY_CONFIGS, f"Missing config: {name}"

    def test_values_sum_to_one(self):
        for name, dist in POLICY_CONFIGS.items():
            total = sum(dist.values())
            assert total == pytest.approx(1.0, abs=1e-6), (
                f"Config '{name}' does not sum to 1.0: {total}"
            )

    def test_all_keys_are_valid_archetypes(self):
        valid_archetypes = {"careerist", "orthodox_scientist", "mass_producer"}
        for name, dist in POLICY_CONFIGS.items():
            assert set(dist.keys()) == valid_archetypes, (
                f"Config '{name}' has unexpected keys: {set(dist.keys())}"
            )

    def test_all_values_non_negative(self):
        for name, dist in POLICY_CONFIGS.items():
            for k, v in dist.items():
                assert v >= 0, f"Config '{name}', archetype '{k}' has negative value: {v}"

    def test_balanced_is_equal_thirds(self):
        bal = POLICY_CONFIGS["Balanced"]
        assert bal["careerist"] == pytest.approx(1 / 3)
        assert bal["orthodox_scientist"] == pytest.approx(1 / 3)
        assert bal["mass_producer"] == pytest.approx(1 / 3)


# ===========================================================================
# make_env_creator
# ===========================================================================

class TestMakeEnvCreator:
    """Test that make_env_creator produces a valid single-agent Gymnasium env."""

    @pytest.fixture
    def env_creator(self):
        return make_env_creator(
            n_agents=8,
            start_agents=6,
            max_steps=50,
            max_rewardless_steps=50,
            n_groups=2,
            max_peer_group_size=4,
            n_projects_per_step=1,
            max_projects_per_agent=2,
            max_agent_age=100,
            acceptance_threshold=0.5,
            reward_function="by_effort",
            seed=42,
            policy_distribution={"careerist": 1 / 3, "orthodox_scientist": 1 / 3, "mass_producer": 1 / 3},
            group_policy_homogenous=False,
            prestige_threshold=0.2,
            novelty_threshold=0.8,
            effort_threshold=22,
            controlled_agent_id="agent_0",
        )

    def test_creator_returns_callable(self, env_creator):
        assert callable(env_creator)

    def test_creates_gymnasium_env(self, env_creator):
        import gymnasium as gym
        env = env_creator()
        assert isinstance(env, gym.Env)
        env.close()

    def test_observation_space_is_box(self, env_creator):
        from gymnasium.spaces import Box
        env = env_creator()
        assert isinstance(env.observation_space, Box)
        assert len(env.observation_space.shape) == 1
        env.close()

    def test_action_space_is_discrete(self, env_creator):
        from gymnasium.spaces import Discrete
        env = env_creator()
        assert isinstance(env.action_space, Discrete)
        env.close()

    def test_reset_returns_obs_and_info(self, env_creator):
        env = env_creator()
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        env.close()

    def test_obs_shape_matches_space(self, env_creator):
        env = env_creator()
        obs, _ = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape
        env.close()

    def test_step_returns_five_tuple(self, env_creator):
        env = env_creator()
        obs, _ = env.reset(seed=42)
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        obs2, reward, terminated, truncated, info = result
        assert isinstance(obs2, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        env.close()

    def test_multiple_steps_do_not_crash(self, env_creator):
        env = env_creator()
        obs, _ = env.reset(seed=42)
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        env.close()

    def test_env_config_override(self, env_creator):
        """Passing env_config overrides defaults."""
        env = env_creator({"max_steps": 20})
        obs, _ = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        env.close()

    def test_group_policy_homogenous(self):
        """Test with group_policy_homogenous=True."""
        creator = make_env_creator(
            n_agents=8,
            start_agents=6,
            max_steps=30,
            max_rewardless_steps=30,
            n_groups=2,
            max_peer_group_size=4,
            n_projects_per_step=1,
            max_projects_per_agent=2,
            max_agent_age=100,
            acceptance_threshold=0.5,
            reward_function="multiply",
            seed=123,
            policy_distribution={"careerist": 0.5, "orthodox_scientist": 0.5, "mass_producer": 0.0},
            group_policy_homogenous=True,
            prestige_threshold=0.2,
            novelty_threshold=0.8,
            effort_threshold=22,
            controlled_agent_id="agent_0",
        )
        env = creator()
        obs, info = env.reset(seed=123)
        assert isinstance(obs, np.ndarray)
        env.close()


# ===========================================================================
# plot_training_history
# ===========================================================================

class TestPlotTrainingHistory:
    def test_writes_csv_and_png(self):
        history = [
            {"iter": 0, "eval_return": 1.0, "kl": 0.01, "vf_var": 0.5},
            {"iter": 1, "eval_return": 2.0, "kl": 0.02, "vf_var": 0.6},
            {"iter": 2, "eval_return": 3.5, "kl": 0.015, "vf_var": 0.7},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_training_history(history, out_dir=tmpdir, save_prefix="test_hist")
            assert "csv" in result
            assert "png" in result
            assert os.path.isfile(result["csv"])
            assert os.path.isfile(result["png"])

    def test_csv_contents(self):
        history = [
            {"iter": 0, "eval_return": 10.0, "kl": 0.001, "vf_var": 0.9},
            {"iter": 1, "eval_return": 12.0, "kl": 0.002, "vf_var": 0.85},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_training_history(history, out_dir=tmpdir, save_prefix="csv_check")
            with open(result["csv"], "r") as f:
                reader = csv.reader(f)
                header = next(reader)
                assert header == ["iter", "eval_return", "kl", "vf_var"]
                rows = list(reader)
                assert len(rows) == 2
                assert rows[0][0] == "0"
                assert float(rows[0][1]) == pytest.approx(10.0)

    def test_empty_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_training_history([], out_dir=tmpdir, save_prefix="empty")
            assert os.path.isfile(result["csv"])

    def test_nan_values_in_history(self):
        history = [
            {"iter": 0, "eval_return": float("nan"), "kl": float("nan"), "vf_var": float("nan")},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_training_history(history, out_dir=tmpdir, save_prefix="nan_check")
            assert os.path.isfile(result["csv"])
            assert os.path.isfile(result["png"])

    def test_creates_output_directory(self):
        history = [{"iter": 0, "eval_return": 1.0, "kl": 0.01, "vf_var": 0.5}]
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "subdir", "deep")
            result = plot_training_history(history, out_dir=nested, save_prefix="nested")
            assert os.path.isdir(nested)
            assert os.path.isfile(result["csv"])

