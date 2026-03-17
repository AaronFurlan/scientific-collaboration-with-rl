"""
Tests for run_policy_simulation_with_rlagent.py

Verifies (without loading a real checkpoint or starting Ray):
- EvalConfig: defaults, policy_distribution property, print_summary
- POLICY_CONFIGS: completeness and value constraints
- build_env: creates a valid PeerGroupEnvironment
- build_heuristic_population: returns correct list of archetype names
- make_policy_callable: returns callables for all archetypes + None/fallback
- build_other_policies: maps non-controlled agents to callables
- build_eval_wrapper: creates a valid RLLibSingleAgentWrapper
- compute_rl_action: deterministic and stochastic decoding from logits
- RLAgentStatus: snapshot, reward tracking, termination, formatting
- make_env_creator_from_config: produces a valid env creator
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from run_policy_simulation_with_rlagent import (
    POLICY_CONFIGS,
    EvalConfig,
    RLAgentStatus,
    build_env,
    build_eval_wrapper,
    build_heuristic_population,
    build_other_policies,
    compute_rl_action,
    make_env_creator_from_config,
    make_policy_callable,
)
from env.peer_group_environment import PeerGroupEnvironment
from rllib_single_agent_wrapper import RLLibSingleAgentWrapper


# ===========================================================================
# Shared fixtures
# ===========================================================================

@pytest.fixture
def default_cfg() -> EvalConfig:
    """A small-scale EvalConfig for fast unit tests."""
    return EvalConfig(
        checkpoint_path="models",
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
        reward_function="by_effort",
        prestige_threshold=0.2,
        novelty_threshold=0.8,
        effort_threshold=22,
        policy_config_name="Balanced",
        group_policy_homogenous=False,
        topk_collab=3,
        topk_apply_to_all_agents=True,
        controlled_agent_id="agent_0",
        deterministic=True,
        seed=42,
        output_file_prefix="test_sim",
    )


@pytest.fixture
def small_env(default_cfg) -> PeerGroupEnvironment:
    """A reset PeerGroupEnvironment for testing."""
    env = build_env(default_cfg)
    env.reset(seed=default_cfg.seed)
    return env


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
                f"Config '{name}' sums to {total}"
            )

    def test_all_keys_are_valid_archetypes(self):
        valid = {"careerist", "orthodox_scientist", "mass_producer"}
        for name, dist in POLICY_CONFIGS.items():
            assert set(dist.keys()) == valid, (
                f"Config '{name}' has unexpected keys: {set(dist.keys())}"
            )

    def test_all_values_non_negative(self):
        for name, dist in POLICY_CONFIGS.items():
            for k, v in dist.items():
                assert v >= 0, f"Config '{name}', key '{k}' has value {v}"


# ===========================================================================
# EvalConfig
# ===========================================================================

class TestEvalConfig:
    def test_defaults(self):
        cfg = EvalConfig()
        assert cfg.n_agents == 64
        assert cfg.controlled_agent_id == "agent_0"
        assert cfg.deterministic is True
        assert cfg.seed == 42
        assert cfg.policy_config_name == "Balanced"

    def test_policy_distribution_property(self):
        cfg = EvalConfig(policy_config_name="All Careerist")
        dist = cfg.policy_distribution
        assert dist["careerist"] == 1.0
        assert dist["orthodox_scientist"] == 0.0
        assert dist["mass_producer"] == 0.0

    def test_balanced_distribution(self):
        cfg = EvalConfig(policy_config_name="Balanced")
        dist = cfg.policy_distribution
        assert dist["careerist"] == pytest.approx(1 / 3)
        assert dist["orthodox_scientist"] == pytest.approx(1 / 3)
        assert dist["mass_producer"] == pytest.approx(1 / 3)

    def test_asdict_roundtrip(self):
        cfg = EvalConfig(seed=99, n_agents=16)
        d = asdict(cfg)
        assert d["seed"] == 99
        assert d["n_agents"] == 16
        assert isinstance(d, dict)

    def test_print_summary_does_not_crash(self, default_cfg, capsys):
        default_cfg.print_summary()
        captured = capsys.readouterr()
        assert "EVALUATION CONFIG" in captured.out
        assert default_cfg.controlled_agent_id in captured.out

    def test_invalid_policy_config_name(self):
        cfg = EvalConfig(policy_config_name="Nonexistent")
        with pytest.raises(KeyError):
            _ = cfg.policy_distribution


# ===========================================================================
# build_env
# ===========================================================================

class TestBuildEnv:
    def test_returns_peer_group_environment(self, default_cfg):
        env = build_env(default_cfg)
        assert isinstance(env, PeerGroupEnvironment)

    def test_env_has_correct_agent_count(self, default_cfg):
        env = build_env(default_cfg)
        assert len(env.possible_agents) == default_cfg.n_agents

    def test_env_reset_works(self, default_cfg):
        env = build_env(default_cfg)
        obs, infos = env.reset(seed=default_cfg.seed)
        assert isinstance(obs, dict)
        assert len(obs) > 0

    def test_env_parameters_match_config(self, default_cfg):
        env = build_env(default_cfg)
        assert env.n_groups == default_cfg.n_groups
        assert env.max_peer_group_size == default_cfg.max_peer_group_size
        assert env.n_projects_per_step == default_cfg.n_projects_per_step
        assert env.max_projects_per_agent == default_cfg.max_projects_per_agent


# ===========================================================================
# build_heuristic_population
# ===========================================================================

class TestBuildHeuristicPopulation:
    def test_returns_list_of_correct_length(self, default_cfg):
        policies = build_heuristic_population(default_cfg)
        assert isinstance(policies, list)
        assert len(policies) == default_cfg.n_agents

    def test_all_entries_are_valid_archetypes(self, default_cfg):
        valid = {"careerist", "orthodox_scientist", "mass_producer"}
        policies = build_heuristic_population(default_cfg)
        for p in policies:
            assert p in valid, f"Unexpected policy: {p}"

    def test_deterministic_with_same_seed(self, default_cfg):
        p1 = build_heuristic_population(default_cfg)
        p2 = build_heuristic_population(default_cfg)
        assert p1 == p2

    def test_homogenous_mode(self, default_cfg):
        default_cfg.group_policy_homogenous = True
        policies = build_heuristic_population(default_cfg)
        assert len(policies) == default_cfg.n_agents

    def test_all_careerist_distribution(self):
        cfg = EvalConfig(
            policy_config_name="All Careerist",
            n_agents=10,
            seed=0,
        )
        policies = build_heuristic_population(cfg)
        assert all(p == "careerist" for p in policies)


# ===========================================================================
# make_policy_callable
# ===========================================================================

class TestMakePolicyCallable:
    """Test that make_policy_callable returns working callables for each archetype."""

    @pytest.fixture
    def sample_nested_obs(self, small_env):
        """Return a valid nested observation from the environment."""
        obs, _ = small_env.reset(seed=42)
        agent_id = small_env.possible_agents[0]
        return obs[agent_id]

    def test_careerist_callable(self, default_cfg, sample_nested_obs):
        fn = make_policy_callable("careerist", default_cfg)
        action = fn(sample_nested_obs)
        assert isinstance(action, dict)
        assert "choose_project" in action

    def test_orthodox_callable(self, default_cfg, sample_nested_obs):
        fn = make_policy_callable("orthodox_scientist", default_cfg)
        action = fn(sample_nested_obs)
        assert isinstance(action, dict)
        assert "choose_project" in action

    def test_mass_producer_callable(self, default_cfg, sample_nested_obs):
        fn = make_policy_callable("mass_producer", default_cfg)
        action = fn(sample_nested_obs)
        assert isinstance(action, dict)
        assert "choose_project" in action

    def test_none_returns_do_nothing(self, default_cfg, sample_nested_obs):
        fn = make_policy_callable(None, default_cfg)
        action = fn(sample_nested_obs)
        assert isinstance(action, dict)
        assert "choose_project" in action

    def test_action_has_expected_keys(self, default_cfg, sample_nested_obs):
        fn = make_policy_callable("careerist", default_cfg)
        action = fn(sample_nested_obs)
        assert "choose_project" in action
        assert "put_effort" in action
        assert "collaborate_with" in action


# ===========================================================================
# build_other_policies
# ===========================================================================

class TestBuildOtherPolicies:
    def test_excludes_controlled_agent(self, small_env, default_cfg):
        agent_policies = build_heuristic_population(default_cfg)
        other = build_other_policies(small_env, agent_policies, default_cfg)
        assert default_cfg.controlled_agent_id not in other

    def test_covers_all_non_controlled_agents(self, small_env, default_cfg):
        agent_policies = build_heuristic_population(default_cfg)
        other = build_other_policies(small_env, agent_policies, default_cfg)
        expected = {a for a in small_env.possible_agents if a != default_cfg.controlled_agent_id}
        assert set(other.keys()) == expected

    def test_all_values_are_callable(self, small_env, default_cfg):
        agent_policies = build_heuristic_population(default_cfg)
        other = build_other_policies(small_env, agent_policies, default_cfg)
        for agent_id, fn in other.items():
            assert callable(fn), f"Policy for {agent_id} is not callable"

    def test_policies_produce_valid_actions(self, small_env, default_cfg):
        obs, _ = small_env.reset(seed=42)
        agent_policies = build_heuristic_population(default_cfg)
        other = build_other_policies(small_env, agent_policies, default_cfg)
        for agent_id, fn in other.items():
            if agent_id in obs:
                action = fn(obs[agent_id])
                assert isinstance(action, dict)
                assert "choose_project" in action


# ===========================================================================
# build_eval_wrapper
# ===========================================================================

class TestBuildEvalWrapper:
    def test_returns_wrapper(self, small_env, default_cfg):
        agent_policies = build_heuristic_population(default_cfg)
        other = build_other_policies(small_env, agent_policies, default_cfg)
        wrapper = build_eval_wrapper(small_env, other, default_cfg)
        assert isinstance(wrapper, RLLibSingleAgentWrapper)

    def test_wrapper_has_correct_spaces(self, small_env, default_cfg):
        from gymnasium.spaces import Box, Discrete
        agent_policies = build_heuristic_population(default_cfg)
        other = build_other_policies(small_env, agent_policies, default_cfg)
        wrapper = build_eval_wrapper(small_env, other, default_cfg)
        assert isinstance(wrapper.action_space, Discrete)
        assert isinstance(wrapper.observation_space, Box)

    def test_wrapper_reset_and_step(self, small_env, default_cfg):
        agent_policies = build_heuristic_population(default_cfg)
        other = build_other_policies(small_env, agent_policies, default_cfg)
        wrapper = build_eval_wrapper(small_env, other, default_cfg)
        obs, info = wrapper.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        action = wrapper.action_space.sample()
        obs2, reward, terminated, truncated, info = wrapper.step(action)
        assert isinstance(obs2, np.ndarray)


# ===========================================================================
# compute_rl_action
# ===========================================================================

class TestComputeRLAction:
    """Test compute_rl_action with a mock RLModule."""

    @pytest.fixture
    def mock_rl_module(self):
        """A mock RLModule that returns predictable logits."""
        module = MagicMock()

        def fake_forward(batch):
            # Return logits where action 2 has the highest value
            obs = batch["obs"]
            batch_size = obs.shape[0]
            logits = torch.zeros(batch_size, 5)
            logits[:, 2] = 10.0  # action 2 is strongly preferred
            return {"action_dist_inputs": logits}

        module.forward_inference = MagicMock(side_effect=fake_forward)
        return module

    @pytest.fixture
    def dummy_obs(self) -> np.ndarray:
        return np.zeros(10, dtype=np.float32)

    def test_deterministic_returns_argmax(self, mock_rl_module, dummy_obs):
        action = compute_rl_action(mock_rl_module, dummy_obs, deterministic=True)
        assert action == 2  # argmax of logits

    def test_deterministic_returns_int(self, mock_rl_module, dummy_obs):
        action = compute_rl_action(mock_rl_module, dummy_obs, deterministic=True)
        assert isinstance(action, int)

    def test_stochastic_returns_int(self, mock_rl_module, dummy_obs):
        action = compute_rl_action(mock_rl_module, dummy_obs, deterministic=False)
        assert isinstance(action, int)
        assert 0 <= action < 5

    def test_stochastic_mostly_picks_best_action(self, mock_rl_module, dummy_obs):
        """With logits strongly favouring action 2, stochastic should almost always pick 2."""
        actions = [
            compute_rl_action(mock_rl_module, dummy_obs, deterministic=False)
            for _ in range(50)
        ]
        assert actions.count(2) >= 45  # very high logit → near-deterministic

    def test_forward_inference_called_with_correct_shape(self, mock_rl_module, dummy_obs):
        compute_rl_action(mock_rl_module, dummy_obs, deterministic=True)
        call_args = mock_rl_module.forward_inference.call_args[0][0]
        assert "obs" in call_args
        assert call_args["obs"].shape == (1, 10)

    def test_uniform_logits_produce_varied_actions(self):
        """With uniform logits, stochastic should produce a variety of actions."""
        module = MagicMock()

        def uniform_forward(batch):
            obs = batch["obs"]
            batch_size = obs.shape[0]
            logits = torch.zeros(batch_size, 4)  # equal logits
            return {"action_dist_inputs": logits}

        module.forward_inference = MagicMock(side_effect=uniform_forward)
        obs = np.zeros(5, dtype=np.float32)

        actions = set()
        for _ in range(100):
            a = compute_rl_action(module, obs, deterministic=False)
            actions.add(a)
        # With 4 equal-probability actions and 100 samples, expect at least 2 distinct
        assert len(actions) >= 2


# ===========================================================================
# RLAgentStatus
# ===========================================================================

class TestRLAgentStatus:
    @pytest.fixture
    def status(self) -> RLAgentStatus:
        return RLAgentStatus(agent_id="agent_0", agent_idx=0)

    def test_initial_state(self, status):
        assert status.agent_id == "agent_0"
        assert status.agent_idx == 0
        assert status.total_reward == 0.0
        assert status.terminated_step is None
        assert status.is_active is True

    def test_record_step_reward(self, status):
        status.record_step_reward(1.5)
        assert status.step_reward == 1.5
        assert status.total_reward == 1.5
        status.record_step_reward(0.5)
        assert status.step_reward == 0.5
        assert status.total_reward == 2.0

    def test_record_termination(self, status):
        status.record_termination(step=25)
        assert status.terminated_step == 25

    def test_record_termination_only_first(self, status):
        """Only the first termination step is recorded."""
        status.record_termination(step=10)
        status.record_termination(step=20)
        assert status.terminated_step == 10

    def test_status_label_active(self, status):
        assert status.status_label == "ACTIVE"

    def test_status_label_terminated(self, status):
        status.record_termination(step=15)
        status.is_active = False  # snapshot() would set this
        assert "TERMINATED" in status.status_label
        assert "15" in status.status_label

    def test_format_log_line(self, status):
        line = status.format_log_line()
        assert "agent_0" in line
        assert "ACTIVE" in line

    def test_format_log_line_with_action(self, status):
        action = {
            "choose_project": 1,
            "put_effort": 3,
            "collaborate_with": np.array([1, 0, 1, 0], dtype=np.int8),
        }
        line = status.format_log_line(action_dict=action)
        assert "proj=1" in line
        assert "effort=3" in line
        assert "collab=2" in line

    def test_format_termination_banner(self, status):
        status.record_termination(step=42)
        status.total_reward = 5.5
        banner = status.format_termination_banner()
        assert "TERMINATED" in banner
        assert "42" in banner
        assert "5.5" in banner

    def test_snapshot(self, small_env, default_cfg):
        """Snapshot captures real env state."""
        status = RLAgentStatus(
            agent_id=default_cfg.controlled_agent_id,
            agent_idx=small_env.agent_to_id[default_cfg.controlled_agent_id],
        )
        status.snapshot(small_env)
        assert isinstance(status.is_active, bool)
        assert isinstance(status.age, int)
        assert isinstance(status.n_active_projects, int)

    def test_final_summary(self, small_env, default_cfg):
        status = RLAgentStatus(
            agent_id=default_cfg.controlled_agent_id,
            agent_idx=small_env.agent_to_id[default_cfg.controlled_agent_id],
        )
        status.total_reward = 3.0
        status.record_termination(step=10)
        summary = status.final_summary(small_env)
        assert isinstance(summary, dict)
        assert summary["rl_agent_total_reward"] == 3.0
        assert summary["rl_agent_terminated_step"] == 10
        assert "rl_agent_h_index" in summary
        assert "rl_agent_completed_projects" in summary
        assert "rl_agent_successful_projects" in summary
        assert "rl_agent_age" in summary


# ===========================================================================
# make_env_creator_from_config
# ===========================================================================

class TestMakeEnvCreatorFromConfig:
    def test_returns_callable(self, default_cfg):
        creator = make_env_creator_from_config(default_cfg)
        assert callable(creator)

    def test_creates_valid_env(self, default_cfg):
        import gymnasium as gym
        creator = make_env_creator_from_config(default_cfg)
        env = creator()
        assert isinstance(env, gym.Env)
        env.close()

    def test_env_reset_and_step(self, default_cfg):
        creator = make_env_creator_from_config(default_cfg)
        env = creator()
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs2, np.ndarray)
        assert isinstance(reward, (int, float))
        env.close()

    def test_env_spaces(self, default_cfg):
        from gymnasium.spaces import Box, Discrete
        creator = make_env_creator_from_config(default_cfg)
        env = creator()
        assert isinstance(env.action_space, Discrete)
        assert isinstance(env.observation_space, Box)
        assert len(env.observation_space.shape) == 1
        env.close()

    def test_multiple_resets_stable(self, default_cfg):
        """Multiple resets should not crash."""
        creator = make_env_creator_from_config(default_cfg)
        env = creator()
        for i in range(3):
            obs, info = env.reset(seed=i)
            assert isinstance(obs, np.ndarray)
        env.close()


# ===========================================================================
# Integration: full env lifecycle without checkpoint
# ===========================================================================

class TestIntegrationEnvLifecycle:
    """End-to-end test: build env, wrapper, run steps with random actions."""

    def test_full_episode_random_actions(self, default_cfg):
        env = build_env(default_cfg)
        agent_policies = build_heuristic_population(default_cfg)
        other = build_other_policies(env, agent_policies, default_cfg)
        wrapper = build_eval_wrapper(env, other, default_cfg)

        obs, info = wrapper.reset(seed=default_cfg.seed)
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < default_cfg.max_steps:
            action = wrapper.action_space.sample()
            obs, reward, terminated, truncated, info = wrapper.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

        assert steps > 0
        assert isinstance(total_reward, float)

    def test_rl_agent_status_through_episode(self, default_cfg):
        """RLAgentStatus tracking through a short episode."""
        env = build_env(default_cfg)
        env.reset(seed=default_cfg.seed)

        status = RLAgentStatus(
            agent_id=default_cfg.controlled_agent_id,
            agent_idx=env.agent_to_id[default_cfg.controlled_agent_id],
        )

        agent_policies = build_heuristic_population(default_cfg)
        other = build_other_policies(env, agent_policies, default_cfg)
        wrapper = build_eval_wrapper(env, other, default_cfg)
        obs, _ = wrapper.reset(seed=default_cfg.seed)

        for step in range(5):
            status.snapshot(env)
            action = wrapper.action_space.sample()
            obs, reward, terminated, truncated, info = wrapper.step(action)
            status.record_step_reward(reward)
            if terminated:
                status.record_termination(step)
                break

        # Status should have accumulated some data
        assert status.age >= 0
        assert isinstance(status.total_reward, float)

