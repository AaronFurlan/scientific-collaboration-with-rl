"""
train_ppo_rllib.py

Close in spirit to `run_policy_simulation.py`, but uses RLlib PPO to control
ONE agent while the rest of the population follows fixed (hand-crafted) policies.

Key similarities to run_policy_simulation.py:
- You pick a policy distribution (careerist / orthodox_scientist / mass_producer)
  for the *other* agents (fixed policies).
- You configure the same environment knobs (n_agents, max_steps, n_groups, etc.).
- You run "episodes" (rollouts) for a fixed horizon.

Key differences:
- RLlib PPO controls exactly one "controlled agent" (by default agent_0).
- The wrapper handles:
  - observation flattening (for RLlib encoders),
  - macro-action encoding/decoding,
  - action-mask repair,
  - fixed policies for non-controlled agents.

Install:
  pip install "ray[rllib]" torch
  # or
  pip install "ray[rllib]" tensorflow

Run:
  python train_ppo_rllib.py --iterations 5 --policy-config Balanced
"""

from __future__ import annotations

import argparse
import math
import os
import csv
import wandb
import ray
import matplotlib.pyplot as plt
import numpy as np
import torch

from ray import tune
from typing import Any, Callable, Dict, Optional
from ray.rllib.algorithms.ppo import PPOConfig
from lightning.pytorch import seed_everything

from agent_policies import (
    create_mixed_policy_population,
    create_per_group_policy_population,
    do_nothing_policy,
    get_policy_function,
)

from env.peer_group_environment import PeerGroupEnvironment
from rllib_single_agent_wrapper import RLLibSingleAgentWrapper
from callbacks.debug_actions_callback import make_action_info_callback
from callbacks.papers_metrics_callback import PapersMetricsCallback
from checkpoint_utils import build_checkpoint_path


# Helper: safe float conversion
def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        if isinstance(x, float):
            return float(x)
        return float(x)
    except Exception:
        return float("nan")


def _first_not_none(*values: Any) -> Any:
    """Return the first value that is not None. Unlike ``or``, preserves 0 and 0.0."""
    for v in values:
        if v is not None:
            return v
    return None


def wandb_sanitize(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Return a WandB-safe copy of metrics.

    - Keep only scalar ints/floats/bools.
    - Drop None and NaN/Inf.
    - Convert numpy scalar types to Python scalars if numpy is available.
    - Ignore dict/list/array-like structures (e.g. histograms).
    """

    safe: Dict[str, Any] = {}
    for k, v in metrics.items():
        # Skip obvious non-scalars
        if isinstance(v, (dict, list, tuple, set)):
            continue

        # Unwrap numpy scalars if possible
        if np is not None and isinstance(v, (np.generic,)):
            v = v.item()

        # Only allow basic scalar types
        if not isinstance(v, (int, float, bool)):
            continue

        # Drop NaN/Inf
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            continue

        safe[k] = v

    return safe


def resolve_checkpoint_path(save_result: Any) -> Optional[str]:
    """Best-effort extraction of a filesystem path from PPOAlgo.save() result.

    RLlib versions differ in what Algorithm.save() returns. This helper tries to
    find a usable path without ever assuming that str(obj) is a real path.
    """
    if save_result is None:
        return None

    # Common: Checkpoint-like objects with a .path attribute
    if hasattr(save_result, "path"):
        path = getattr(save_result, "path", None)
        if path:
            return str(path)

    # Some wrappers expose a .checkpoint with .path
    if hasattr(save_result, "checkpoint"):
        cp = getattr(save_result, "checkpoint", None)
        if cp is not None:
            if hasattr(cp, "path"):
                path = getattr(cp, "path", None)
                if path:
                    return str(path)
            if isinstance(cp, str):
                return cp

    # Direct string path
    if isinstance(save_result, str):
        return save_result

    # Dict-like training result that contains checkpoint info
    if isinstance(save_result, dict):
        for key in ("checkpoint_path", "best_checkpoint", "checkpoint"):
            val = save_result.get(key)
            if hasattr(val, "path"):
                path = getattr(val, "path", None)
                if path:
                    return str(path)
            if isinstance(val, str):
                return val

    # Fallback: give up instead of returning a bogus path
    return None


def extract_metrics(result: Dict[str, Any], iteration: int, prev_total_env_steps: int) -> tuple[Dict[str, Any], int]:
    """Extract a flat metrics dict from an RLlib v2 training result.

    Handles train/eval env_runners, learner stats, timers, perf, and
    custom_metrics. Also computes a stable global step counter based on
    timesteps_total (preferred) or num_env_steps_sampled_lifetime/this_iter.
    """
    metrics: Dict[str, Any] = {}

    # --------------------
    # Train env_runners
    # --------------------
    train_env = result.get("env_runners", {}) or {}

    train_return_mean = train_env.get("episode_return_mean")
    train_return_min = train_env.get("episode_return_min")
    train_return_max = train_env.get("episode_return_max")
    train_len_mean = train_env.get("episode_len_mean")
    train_len_min = train_env.get("episode_len_min")
    train_len_max = train_env.get("episode_len_max")
    train_num_env_steps = train_env.get("num_env_steps_sampled")
    train_num_episodes = train_env.get("num_episodes")

    # Timers inside env_runners (may or may not exist)
    train_env_timers = train_env.get("timers", {}) or {}
    env_step_timer = train_env_timers.get("env_step_timer")
    env_reset_timer = train_env_timers.get("env_reset_timer")
    rlmodule_inference_timer = train_env_timers.get("rl_module_inference_timer") or train_env_timers.get("rlmodule_inference_timer")
    sample_timer = train_env_timers.get("sample")

    num_env_steps_sampled_lifetime = train_env.get("num_env_steps_sampled_lifetime")

    metrics.update(
        {
            "train/episode_return_mean": train_return_mean,
            "train/episode_return_min": train_return_min,
            "train/episode_return_max": train_return_max,
            "train/episode_len_mean": train_len_mean,
            "train/episode_len_min": train_len_min,
            "train/episode_len_max": train_len_max,
            "train/num_env_steps_sampled": train_num_env_steps,
            "train/num_episodes": train_num_episodes,
            "train/env_step_timer": env_step_timer,
            "train/env_reset_timer": env_reset_timer,
            "train/rlmodule_inference_timer": rlmodule_inference_timer,
            "train/sample_timer": sample_timer,
            "train/num_env_steps_sampled_lifetime": num_env_steps_sampled_lifetime,
        }
    )

    # --------------------
    # Evaluation env_runners
    # --------------------
    eval_block = result.get("evaluation", {}) or {}
    eval_env = eval_block.get("env_runners", {}) or {}

    eval_return_mean = eval_env.get("episode_return_mean")
    eval_return_min = eval_env.get("episode_return_min")
    eval_return_max = eval_env.get("episode_return_max")
    eval_return_std = eval_env.get("episode_return_std")
    eval_len_mean = eval_env.get("episode_len_mean")
    eval_num_episodes = eval_env.get("num_episodes")
    eval_num_env_steps = eval_env.get("num_env_steps_sampled")
    eval_episodes_with_reward_frac = eval_env.get("episodes_with_reward_frac")

    metrics.update(
        {
            "eval/episode_return_mean": eval_return_mean,
            "eval/episode_return_min": eval_return_min,
            "eval/episode_return_max": eval_return_max,
            "eval/episode_return_std": eval_return_std,
            "eval/episode_len_mean": eval_len_mean,
            "eval/num_episodes": eval_num_episodes,
            "eval/num_env_steps_sampled": eval_num_env_steps,
            "eval/episodes_with_reward_frac": eval_episodes_with_reward_frac,
        }
    )

    # Explicit ep_return aliases for convenience
    metrics.update(
        {
            "eval/ep_return_mean": eval_return_mean,
            "eval/ep_return_min": eval_return_min,
            "eval/ep_return_max": eval_return_max,
            "train/ep_return_mean": train_return_mean,
            "train/ep_return_min": train_return_min,
            "train/ep_return_max": train_return_max,
        }
    )

    # --------------------
    # PPO learner internals
    # --------------------
    learner = result.get("learners", {}).get("default_policy", {}) or {}
    mean_kl = learner.get("mean_kl_loss")
    entropy = learner.get("entropy")
    policy_loss = learner.get("policy_loss")
    value_loss = learner.get("vf_loss")
    vf_explained_var = learner.get("vf_explained_var")

    # learning rate: try modern key, fall back to older ones
    lr = (
        learner.get("default_optimizer_learning_rate")
        or learner.get("cur_lr")
        or learner.get("learning_rate")
    )
    grad_global_norm = learner.get("gradients_default_optimizer_global_norm")
    curr_kl_coeff = learner.get("curr_kl_coeff")

    metrics.update(
        {
            "ppo/mean_kl": mean_kl,
            "ppo/entropy": entropy,
            "ppo/policy_loss": policy_loss,
            "ppo/value_loss": value_loss,
            "ppo/vf_explained_var": vf_explained_var,
            "ppo/lr": lr,
            "ppo/gradients_default_optimizer_global_norm": grad_global_norm,
            "ppo/curr_kl_coeff": curr_kl_coeff,
        }
    )

    # --------------------
    # Custom env metrics from callbacks
    # --------------------
    custom = result.get("custom_metrics", {}) or {}

    metrics.update(
        {
            # sparse reward debug
            "env/reward_nonzero_frac": custom.get("reward_nonzero_frac_mean"),
            "env/steps_until_first_reward": custom.get("steps_until_first_reward_mean"),
            "env/positive_reward_events_per_episode": custom.get("positive_reward_events_per_episode_mean"),
            "env/rewardless_termination_frac": custom.get("rewardless_termination_flag_mean"),
            "env/truncated_frac": custom.get("truncated_flag_mean"),
            "env/terminated_frac": custom.get("terminated_flag_mean"),
            # environment dynamics
            "env/published_projects_count": custom.get("published_projects_count_mean"),
            "env/accepted_projects_count": custom.get("accepted_projects_count_mean"),
            "env/rejected_projects_count": custom.get("rejected_projects_count_mean"),
            "env/agent_age_at_end": custom.get("agent_age_at_end_mean"),
            "env/agent_eliminated_flag": custom.get("agent_eliminated_flag_mean"),
            # mask repair + action diagnostics (if your wrapper/Callbacks provide them)
            "mask/repair_rate_total": custom.get("mask_repair_rate_total_mean"),
            "mask/repair_choose_project_rate": custom.get("mask_repair_choose_project_rate_mean"),
            "mask/repair_put_effort_rate": custom.get("mask_repair_put_effort_rate_mean"),
            "mask/repair_collab_rate": custom.get("mask_repair_collab_rate_mean"),
            "action/choose_project_histogram": custom.get("action_choose_project_histogram_mean"),
            "action/put_effort_histogram": custom.get("action_put_effort_histogram_mean"),
            "action/collab_popcount_histogram": custom.get("action_collab_popcount_histogram_mean"),
            # per-agent paper acceptance stats (from wrapper info dict)
            "agent0/papers_accepted": custom.get("agent0_papers_accepted_mean"),
            "agent0/papers_rejected": custom.get("agent0_papers_rejected_mean"),
            "agent0/papers_completed": custom.get("agent0_papers_completed_mean"),
            # paper_stats & effort diagnostics (from PapersMetricsCallback)
            # New API stack: metrics_logger puts values into env_runners directly
            # Old API stack: episode.custom_metrics puts values with _mean suffix
            "env/papers_total": _first_not_none(
                train_env.get("papers_total"),
                custom.get("papers_total_mean"),
            ),
            "env/papers_active_mean": _first_not_none(
                train_env.get("papers_active_mean"),
                custom.get("papers_active_mean_mean"),
            ),
            "env/papers_published_count": _first_not_none(
                train_env.get("papers_published_count"),
                custom.get("papers_published_count_mean"),
            ),
            "env/papers_rejected_count": _first_not_none(
                train_env.get("papers_rejected_count"),
                custom.get("papers_rejected_count_mean"),
            ),
            "agent/effort_applied_sum": _first_not_none(
                train_env.get("agent0_effort_applied_sum"),
                custom.get("agent0_effort_applied_sum_mean"),
            ),
            "agent/effort_invalid_frac": _first_not_none(
                train_env.get("agent0_effort_invalid_frac"),
                custom.get("agent0_effort_invalid_frac_mean"),
            ),
            "agent/choose_effective_frac": _first_not_none(
                train_env.get("agent0_choose_effective_frac"),
                custom.get("agent0_choose_effective_frac_mean"),
            ),
            "agent/active_projects_mean": _first_not_none(
                train_env.get("agent0_active_projects_mean"),
                custom.get("agent0_active_projects_mean_mean"),
            ),
        }
    )

    # --------------------
    # Horizon / truncation analysis metrics (from PapersMetricsCallback)
    # Callback writes keys like "horizon_projects_started_total" etc.
    # New stack: appear in train_env directly.
    # Old stack: appear in custom with _mean suffix.
    # --------------------
    _HORIZON_KEYS = [
        "projects_started_total",
        "projects_due_total",
        "projects_evaluated_total",
        "projects_published_total",
        "projects_rejected_total",
        "projects_due_within_episode_count",
        "due_within_episode_rate",
        "started_start_time_mean",
        "started_start_time_p95",
        "started_start_time_max",
        "started_time_window_mean",
        "started_time_window_p95",
        "started_time_window_max",
        "projects_open_end",
        "open_time_to_deadline_mean",
        "open_time_to_deadline_min",
        "open_time_to_deadline_p95",
        "open_time_to_deadline_max",
        "n_time_window_clipped",
        "clipped_rate",
        "true_time_window_over_200_max",
    ]
    for hk in _HORIZON_KEYS:
        cb_key = f"horizon_{hk}"
        val = _first_not_none(
            train_env.get(cb_key),
            custom.get(f"{cb_key}_mean"),
        )
        if val is not None:
            metrics[f"horizon/{hk}"] = val

    # --------------------
    # Timers & perf
    # --------------------
    timers = result.get("timers", {}) or {}
    metrics.update(
        {
            "timer/env_runner_sampling": timers.get("env_runner_sampling_timer"),
            "timer/learner_update": timers.get("learner_update_timer"),
            "timer/synch_weights": timers.get("synch_weights"),
            "timer/training_iteration": timers.get("training_iteration"),
            "timer/training_step": timers.get("training_step"),
        }
    )

    perf = result.get("perf", {}) or {}
    metrics.update(
        {
            "perf/cpu_util_percent": perf.get("cpu_util_percent"),
            "perf/ram_util_percent": perf.get("ram_util_percent"),
            "perf/time_this_iter_s": perf.get("time_this_iter_s"),
            "perf/time_total_s": perf.get("time_total_s"),
        }
    )

    # Throughput / env steps per sec
    env_steps_per_sec = result.get("env_steps_per_sec")
    if env_steps_per_sec is None:
        env_steps_per_sec = result.get("sampler_results", {}).get("env_steps_per_sec")

    metrics["perf/env_steps_per_sec"] = env_steps_per_sec

    # Detailed train/eval sample times, if present
    info = result.get("info", {}) or {}
    learner_info = info.get("learner", {}).get("default_policy", {}) or {}
    sampler_info = info.get("sampler_results", {}) or {}

    metrics.update(
        {
            "perf/train_time_ms": learner_info.get("train_time_ms"),
            "perf/sample_time_ms": sampler_info.get("sample_time_ms"),
        }
    )

    # --------------------
    # Global step counter (stable)
    # --------------------
    timesteps_total = result.get("timesteps_total")
    timesteps_this_iter = result.get("timesteps_this_iter")

    # Prefer timesteps_total if it's a valid scalar
    global_step: int
    if isinstance(timesteps_total, (int, float)):
        global_step = int(timesteps_total)
    else:
        # Try lifetime env steps from env_runners
        if isinstance(num_env_steps_sampled_lifetime, (int, float)):
            global_step = int(num_env_steps_sampled_lifetime)
        else:
            inc = int(timesteps_this_iter or 0)
            global_step = int(prev_total_env_steps) + inc

    metrics["iteration"] = iteration
    metrics["env/total_env_steps"] = global_step
    metrics["raw/timesteps_total"] = timesteps_total
    metrics["raw/timesteps_this_iter"] = timesteps_this_iter

    return metrics, global_step


# Match run_policy_simulation.py style
POLICY_CONFIGS: Dict[str, Dict[str, float]] = {
    "All Careerist": {"careerist": 1.0, "orthodox_scientist": 0.0, "mass_producer": 0.0},
    "All Orthodox": {"careerist": 0.0, "orthodox_scientist": 1.0, "mass_producer": 0.0},
    "All Mass Producer": {"careerist": 0.0, "orthodox_scientist": 0.0, "mass_producer": 1.0},
    "Balanced": {"careerist": 1 / 3, "orthodox_scientist": 1 / 3, "mass_producer": 1 / 3},
    "Careerist Heavy": {"careerist": 0.5, "orthodox_scientist": 0.5, "mass_producer": 0.0},
    "Orthodox Heavy": {"careerist": 0.5, "orthodox_scientist": 0.0, "mass_producer": 0.5},
    "Mass Producer Heavy": {"careerist": 0.5, "orthodox_scientist": 0.0, "mass_producer": 0.5},
}


def _set_rollout_workers_compat(config, num_workers: int):
    """
    RLlib 2.x naming drift:
      - older 2.x: config.rollouts(num_rollout_workers=...)
      - newer 2.x: config.env_runners(num_env_runners=...)
    """
    if hasattr(config, "env_runners"):
        return config.env_runners(num_env_runners=num_workers)
    return config.rollouts(num_rollout_workers=num_workers)


def make_env_creator(
    *,
    # Env params (similar to run_policy_simulation.py)
    n_agents: int,
    start_agents: int,
    max_steps: int,
    max_rewardless_steps: int,
    n_groups: int,
    max_peer_group_size: int,
    n_projects_per_step: int,
    max_projects_per_agent: int,
    max_agent_age: int,
    acceptance_threshold: float,
    reward_function: str,
    seed: int,
    # Fixed-policy population params
    policy_distribution: Dict[str, float],
    group_policy_homogenous: bool,
    prestige_threshold: float,
    novelty_threshold: float,
    effort_threshold: int,
    # Which agent PPO controls
    controlled_agent_id: str,
) -> Callable[[Optional[Dict[str, Any]]], Any]:
    """
    Returns an RLlib-compatible env creator: f(env_config) -> gymnasium.Env
    We keep env_config support so RLlib can recreate envs if needed.
    """

    # Pre-build policy functions (same as run_policy_simulation.py)
    careerist_fn = get_policy_function("careerist")
    orthodox_fn = get_policy_function("orthodox_scientist")
    mass_prod_fn = get_policy_function("mass_producer")

    def _policy_from_name(policy_name: Optional[str]):
        # Returns a callable(nested_obs)->action_dict
        if policy_name is None:
            def _do_nothing(nested_obs):
                obs = nested_obs["observation"]
                mask = nested_obs["action_mask"]
                return do_nothing_policy(obs, mask)
            return _do_nothing

        if policy_name == "careerist":
            def _fn(nested_obs):
                obs = nested_obs["observation"]
                mask = nested_obs["action_mask"]
                return careerist_fn(obs, mask, prestige_threshold)
            return _fn

        if policy_name == "orthodox_scientist":
            def _fn(nested_obs):
                obs = nested_obs["observation"]
                mask = nested_obs["action_mask"]
                return orthodox_fn(obs, mask, novelty_threshold)
            return _fn

        if policy_name == "mass_producer":
            def _fn(nested_obs):
                obs = nested_obs["observation"]
                mask = nested_obs["action_mask"]
                return mass_prod_fn(obs, mask, effort_threshold)
            return _fn

        # Fallback
        def _fallback(nested_obs):
            obs = nested_obs["observation"]
            mask = nested_obs["action_mask"]
            return do_nothing_policy(obs, mask)
        return _fallback

    def _env_creator(env_config: Optional[Dict[str, Any]] = None):
        env_config = env_config or {}

        # 1) Build env (same knobs as your simulation script)
        env = PeerGroupEnvironment(
            start_agents=env_config.get("start_agents", start_agents),
            max_agents=env_config.get("n_agents", n_agents),
            max_steps=env_config.get("max_steps", max_steps),
            n_groups=env_config.get("n_groups", n_groups),
            max_peer_group_size=env_config.get("max_peer_group_size", max_peer_group_size),
            n_projects_per_step=env_config.get("n_projects_per_step", n_projects_per_step),
            max_projects_per_agent=env_config.get("max_projects_per_agent", max_projects_per_agent),
            max_agent_age=env_config.get("max_agent_age", max_agent_age),
            max_rewardless_steps=env_config.get("max_rewardless_steps", max_rewardless_steps),
            acceptance_threshold=env_config.get("acceptance_threshold", acceptance_threshold),
            reward_mode=env_config.get("reward_function", reward_function),
            render_mode=None,
        )

        # 2) Create fixed-policy assignments (same logic as run_policy_simulation.py)
        if group_policy_homogenous:
            agent_policy_names = create_per_group_policy_population(
                n_agents, policy_distribution
            )
        else:
            agent_policy_names = create_mixed_policy_population(
                n_agents, policy_distribution, seed=seed
            )

        # 3) Build other_policies mapping: agent_id -> callable(nested_obs)->action_dict
        other_policies: Dict[str, Callable[[Any], Any]] = {}
        for agent_id in env.possible_agents:
            if agent_id == controlled_agent_id:
                continue
            idx = env.agent_to_id[agent_id]
            pol_name = agent_policy_names[idx]
            other_policies[agent_id] = _policy_from_name(pol_name)

        # 4) Wrap to single-agent env for PPO
        # Force horizon -> ensures RLlib gets completed episodes & metrics
        wrapper = RLLibSingleAgentWrapper(
            env,
            controlled_agent=controlled_agent_id,
            other_policies=other_policies,
            force_episode_horizon=max_steps,
            topk_collab=3,
            topk_apply_to_all_agents=True,
        )

        return wrapper

    return _env_creator


def main(
    *,
    iterations: int,
    framework: str,
    policy_config_name: str,
    group_policy_homogenous: bool,
    seed: int,
    # Env knobs
    n_agents: int,
    start_agents: int,
    max_steps: int,
    max_rewardless_steps: int,
    n_groups: int,
    max_peer_group_size: int,
    n_projects_per_step: int,
    max_projects_per_agent: int,
    max_agent_age: int,
    acceptance_threshold: float,
    reward_function: str,
    # Threshold knobs for heuristics
    prestige_threshold: float,
    novelty_threshold: float,
    effort_threshold: int,
    # Controlled agent
    controlled_agent_id: str,
    # Wandb options
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
    wandb_mode: str = "online",
    # Debug/action info options
    info_action: bool = False,
    info_intervall: int = 50,
    train_batch_size: int = 32000,
    # Checkpoint options
    save_every_n_iters: int = 50,
):

    # Seed all RNGs (random, numpy, torch, cuda) for reproducibility
    seed_everything(seed, workers=True)

    # RL Reproducibility: force deterministic PyTorch operations where possible.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True   # deterministic cuDNN convolutions
    torch.backends.cudnn.benchmark = False       # disable auto-tuner (non-deterministic)

    if framework not in {"torch", "tf2"}:
        raise ValueError('framework must be "torch" or "tf2"')

    if policy_config_name not in POLICY_CONFIGS:
        raise ValueError(f"Unknown policy config '{policy_config_name}'. Options: {list(POLICY_CONFIGS.keys())}")

    policy_distribution = POLICY_CONFIGS[policy_config_name]

    # IMPORTANT: macro-action encoding explodes with large max_peer_group_size.
    if max_peer_group_size > 16:
        raise ValueError(
            f"max_peer_group_size={max_peer_group_size} is too large for the current "
            "macro-action wrapper approach. Use <= 12-ish for PPO training, or implement "
            "a multi-head action model."
        )

    # 1) Ray init (robust, begrenzte Ressourcen)
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        logging_level="WARNING",
        _node_ip_address="127.0.0.1",
        num_cpus=os.cpu_count() or 4,
        _system_config={
            # verlängert die Wartezeit für raylet-Startup/GCS
            "raylet_start_wait_time_s": 60.0,
        },
    )

    # 1b) Optional Weights & Biases init (driver-only)
    use_wandb = False
    wandb_run = None
    try:
        if wandb_mode != "disabled":
            # Build a flat config dict for wandb
            wandb_config: Dict[str, Any] = {
                "algo": "PPO",
                "framework": framework,
                "iterations": iterations,
                "seed": seed,
                "policy_config_name": policy_config_name,
                "policy_distribution": policy_distribution,
                "controlled_agent_id": controlled_agent_id,
                "reward_function": reward_function,
                "acceptance_threshold": acceptance_threshold,
                "env": {
                    "n_agents": n_agents,
                    "start_agents": start_agents,
                    "max_steps": max_steps,
                    "max_rewardless_steps": max_rewardless_steps,
                    "n_groups": n_groups,
                    "max_peer_group_size": max_peer_group_size,
                    "n_projects_per_step": n_projects_per_step,
                    "max_projects_per_agent": max_projects_per_agent,
                    "max_agent_age": max_agent_age,
                },
                "heuristics": {
                    "prestige_threshold": prestige_threshold,
                    "novelty_threshold": novelty_threshold,
                    "effort_threshold": effort_threshold,
                },
            }

            project = wandb_project or "game-of-science-ppo"
            run_name = (
                f"ppo_{policy_config_name}_{reward_function}_s{seed}_n{n_agents}"
            )

            init_kwargs: Dict[str, Any] = {
                "project": project,
                "config": wandb_config,
                "name": run_name,
                "mode": wandb_mode,
            }
            if wandb_entity:
                init_kwargs["entity"] = wandb_entity
            if wandb_group:
                init_kwargs["group"] = wandb_group

            wandb_run = wandb.init(**init_kwargs)
            use_wandb = wandb_run is not None
            if use_wandb:
                print(f"[wandb] enabled: project={project}, entity={wandb_entity}, run={run_name}")
            else:
                print("[wandb] init returned None; WandB disabled.")
        else:
            print("[wandb] mode='disabled'; WandB logging turned off.")
    except Exception as e:
        # Fall back gracefully if wandb init fails
        print(f"wandb init failed or disabled: {e}")
        use_wandb = False
        wandb_run = None

    # 2) Register env
    env_name = "peer_group_single_agent_fixed_population"
    env_creator = make_env_creator(
        n_agents=n_agents,
        start_agents=start_agents,
        max_steps=max_steps,
        max_rewardless_steps=max_rewardless_steps,
        n_groups=n_groups,
        max_peer_group_size=max_peer_group_size,
        n_projects_per_step=n_projects_per_step,
        max_projects_per_agent=max_projects_per_agent,
        max_agent_age=max_agent_age,
        acceptance_threshold=acceptance_threshold,
        reward_function=reward_function,
        seed=seed,
        policy_distribution=policy_distribution,
        group_policy_homogenous=group_policy_homogenous,
        prestige_threshold=prestige_threshold,
        novelty_threshold=novelty_threshold,
        effort_threshold=effort_threshold,
        controlled_agent_id=controlled_agent_id,
    )
    tune.register_env(env_name, env_creator)

    # 3) Build PPO config
    config = (
        PPOConfig()
        # RL Reproducibility: seed= makes RLlib offset per-worker seed deterministically
        .debugging(seed=seed, log_level="WARN")
        .environment(
            env=env_name,
            env_config={
                "n_agents": n_agents,
                "start_agents": start_agents,
                "max_steps": max_steps,
                "max_rewardless_steps": max_rewardless_steps,
                "n_groups": n_groups,
                "max_peer_group_size": max_peer_group_size,
                "n_projects_per_step": n_projects_per_step,
                "max_projects_per_agent": max_projects_per_agent,
                "max_agent_age": max_agent_age,
                "acceptance_threshold": acceptance_threshold,
                "reward_function": reward_function,
            },
        )
        .framework(framework)
        # Keep small for debugging; increase later
        .training(
            train_batch_size=train_batch_size, #32000
            gamma=0.99,
            lambda_=0.95,
            minibatch_size=min(512, train_batch_size),  # can't exceed train_batch_size
            num_epochs=8,
            lr=1e-4,
            grad_clip=0.5,
            entropy_coeff=0.01,
            vf_loss_coeff=1.0,
            vf_clip_param=5.0,
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "tanh",
            },
        )
    )

    config = config.env_runners(
        rollout_fragment_length="auto",
        # optional, falls verfügbar:
        sample_timeout_s=300,
    )

    # ------------------------------------------------------------------
    # Compose callbacks: PapersMetricsCallback (always) + ActionInfoCallback (optional)
    # ------------------------------------------------------------------
    callback_classes = [PapersMetricsCallback]

    if info_action:
        ActionInfoCallback = make_action_info_callback(
            controlled_agent_id=controlled_agent_id,
            info_interval=info_intervall,
            n_projects_per_step=n_projects_per_step,
            max_projects_per_agent=max_projects_per_agent,
            max_peer_group_size=max_peer_group_size,
        )
        callback_classes.append(ActionInfoCallback)

    if len(callback_classes) == 1:
        config = config.callbacks(callback_classes[0])
    else:
        from ray.rllib.algorithms.callbacks import make_multi_callbacks
        config = config.callbacks(make_multi_callbacks(callback_classes))

    # Enable evaluation runners and periodic evaluation
    config = config.evaluation(
        # Run one evaluation round every iteration.
        evaluation_interval=5,

        # Create extra evaluation EnvRunners in the evaluation group.
        evaluation_num_env_runners=2,

        # Run evaluation for a fixed number of episodes (total across all eval runners).
        evaluation_duration_unit="episodes",
        evaluation_duration=5,
        evaluation_config={"explore": False},
    )

    if info_action:
        config = _set_rollout_workers_compat(config, num_workers=1)
        config = config.evaluation(evaluation_num_env_runners=0)
    else:
        config = _set_rollout_workers_compat(config, num_workers=10)

    # Rebuild the PPO algorithm with evaluation enabled
    ppo_with_evaluation = config.build_algo()

    history = []

    best_eval = float("-inf")
    best_eval_iter: int = -1
    best_train_return = float("-inf")
    best_checkpoint_path: Optional[str] = None
    last_checkpoint_path: Optional[str] = None

    prev_total_env_steps = 0

    try:
        print("\n=== PPO TRAINING (single controlled agent) with evaluation ===")
        print(f"controlled_agent_id: {controlled_agent_id}")
        print(f"policy_config: {policy_config_name}  (group_policy_homogenous={group_policy_homogenous})")
        print(f"env: n_agents={n_agents}, start_agents={start_agents}, max_steps={max_steps}, "
              f"n_groups={n_groups}, max_peer_group_size={max_peer_group_size}")
        print(f"thresholds: prestige={prestige_threshold}, novelty={novelty_threshold}, effort={effort_threshold}")
        print(f"reward_function: {reward_function}, acceptance_threshold: {acceptance_threshold}\n")

        for i in range(iterations):
            result = ppo_with_evaluation.train()

            # Metriken extrahieren (uses env_runners-/evaluation-block out of results)
            metrics, global_step = extract_metrics(result, i, prev_total_env_steps)
            prev_total_env_steps = global_step

            # Optionale, detaillierte Debug-Ausgaben zu den Raw-Result-Strukturen
            train_env = result.get("env_runners") or {}
            eval_env = (result.get("evaluation") or {}).get("env_runners") or {}

            # Safe extraction of eval and train returns for Logging/Checkpointing
            raw_eval_return = metrics.get("eval/episode_return_mean")
            raw_train_return = metrics.get("train/episode_return_mean")
            eval_return_val = _safe_float(raw_eval_return)
            train_return_val = _safe_float(raw_train_return)

            has_valid_eval = not math.isnan(eval_return_val)
            has_valid_train = not math.isnan(train_return_val)

            status = ""
            _kl = _safe_float(metrics.get("ppo/mean_kl"))
            _vf = _safe_float(metrics.get("ppo/vf_explained_var"))
            if not math.isnan(_kl) and _kl > 0.05:
                status += "⚠ KL high "
            if not math.isnan(_vf) and _vf < 0:
                status += "⚠ critic bad "
            if not has_valid_eval:
                status += "(no_eval) "

            # Human-readable Returns for logging (handles None/NaN)
            eval_return_str = "n/a" if not has_valid_eval else f"{eval_return_val:8.2f}"
            train_return_str = "n/a" if not has_valid_train else f"{train_return_val:8.2f}"

            print(
                f"Iter {i:03d} | "
                f"TrainReturn: {train_return_str} | "
                f"EvalReturn: {eval_return_str} | "
                f"KL: {_kl:6.4f} | "
                f"VF_var: {_vf:7.4f} "
                f"{status}"
            )

            history_entry: Dict[str, Any] = {
                "iter": i,
                "eval_return": eval_return_val if has_valid_eval else float("nan"),
                "kl": metrics["ppo/mean_kl"],
                "vf_var": metrics["ppo/vf_explained_var"],
                "episode_reward_mean": raw_train_return,
                "episode_len_mean": metrics.get("train/episode_len_mean"),
                "timesteps_total": result.get("timesteps_total"),
            }
            history.append(history_entry)

            # ------------------------------------------------------------------
            # Track best train return (informational only, NOT used for checkpointing)
            # ------------------------------------------------------------------
            if has_valid_train and train_return_val > best_train_return:
                best_train_return = train_return_val

            # ------------------------------------------------------------------
            # Best-eval checkpoint: save only when a valid eval improves on the best
            # ------------------------------------------------------------------
            if has_valid_eval and eval_return_val > best_eval:
                best_eval = eval_return_val
                best_eval_iter = i
                try:
                    chkpt_dir = build_checkpoint_path(
                        policy_config_name=policy_config_name,
                        reward_function=reward_function,
                        iteration=i,
                        max_rewardless_steps=max_rewardless_steps,
                        eval_return=eval_return_val,
                        tag="best",
                    )
                    os.makedirs(chkpt_dir, exist_ok=True)
                    ppo_with_evaluation.save_checkpoint(chkpt_dir)
                    best_checkpoint_path = chkpt_dir
                    print(
                        f"[checkpoint] new best eval return {eval_return_val:.4f} "
                        f"at iter {i} -> {best_checkpoint_path}"
                    )
                except Exception as e:
                    print(f"[checkpoint] ERROR: could not save best checkpoint at iter {i}: {e}")

                # WandB artifact for new best model
                if use_wandb and best_checkpoint_path is not None:
                    try:
                        artifact = wandb.Artifact(
                            name=f"ppo-best-{policy_config_name}-s{seed}",
                            type="model",
                        )
                        if os.path.isdir(best_checkpoint_path):
                            artifact.add_dir(best_checkpoint_path)
                        elif os.path.isfile(best_checkpoint_path):
                            artifact.add_file(best_checkpoint_path)
                        else:
                            print(f"[checkpoint] WARNING: best checkpoint path not found: {best_checkpoint_path}")
                        wandb.log_artifact(artifact)
                    except Exception as e:
                        print(f"[checkpoint] WARNING: wandb artifact logging failed at iter {i}: {e}")
            elif not has_valid_eval:
                # No evaluation was run this iteration (expected for non-eval iterations)
                if (i + 1) % 5 == 0:
                    # Only warn when evaluation *should* have happened
                    print(f"[checkpoint] no valid eval at iter {i} (eval expected but missing/NaN)")
            # else: valid eval, but not a new best – nothing to do

            # ------------------------------------------------------------------
            # Periodic checkpoint: save every N iterations regardless of eval
            # ------------------------------------------------------------------
            if save_every_n_iters > 0 and (i + 1) % save_every_n_iters == 0:
                try:
                    chkpt_dir = build_checkpoint_path(
                        policy_config_name=policy_config_name,
                        reward_function=reward_function,
                        iteration=i,
                        max_rewardless_steps=max_rewardless_steps,
                        eval_return=eval_return_val if has_valid_eval else None,
                        tag="periodic",
                    )
                    os.makedirs(chkpt_dir, exist_ok=True)
                    ppo_with_evaluation.save_checkpoint(chkpt_dir)
                    last_checkpoint_path = chkpt_dir
                    print(
                        f"[checkpoint] periodic save at iter {i} -> {last_checkpoint_path}"
                    )
                except Exception as e:
                    print(f"[checkpoint] ERROR: periodic save failed at iter {i}: {e}")

            # ------------------------------------------------------------------
            # WandB logging (driver-only)
            # ------------------------------------------------------------------
            if use_wandb:
                try:
                    # Stable step: prefer timesteps_total/env steps, else fallback to iteration index
                    raw_step = result.get("timesteps_total")
                    if not isinstance(raw_step, (int, float)) or raw_step is None:
                        raw_step = metrics.get("env/total_env_steps")
                    if not isinstance(raw_step, (int, float)) or raw_step is None:
                        raw_step = global_step
                    if not isinstance(raw_step, (int, float)) or raw_step is None:
                        raw_step = i
                    step = int(raw_step)

                    log_dict = wandb_sanitize(metrics)
                    # Mark first iteration as connectivity check
                    if i == 0:
                        log_dict["debug/it_works"] = 1

                    wandb.log(log_dict, step=step)
                except Exception as e:
                    print(f"wandb.log failed at iter {i}: {e}")

    finally:
        # ------------------------------------------------------------------
        # Training summary
        # ------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        if best_eval_iter >= 0:
            print(f"  Best eval return:      {best_eval:.4f}")
            print(f"  Best eval iteration:   {best_eval_iter}")
            print(f"  Best checkpoint path:  {best_checkpoint_path}")
        else:
            print("  No valid evaluation checkpoint was saved.")
        if not math.isinf(best_train_return) and best_train_return > float("-inf"):
            print(f"  Best train return:     {best_train_return:.4f} (informational only)")
        if last_checkpoint_path:
            print(f"  Last periodic checkpoint: {last_checkpoint_path}")
        print("=" * 60 + "\n")

        ppo_with_evaluation.stop()
        ray.shutdown()

        # Upload best checkpoint as artifact if available (final, deduplicated)
        if use_wandb and best_checkpoint_path is not None:
            try:
                artifact = wandb.Artifact(
                    name=f"ppo-best-{policy_config_name}-s{seed}",
                    type="model",
                )
                if os.path.isdir(best_checkpoint_path):
                    artifact.add_dir(best_checkpoint_path)
                elif os.path.isfile(best_checkpoint_path):
                    artifact.add_file(best_checkpoint_path)
                wandb.log_artifact(artifact)
            except Exception as e:
                print(f"[checkpoint] WARNING: final wandb artifact upload failed: {e}")

        if use_wandb and wandb_run is not None:
            try:
                wandb.finish()
            except Exception as e:
                print(f"wandb.finish() failed: {e}")

    # Persist training history to CSV and plot (if matplotlib is available)
    results_paths = plot_training_history(history, save_prefix=f"ppo_history_{policy_config_name}_{seed}")

    # Optionally log CSV/PNG as artifacts/files in wandb
    if use_wandb and results_paths:
        try:
            csv_path = results_paths.get("csv")
            png_path = results_paths.get("png")
            if csv_path:
                wandb.save(csv_path)
            if png_path:
                wandb.save(png_path)
        except Exception as e:
            print(f"wandb save of history artifacts failed: {e}")


def plot_training_history(history: list, out_dir: str = "results", save_prefix: str = "ppo_history"):
    """Save training history to CSV and (if available) plot eval_return, kl and vf_var.

    Expects history to be a list of dicts with keys: 'iter', 'eval_return', 'kl', 'vf_var'.
    Creates `out_dir/{save_prefix}.csv` and `out_dir/{save_prefix}.png` (if matplotlib present).
    Returns paths written.
    """
    os.makedirs(out_dir, exist_ok=True)

    iters = [h.get("iter", idx) for idx, h in enumerate(history)]
    evals = [float(h.get("eval_return", float("nan"))) for h in history]
    kls = [float(h.get("kl", float("nan"))) for h in history]
    vfs = [float(h.get("vf_var", float("nan"))) for h in history]

    csv_path = os.path.join(out_dir, f"{save_prefix}.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["iter", "eval_return", "kl", "vf_var"])
        for i, e, k, v in zip(iters, evals, kls, vfs):
            writer.writerow([i, e, k, v])
    print(f"Wrote training CSV: {csv_path}")

    png_path = None

    # Create a single figure with eval_return on left axis and kl/vf_var on right axis
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(iters, evals, label="eval_return", color="tab:blue", marker="o")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("eval_return", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(iters, kls, label="kl", color="tab:orange", linestyle="--", marker="x")
    ax2.plot(iters, vfs, label="vf_var", color="tab:green", linestyle=":", marker="s")
    ax2.set_ylabel("kl / vf_var")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title(save_prefix)
    fig.tight_layout()

    png_path = os.path.join(out_dir, f"{save_prefix}.png")
    fig.savefig(png_path)
    plt.close(fig)
    print(f"Wrote training plot: {png_path}")

    return {"csv": csv_path, "png": png_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # RLlib
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--framework",
        type=str,
        default="torch",
        choices=["torch", "tf2"],
        help='Deep learning backend: "torch" or "tf2"',
    )

    # Match run_policy_simulation style
    parser.add_argument(
        "--policy-config",
        type=str,
        default="Balanced",
        choices=list(POLICY_CONFIGS.keys()),
        help="Which fixed-policy mixture to use for the non-controlled agents",
    )
    parser.add_argument(
        "--group-policy-homogenous",
        action="store_true",
        help="If set, assigns the same archetype per group (like create_per_group_policy_population). "
             "If not set, mixes per agent (like create_mixed_policy_population).",
    )
    parser.add_argument("--seed", type=int, default=42)

    # Env knobs (keep small for PPO + macro-action)
    parser.add_argument("--n-agents", type=int, default=64)
    parser.add_argument("--start-agents", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-rewardless-steps", type=int, default=500)
    parser.add_argument("--n-groups", type=int, default=8)
    parser.add_argument("--max-peer-group-size", type=int, default=8)  # keep small!
    parser.add_argument("--n-projects-per-step", type=int, default=1)
    parser.add_argument("--max-projects-per-agent", type=int, default=6)
    parser.add_argument("--max-agent-age", type=int, default=750)

    # Reward knobs
    parser.add_argument("--acceptance-threshold", type=float, default=0.5)
    parser.add_argument("--reward-function", type=str, default="by_effort", choices=["multiply", "evenly", "by_effort"])

    # Heuristic thresholds (same as your simulation script)
    parser.add_argument("--prestige-threshold", type=float, default=0.2)
    parser.add_argument("--novelty-threshold", type=float, default=0.8)
    parser.add_argument("--effort-threshold", type=int, default=22)

    # Controlled agent
    parser.add_argument("--controlled-agent-id", type=str, default="agent_0")

    # Wandb options
    parser.add_argument("--wandb-project", type=str, default="RL in the Game of Science")
    parser.add_argument("--wandb-entity", type=str, default="rl_in_the_game_of_science")
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])

    # Action debug / info options
    parser.add_argument(
        "--info-action",
        action="store_true",
        default=True,
        help="Periodically prints the chosen action of the controlled agent (enabled by default).",
    )
    parser.add_argument(
        "--no-info-action",
        dest="info_action",
        action="store_false",
        help="Disable periodic action printing.",
    )
    parser.add_argument(
        "--info-intervall",
        type=int,
        default=50,
        help="Print the controlled agent's action every N env steps within an episode.",
    )

    parser.add_argument("--train-batch-size", type=int, default=32000,
                        help="Number of env steps collected per training iteration.")

    parser.add_argument("--save-every-n-iters", type=int, default=50,
                        help="Save a periodic checkpoint every N iterations (0 = disabled).")

    args = parser.parse_args()

    main(
        iterations=args.iterations,
        framework=args.framework,
        policy_config_name=args.policy_config,
        group_policy_homogenous=args.group_policy_homogenous,
        seed=args.seed,
        n_agents=args.n_agents,
        start_agents=args.start_agents,
        max_steps=args.max_steps,
        max_rewardless_steps=args.max_rewardless_steps,
        n_groups=args.n_groups,
        max_peer_group_size=args.max_peer_group_size,
        n_projects_per_step=args.n_projects_per_step,
        max_projects_per_agent=args.max_projects_per_agent,
        max_agent_age=args.max_agent_age,
        acceptance_threshold=args.acceptance_threshold,
        reward_function=args.reward_function,
        prestige_threshold=args.prestige_threshold,
        novelty_threshold=args.novelty_threshold,
        effort_threshold=args.effort_threshold,
        controlled_agent_id=args.controlled_agent_id,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_mode=args.wandb_mode,
        info_action=args.info_action,
        info_intervall=args.info_intervall,
        train_batch_size=args.train_batch_size,
        save_every_n_iters=args.save_every_n_iters,
    )
