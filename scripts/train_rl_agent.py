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
import sys

# Add the project root to sys.path so we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import json
import wandb
import ray
import time
import traceback
import gc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

from ray import tune
from typing import Any, Callable, Dict, Optional
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.appo import APPOConfig
from lightning.pytorch import seed_everything

from src.agent_policies import (
    create_mixed_policy_population,
    create_per_group_policy_population,
    do_nothing_policy,
    get_policy_function,
)

from src.env.peer_group_environment import PeerGroupEnvironment
from src.rllib_single_agent_wrapper import RLLibSingleAgentWrapper
from src.callbacks.debug_actions_callback import make_action_info_callback
from src.callbacks.papers_metrics_callback import PapersMetricsCallback
from src.checkpoint_utils import build_checkpoint_path

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

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

        # Convert to float if it is a float-like type to ensure it is JSON serializable
        if isinstance(v, float):
            v = float(v)
        elif isinstance(v, int):
            v = int(v)

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
    # Learner internals (PPO, APPO, etc.)
    # --------------------
    # RLlib v2/v3 structures: 
    # - result['learners']['default_policy']
    # - result['info']['learner']['default_policy']['learner_stats']
    # - result['info']['learner']['default_policy']
    learners = result.get("learners", {}) or {}
    info_learner = result.get("info", {}).get("learner", {}) or {}
    
    # Try multiple paths to find the actual learner stats dictionary
    learner = (
        learners.get("default_policy") 
        or learners.get("default_learner")
        or info_learner.get("default_policy", {}).get("learner_stats")
        or info_learner.get("default_policy")
        or {}
    )
    
    # If it's a list, take the last/mean
    if isinstance(learner, list) and len(learner) > 0:
        learner = learner[-1]
    
    mean_kl = _first_not_none(learner.get("mean_kl_loss"), learner.get("kl"))
    entropy = _first_not_none(learner.get("entropy"), learner.get("mean_entropy"))
    policy_loss = _first_not_none(learner.get("policy_loss"), learner.get("mean_policy_loss"))
    value_loss = _first_not_none(learner.get("vf_loss"), learner.get("vf_loss_mean"), learner.get("mean_vf_loss"))
    vf_explained_var = _first_not_none(learner.get("vf_explained_var"), learner.get("explained_variance"))

    # learning rate: try modern key, fall back to older ones
    lr = _first_not_none(
        learner.get("default_optimizer_learning_rate"),
        learner.get("learning_rate"),
        learner.get("cur_lr"),
        learner.get("lr")
    )
    grad_global_norm = _first_not_none(
        learner.get("gradients_default_optimizer_global_norm"),
        learner.get("grad_gnorm"),
        learner.get("gradients_global_norm")
    )
    curr_kl_coeff = _first_not_none(learner.get("curr_kl_coeff"), learner.get("kl_coeff"))

    metrics.update(
        {
            "algo/mean_kl": mean_kl,
            "algo/entropy": entropy,
            "algo/policy_loss": policy_loss,
            "algo/value_loss": value_loss,
            "algo/vf_explained_var": vf_explained_var,
            "algo/lr": lr,
            "algo/gradients_default_optimizer_global_norm": grad_global_norm,
            "algo/curr_kl_coeff": curr_kl_coeff,
            # Legacy/compatible keys
            "ppo/mean_kl": mean_kl,
            "ppo/vf_explained_var": vf_explained_var,
            "ppo/policy_loss": policy_loss,
            "ppo/entropy": entropy,
        }
    )

    # --------------------
    # Custom env metrics from callbacks
    # --------------------
    # RLlib can put custom_metrics in multiple places depending on version/config
    custom = result.get("custom_metrics", {}) or {}
    if not custom and "env_runners" in result:
        custom = result["env_runners"].get("custom_metrics", {}) or {}

    metrics.update(
        {
            # sparse reward debug
            "env/reward_nonzero_frac": _first_not_none(custom.get("reward_nonzero_frac_mean"), custom.get("reward_nonzero_frac")),
            "env/steps_until_first_reward": _first_not_none(custom.get("steps_until_first_reward_mean"), custom.get("steps_until_first_reward")),
            "env/positive_reward_events_per_episode": _first_not_none(custom.get("positive_reward_events_per_episode_mean"), custom.get("positive_reward_events_per_episode")),
            "env/rewardless_termination_frac": _first_not_none(custom.get("rewardless_termination_flag_mean"), custom.get("rewardless_termination_flag")),
            "env/truncated_frac": _first_not_none(custom.get("truncated_flag_mean"), custom.get("truncated_flag")),
            "env/terminated_frac": _first_not_none(custom.get("terminated_flag_mean"), custom.get("terminated_flag")),
            # environment dynamics
            "env/published_projects_count": _first_not_none(custom.get("published_projects_count_mean"), custom.get("published_projects_count")),
            "env/accepted_projects_count": _first_not_none(custom.get("accepted_projects_count_mean"), custom.get("accepted_projects_count")),
            "env/rejected_projects_count": _first_not_none(custom.get("rejected_projects_count_mean"), custom.get("rejected_projects_count")),
            "env/agent_age_at_end": _first_not_none(custom.get("agent_age_at_end_mean"), custom.get("agent_age_at_end")),
            "env/agent_eliminated_flag": _first_not_none(custom.get("agent_eliminated_flag_mean"), custom.get("agent_eliminated_flag")),
            # mask repair + action diagnostics (if your wrapper/Callbacks provide them)
            "mask/repair_rate_total": _first_not_none(custom.get("mask_repair_rate_total_mean"), custom.get("mask_repair_rate_total")),
            "mask/repair_choose_project_rate": _first_not_none(custom.get("mask_repair_choose_project_rate_mean"), custom.get("mask_repair_choose_project_rate")),
            "mask/repair_put_effort_rate": _first_not_none(custom.get("mask_repair_put_effort_rate_mean"), custom.get("mask_repair_put_effort_rate")),
            "mask/repair_collab_rate": _first_not_none(custom.get("mask_repair_collab_rate_mean"), custom.get("mask_repair_collab_rate")),
            "action/choose_project_histogram": _first_not_none(custom.get("action_choose_project_histogram_mean"), custom.get("action_choose_project_histogram")),
            "action/put_effort_histogram": _first_not_none(custom.get("action_put_effort_histogram_mean"), custom.get("action_put_effort_histogram")),
            "action/collab_popcount_histogram": _first_not_none(custom.get("action_collab_popcount_histogram_mean"), custom.get("action_collab_popcount_histogram")),
            # per-agent paper acceptance stats (from wrapper info dict)
            "agent0/papers_accepted": _first_not_none(custom.get("agent0_papers_accepted_mean"), custom.get("agent0_papers_accepted")),
            "agent0/papers_rejected": _first_not_none(custom.get("agent0_papers_rejected_mean"), custom.get("agent0_papers_rejected")),
            "agent0/papers_completed": _first_not_none(custom.get("agent0_papers_completed_mean"), custom.get("agent0_papers_completed")),
            # paper_stats & effort diagnostics (from PapersMetricsCallback)
            # New API stack: metrics_logger puts values into env_runners directly
            # Old API stack: episode.custom_metrics puts values with _mean suffix
            "env/papers_total": _first_not_none(
                train_env.get("papers_total"),
                custom.get("papers_total_mean"),
                custom.get("papers_total"),
            ),
            "env/papers_active_mean": _first_not_none(
                train_env.get("papers_active_mean"),
                custom.get("papers_active_mean_mean"),
                custom.get("papers_active_mean"),
            ),
            "env/papers_published_count": _first_not_none(
                train_env.get("papers_published_count"),
                custom.get("papers_published_count_mean"),
                custom.get("papers_published_count"),
            ),
            "env/papers_rejected_count": _first_not_none(
                train_env.get("papers_rejected_count"),
                custom.get("papers_rejected_count_mean"),
                custom.get("papers_rejected_count"),
            ),
            "agent/effort_applied_sum": _first_not_none(
                train_env.get("agent0_effort_applied_sum"),
                custom.get("agent0_effort_applied_sum_mean"),
                custom.get("agent0_effort_applied_sum"),
            ),
            "agent/effort_invalid_frac": _first_not_none(
                train_env.get("agent0_effort_invalid_frac"),
                custom.get("agent0_effort_invalid_frac_mean"),
                custom.get("agent0_effort_invalid_frac"),
            ),
            "agent/choose_effective_frac": _first_not_none(
                train_env.get("agent0_choose_effective_frac"),
                custom.get("agent0_choose_effective_frac_mean"),
                custom.get("agent0_choose_effective_frac"),
            ),
            "agent/active_projects_mean": _first_not_none(
                train_env.get("agent0_active_projects_mean"),
                custom.get("agent0_active_projects_mean_mean"),
                custom.get("agent0_active_projects_mean"),
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
            custom.get(cb_key),
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
    env_steps_per_sec = (
        result.get("num_env_steps_sampled_throughput_per_sec")
        or result.get("env_steps_per_sec")
        or result.get("sampler_results", {}).get("env_steps_per_sec")
    )

    metrics["perf/env_steps_per_sec"] = env_steps_per_sec

    # Detailed train/eval sample times, if present
    info = result.get("info", {}) or {}
    learner_info = info.get("learner", {}).get("default_policy", {}) or {}
    sampler_info = result.get("sampler_results", {}) or info.get("sampler_results", {}) or {}
    
    # Check timers block (modern RLlib)
    timers = result.get("timers", {}) or {}

    metrics.update(
        {
            "perf/train_time_ms": _first_not_none(
                learner_info.get("train_time_ms"),
                timers.get("learn_time_ms"),
                timers.get("training_step_time_ms"),
            ),
            "perf/sample_time_ms": _first_not_none(
                sampler_info.get("sample_time_ms"),
                timers.get("sample_time_ms"),
            ),
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
    # Top-k ablation options
    topk_collab: Optional[int] = None,
    topk_apply_to_all_agents: bool = False,
    info_action: bool = False,
    info_interval: int = 50,
    debug_effort: bool = False,
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

        # 1) Build env
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
            topk_collab=env_config.get("topk_collab", topk_collab),
            topk_apply_to_all_agents=env_config.get("topk_apply_to_all_agents", topk_apply_to_all_agents),
            info_action=env_config.get("info_action", info_action),
            info_interval=env_config.get("info_interval", info_interval),
            debug_effort=env_config.get("debug_effort", debug_effort),
            is_evaluation=env_config.get("evaluation", False),
        )

        return wrapper

    return _env_creator


def main(
    *,
    algo: str = "PPO",
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
    # Top-k ablation options
    topk_collab: Optional[int] = None,
    topk_apply_to_all_agents: bool = False,
    # Wandb options
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
    wandb_mode: str = "online",
    wandb_run_id: str | None = None,
    # Debug/action info options
    info_action: bool = False,
    info_intervall: int = 50,
    debug_effort: bool = False,
    train_batch_size: int = 18000,
    # Checkpoint options
    save_every_n_iters: int = 50,
    # Profiling overrides
    num_workers: int = 6,
    evaluation_interval: Optional[int] = None,
    # RL training hyperparameters
    gamma: float = 0.99,
    lambda_: float = 0.95,
    lr: float = 1e-4,
    num_epochs: int = 8,
    entropy_coeff: float = 0.01,
    vf_loss_coeff: float = 1.0,
    grad_clip: float = 0.5,
    vf_share_layers: bool = True,
    num_envs_per_worker: int = 1,
    rollout_fragment_length: str | int = 200,
    total_env_steps: int | None = None,
    checkpoint: str | None = None,
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

    # Multi-head action encoding is used in the wrapper (MultiBinary for collaboration).
    # The exponential explosion of Discrete(2^N) is avoided.
    # We still keep a reasonable limit for observation vector size and PPO convergence.
    if max_peer_group_size > 100:
        raise ValueError(
            f"max_peer_group_size={max_peer_group_size} is very large. "
            "Even with MultiBinary, very large groups might slow down training."
        )

    # GUARD: Ensure that the peer group size in the environment does not exceed max_peer_group_size.
    # The environment distributes agents into n_groups. Average size = n_agents / n_groups.
    # If this ratio is larger than max_peer_group_size, the environment logic (reset/step) will crash.
    avg_group_size = n_agents / n_groups
    if avg_group_size > max_peer_group_size:
        raise ValueError(
            f"CONFIGURATION ERROR: n_agents({n_agents}) / n_groups({n_groups}) = {avg_group_size:.2f}, "
            f"which is greater than max_peer_group_size({max_peer_group_size}).\n"
            "This would cause a broadcast error in the environment because the peer group "
            "size exceeds the pre-allocated space for observations and action masks.\n"
            "Solution: Increase n_groups, decrease n_agents, or increase max_peer_group_size (if < 13)."
        )

    # ------------------------------------------------------------------
    # Prepare training config for persistence
    # ------------------------------------------------------------------
    train_config = {
        "algo": algo,
        "iterations": iterations,
        "framework": framework,
        "policy_config_name": policy_config_name,
        "group_policy_homogenous": group_policy_homogenous,
        "seed": seed,
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
        "prestige_threshold": prestige_threshold,
        "novelty_threshold": novelty_threshold,
        "effort_threshold": effort_threshold,
        "controlled_agent_id": controlled_agent_id,
        "topk_collab": topk_collab,
        "topk_apply_to_all_agents": topk_apply_to_all_agents,
        "train_batch_size": train_batch_size,
        "gamma": gamma,
        "lambda_": lambda_,
        "lr": lr,
        "num_epochs": num_epochs,
        "entropy_coeff": entropy_coeff,
        "vf_loss_coeff": vf_loss_coeff,
        "grad_clip": grad_clip,
        "vf_share_layers": vf_share_layers,
        "num_envs_per_worker": num_envs_per_worker,
        "rollout_fragment_length": rollout_fragment_length,
        "total_env_steps": total_env_steps,
    }

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
    os.environ["WANDB_SILENT"] = "true"
    # Unterdrückt Ray-Deprecation-Warnungen (z.B. TBXLogger)
    os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
    # Erhöht den HTTP-Timeout für GraphQL-Anfragen (Standard ist oft zu niedrig)
    os.environ["WANDB_HTTP_TIMEOUT"] = "180"
    
    # Check if we are running inside a WandB sweep
    is_sweep = wandb.run is not None and getattr(wandb.run, "sweep_id", None) is not None
    
    use_wandb = False
    wandb_run = None
    try:
        if wandb_mode != "disabled":
            # If we're in a sweep, we don't need to re-init unless specific project/entity are needed.
            # Usually, wandb.agent handles init, and wandb.run will be already populated.
            if is_sweep:
                wandb_run = wandb.run
                use_wandb = True
                project = getattr(wandb_run, "project", "unknown")
                run_name = getattr(wandb_run, "name", "unknown")
                print(f"[wandb] using existing sweep run: {run_name}")
            else:
                # Build a flat config dict for wandb
                # Since some values are defined later in the script (workers, rollout_length),
                # we use the arguments/locals to build the config.
                _wandb_num_workers = 1 if info_action else num_workers

                wandb_config: Dict[str, Any] = {
                    "algo": algo,
                    "framework": framework,
                    "iterations": iterations,
                    "seed": seed,
                    "policy_config_name": policy_config_name,
                    "policy_distribution": policy_distribution,
                    "controlled_agent_id": controlled_agent_id,
                    "reward_function": reward_function,
                    "acceptance_threshold": acceptance_threshold,
                    "train_batch_size": train_batch_size,
                    "num_workers": _wandb_num_workers,
                    "num_envs_per_worker": num_envs_per_worker,
                    "rollout_fragment_length": rollout_fragment_length,
                    "total_env_steps": total_env_steps,
                    "training": {
                        "gamma": gamma,
                        "lambda": lambda_,
                        "lr": lr,
                        "num_epochs": num_epochs,
                        "entropy_coeff": entropy_coeff,
                        "vf_loss_coeff": vf_loss_coeff,
                        "grad_clip": grad_clip,
                        "vf_share_layers": vf_share_layers,
                    },
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
                        "topk_collab": topk_collab,
                        "topk_apply_to_all_agents": topk_apply_to_all_agents,
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
                    "id": wandb_run_id,
                    "resume": "allow" if wandb_run_id else None,
                    "settings": wandb.Settings(silent=True, console="off", _disable_stats=True)
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
        topk_collab=topk_collab,
        topk_apply_to_all_agents=topk_apply_to_all_agents,
        info_action=info_action,
        info_interval=info_intervall,
        debug_effort=debug_effort,
    )
    tune.register_env(env_name, env_creator)

    # 3) Build config
    if algo.upper() == "PPO":
        config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            # RL Reproducibility: seed= macht RLlib's per-worker seed deterministisch
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
                    "info_action": info_action,
                    "info_interval": info_intervall,
                    "debug_effort": debug_effort,
                    "evaluation": False,
                },
            )
            .framework(framework)
            .training(
                train_batch_size=train_batch_size,
                gamma=gamma,
                lambda_=lambda_,
                minibatch_size=min(512, train_batch_size),
                num_epochs=num_epochs,
                lr=lr,
                grad_clip=grad_clip,
                entropy_coeff=entropy_coeff,
                vf_loss_coeff=vf_loss_coeff,
                vf_clip_param=5.0,
                model={
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "tanh",
                    "vf_share_layers": vf_share_layers,
                },
            )
        )
    elif algo.upper() == "APPO":
        config = (
            APPOConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
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
                    "info_action": info_action,
                    "info_interval": info_intervall,
                    "debug_effort": debug_effort,
                    "evaluation": False,
                },
            )
            .framework(framework)
            .training(
                train_batch_size=train_batch_size,
                gamma=gamma,
                lr=lr,
                grad_clip=grad_clip,
                entropy_coeff=entropy_coeff,
                vf_loss_coeff=vf_loss_coeff,
                model={
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "tanh",
                },
            )
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    # RLlib handles workers through config.env_runners()
    config = _set_rollout_workers_compat(config, num_workers=num_workers)
    config = config.env_runners(
        rollout_fragment_length=rollout_fragment_length,
        num_envs_per_env_runner=num_envs_per_worker,
        # Create env on local worker to ensure space inference works 
        # even if num_env_runners=0 or remote workers are slow.
        create_env_on_local_worker=True,
        # Allow more time for sampling in complex CPU environments
        sample_timeout_s=3600,
    )

    # Robustness: enable fault tolerance for workers
    # If a worker crashes, RLlib will try to restart it.
    config = config.fault_tolerance(
        restart_failed_env_runners=True,
        max_num_env_runner_restarts=10,
        delay_between_env_runner_restarts_s=10,
        ignore_env_runner_failures=True,
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
    if evaluation_interval is not None and evaluation_interval > 0:
        config = config.evaluation(
            # Run one evaluation round every iteration.
            evaluation_interval=evaluation_interval,

            # Create extra evaluation EnvRunners in the evaluation group.
            evaluation_num_env_runners=2,

            # Run evaluation for a fixed number of episodes (total across all eval runners).
            evaluation_duration_unit="episodes",
            evaluation_duration=20, # Increased for robustness
            evaluation_config={
                "explore": False,
                "env_config": {
                    "evaluation": True,
                }
            },

            # Allow more time for evaluation episodes to complete
            evaluation_sample_timeout_s=1200.0, # Increased from 600
        )
    else:
        # Explicitly disable evaluation
        config = config.evaluation(
            evaluation_interval=0,
            evaluation_num_env_runners=0
        )

    # ------------------------------------------------------------------
    # Step 1: Verify train vs eval config consistency
    # ------------------------------------------------------------------
    def verify_config_consistency(train_cfg, eval_cfg):
        print("\n" + "=" * 40)
        print("CONFIG CONSISTENCY CHECK")
        print("=" * 40)
        
        # Keys to check in env_config
        env_keys = [
            "n_agents", "start_agents", "max_steps", "max_rewardless_steps",
            "n_groups", "max_peer_group_size", "n_projects_per_step",
            "max_projects_per_agent", "max_agent_age", "growth_rate",
            "acceptance_threshold", "reward_mode", "prestige_threshold",
            "novelty_threshold", "effort_threshold"
        ]
        
        train_env_cfg = train_cfg.get("env_config", {})
        eval_env_cfg = eval_cfg.get("env_config", {})
        
        mismatches = []
        for k in env_keys:
            t_val = train_env_cfg.get(k)
            e_val = eval_env_cfg.get(k)
            print(f"  {k:30} | Train: {str(t_val):10} | Eval: {str(e_val):10}")
            if t_val != e_val:
                mismatches.append(k)
        
        if mismatches:
            print("\n!!! CONFIG MISMATCH DETECTED in keys:", mismatches)
            # We don't fail yet, but we warn loudly.
        else:
            print("\n[OK] Train and Eval env_configs match.")
        print("=" * 40 + "\n")

    # The 'config' object here is the base config which is used for both training 
    # and (via evaluation_config overrides) evaluation.
    # RLlib merges evaluation_config into the base config for evaluation workers.
    
    # Extract effective eval config for verification
    eval_config_dict = config.to_dict().get("evaluation_config")
    if eval_config_dict is None:
        eval_config_dict = {}
    # Note: RLlib internally merges these, so we check what we passed.
    
    # Correct consistency check: train_config is a flat dict, let's wrap it
    # and also handle the evaluation config nesting correctly
    verify_config_consistency(
        {"env_config": train_config}, 
        {"env_config": {**train_config, **eval_config_dict.get("env_config", {})}}
    )

    if info_action:
        _num_workers = 1
        config = _set_rollout_workers_compat(config, num_workers=_num_workers)
        # Force evaluation to 0 if info_action
        config = config.evaluation(evaluation_num_env_runners=0, evaluation_interval=0)
    else:
        # Each worker runs its own environment and collects samples in parallel.
        _num_workers = num_workers
        config = _set_rollout_workers_compat(config, num_workers=_num_workers)

    # Ensure fault tolerance is also active for the final config
    config = config.fault_tolerance(
        restart_failed_env_runners=True,
        max_num_env_runner_restarts=10,
        delay_between_env_runner_restarts_s=10,
        ignore_env_runner_failures=True,
    )

    # ------------------------------------------------------------------
    # Step 2: Build or Restore Algorithm
    # ------------------------------------------------------------------
    if checkpoint:
        checkpoint_path = os.path.abspath(checkpoint)
        if not os.path.exists(checkpoint_path):
            alt_path = os.path.join("checkpoints", checkpoint)
            if os.path.exists(alt_path):
                checkpoint_path = os.path.abspath(alt_path)
            else:
                raise ValueError(f"Checkpoint path not found: {checkpoint_path} or {alt_path}")
        
        print(f"\nRestoring algorithm from checkpoint: {checkpoint_path}")
        from ray.rllib.algorithms.algorithm import Algorithm
        
        # We use the config to ensure the restored algo uses our current worker settings etc.
        algo_instance = Algorithm.from_checkpoint(checkpoint_path, config=config)
    else:
        algo_instance = config.build_algo()

    # ------------------------------------------------------------------
    # Step 3: Validate model spaces
    # ------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("MODEL & SPACE VERIFICATION")
    print("=" * 40)
    try:
        policy = algo_instance.get_policy("default_policy")
        print(f"Model type: {type(policy.model)}")
        print(f"Observation space: {policy.observation_space}")
        print(f"Action space:      {policy.action_space}")
    except Exception as e:
        print(f"Could not verify spaces: {e}")
    print("=" * 40 + "\n")

    history = []

    best_eval = float("-inf")
    best_eval_iter: int = -1
    best_train_return = float("-inf")
    best_checkpoint_path: Optional[str] = None
    last_checkpoint_path: Optional[str] = None

    prev_total_env_steps = 0
    total_steps_sampled = 0
    recent_returns = []

    try:
        print(f"\n=== {algo.upper()} TRAINING (single controlled agent) ===")
        if evaluation_interval is not None and evaluation_interval > 0:
            print("Evaluation enabled.")
        print(f"controlled_agent_id: {controlled_agent_id}")
        print(f"policy_config: {policy_config_name}  (group_policy_homogenous={group_policy_homogenous})")
        print(f"env: n_agents={n_agents}, start_agents={start_agents}, max_steps={max_steps}, "
              f"n_groups={n_groups}, max_peer_group_size={max_peer_group_size}")
        print(f"thresholds: prestige={prestige_threshold}, novelty={novelty_threshold}, effort={effort_threshold}")
        print(f"reward_function: {reward_function}, acceptance_threshold: {acceptance_threshold}\n")

        for i in range(iterations):
            # Check if we reached the total environment steps limit
            if total_env_steps is not None:
                if total_steps_sampled >= total_env_steps:
                    print(f"\nReached total_env_steps limit ({total_env_steps}). Total steps sampled: {total_steps_sampled}. Stopping training.")
                    break

            try:
                result = algo_instance.train()
            except Exception as e:
                print(f"\n!!! algo_instance.train() failed at iteration {i}:")
                traceback.print_exc()
                print("Attempting to recover: clearing caches and waiting 10s...")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Wait a bit longer to allow for worker restarts if they are in progress
                time.sleep(10)
                continue

            # Garbage collection every iteration for stability under Windows
            if (i + 1) % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Metriken extrahieren (uses env_runners-/evaluation-block out of results)
            metrics, global_step = extract_metrics(result, i, prev_total_env_steps)
            
            # Explicitly track sampled and trained steps
            current_iter_sampled = result.get("env_runners", {}).get("num_env_steps_sampled", 0)
            current_iter_trained = result.get("info", {}).get("learner", {}).get("default_policy", {}).get("num_agent_steps_trained", 0)
            if not current_iter_trained:
                 # fallback for different RLlib versions
                 current_iter_trained = result.get("num_agent_steps_trained", 0)
            
            total_steps_sampled = global_step
            prev_total_env_steps = global_step

            # Optionale, detaillierte Debug-Ausgaben zu den Raw-Result-Strukturen
            train_env = result.get("env_runners") or {}
            eval_env = (result.get("evaluation") or {}).get("env_runners") or {}

            # Safe extraction of eval and train returns for Logging/Checkpointing
            raw_eval_return = metrics.get("eval/episode_return_mean")
            raw_train_return = metrics.get("train/episode_return_mean")
            eval_return_val = _safe_float(raw_eval_return)
            train_return_val = _safe_float(raw_train_return)

            if not math.isnan(train_return_val):
                recent_returns.append(train_return_val)
            
            # Compute rolling mean over the last 10 iterations
            # and log every 10 iterations.
            mean_return_10 = float("nan")
            if (i + 1) % 10 == 0 and len(recent_returns) > 0:
                subset = recent_returns[-10:]
                mean_return_10 = sum(subset) / len(subset)

            has_valid_eval = not math.isnan(eval_return_val)
            has_valid_train = not math.isnan(train_return_val)

            status = ""
            _kl = _safe_float(metrics.get("ppo/mean_kl") or metrics.get("algo/mean_kl"))
            _vf = _safe_float(metrics.get("ppo/vf_explained_var") or metrics.get("algo/vf_explained_var"))
            if not math.isnan(_kl) and _kl > 0.05:
                status += "⚠ KL high "
            if not math.isnan(_vf) and _vf < 0:
                status += "⚠ critic bad "
            if not has_valid_eval:
                status += "(no_eval) "

            # Human-readable Returns for logging (handles None/NaN)
            eval_return_str = "n/a" if not has_valid_eval else f"{eval_return_val:8.2f}"
            train_return_str = "n/a" if not has_valid_train else f"{train_return_val:8.2f}"

            checkpoint_info = ""

            history_entry: Dict[str, Any] = {
                "iter": i,
                "eval_return": eval_return_val if has_valid_eval else float("nan"),
                "kl": _kl,
                "vf_var": _vf,
                "episode_reward_mean": raw_train_return,
                "episode_len_mean": metrics.get("train/episode_len_mean"),
                "timesteps_total": result.get("timesteps_total"),
                # Extract custom metrics for CSV logging
                "eff_invalid_rate": metrics.get("custom_metrics", {}).get("effort/invalid_rate_mean"),
                "chosen_prestige": metrics.get("custom_metrics", {}).get("effort/chosen_prestige_mean"),
                "chose_max_prestige": metrics.get("custom_metrics", {}).get("effort/chose_max_prestige_mean"),
                "chose_most_urgent": metrics.get("custom_metrics", {}).get("effort/chose_most_urgent_mean"),
                "env_steps_per_sec": metrics.get("perf/env_steps_per_sec"),
                "sample_time_ms": metrics.get("perf/sample_time_ms"),
                "train_time_ms": metrics.get("perf/train_time_ms"),
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
                    algo_instance.save_checkpoint(chkpt_dir)
                    # Save config as json in the checkpoint folder
                    with open(os.path.join(chkpt_dir, "config.json"), "w") as f:
                        json.dump(train_config, f, indent=4)
                    best_checkpoint_path = chkpt_dir
                    checkpoint_info += " | [Checkpoint saved] best"
                except Exception as e:
                    checkpoint_info += f" | [checkpoint] ERROR best: {e}"

                # WandB artifact for new best model
                if use_wandb and best_checkpoint_path is not None:
                    try:
                        artifact = wandb.Artifact(
                            name=f"{algo.lower()}-best-{policy_config_name}-s{seed}",
                            type="model",
                        )
                        if os.path.isdir(best_checkpoint_path):
                            artifact.add_dir(best_checkpoint_path)
                        elif os.path.isfile(best_checkpoint_path):
                            artifact.add_file(best_checkpoint_path)
                        wandb.log_artifact(artifact)
                    except Exception as e:
                        checkpoint_info += f" | [wandb] artifact error: {e}"
            elif not has_valid_eval:
                # No evaluation was run this iteration (expected for non-eval iterations)
                if (i + 1) % 5 == 0:
                    # Only warn when evaluation *should* have happened
                    checkpoint_info += " | [checkpoint] no valid eval"

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
                    algo_instance.save_checkpoint(chkpt_dir)
                    # Save config as json in the checkpoint folder
                    with open(os.path.join(chkpt_dir, "config.json"), "w") as f:
                        json.dump(train_config, f, indent=4)
                    last_checkpoint_path = chkpt_dir
                    checkpoint_info += " | [Checkpoint saved] periodic"
                except Exception as e:
                    checkpoint_info += f" | [checkpoint] ERROR periodic: {e}"

            log_line = (
                f"Iter {i:03d} | "
                f"Steps: {total_steps_sampled:8d} | "
                f"TrainReturn: {train_return_str} | "
                f"EvalReturn: {eval_return_str} | "
            )
            if not math.isnan(_kl):
                log_line += f"KL: {_kl:6.4f} | "
            if not math.isnan(_vf):
                log_line += f"VF_var: {_vf:7.4f} "
            
            log_line += f"{status}{checkpoint_info}"
            print(log_line)

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
                    # Log rolling mean if available
                    if not math.isnan(mean_return_10):
                        log_dict["train/mean_return_10"] = mean_return_10

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

        algo_instance.stop()
        ray.shutdown()

    # Persist training history to CSV and plot (if matplotlib is available)
    results_paths = plot_training_history(history, save_prefix=f"{algo.lower()}_history_{policy_config_name}_{seed}")

    # Upload best checkpoint as artifact if available (final, deduplicated)
    if use_wandb and best_checkpoint_path is not None:
        try:
            artifact = wandb.Artifact(
                name=f"{algo.lower()}-best-{policy_config_name}-s{seed}",
                type="model",
            )
            if os.path.isdir(best_checkpoint_path):
                artifact.add_dir(best_checkpoint_path)
            elif os.path.isfile(best_checkpoint_path):
                artifact.add_file(best_checkpoint_path)
            wandb.log_artifact(artifact)
        except Exception as e:
            print(f"[checkpoint] WARNING: final wandb artifact upload failed: {e}")

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

    if use_wandb:
        try:
            # Final log of the mean over the last 10 iterations (or all if < 10)
            if len(recent_returns) > 0:
                subset = recent_returns[-10:]
                final_mean_return = sum(subset) / len(subset)
                # Use the last known global step or iteration index for final logging
                last_step = i
                if 'result' in locals():
                    last_step = result.get("timesteps_total") or i
                wandb.log({"train/final_mean_return": final_mean_return}, step=int(last_step))

            wandb.finish()
        except Exception as e:
            print(f"wandb.finish() failed: {e}")

    return history


def plot_training_history(history: list, out_dir: str = "results", save_prefix: str = "ppo_history"):
    """Save training history to CSV and (if available) plot eval_return, kl and vf_var.

    Expects history to be a list of dicts with keys: 'iter', 'eval_return', 'kl', 'vf_var'.
    Creates `out_dir/{save_prefix}.csv` and `out_dir/{save_prefix}.png` (if matplotlib present).
    Returns paths written.
    """
    os.makedirs(out_dir, exist_ok=True)

    iters = [h.get("iter", idx) for idx, h in enumerate(history)]
    evals = [float(h.get("eval_return") if h.get("eval_return") is not None else float("nan")) for h in history]
    kls = [float(h.get("kl") if h.get("kl") is not None else float("nan")) for h in history]
    vfs = [float(h.get("vf_var") if h.get("vf_var") is not None else float("nan")) for h in history]
    eff_inv = [float(h.get("eff_invalid_rate") if h.get("eff_invalid_rate") is not None else float("nan")) for h in history]
    chose_pre = [float(h.get("chosen_prestige") if h.get("chosen_prestige") is not None else float("nan")) for h in history]

    csv_path = os.path.join(out_dir, f"{save_prefix}.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["iter", "eval_return", "kl", "vf_var", "eff_invalid_rate", "chosen_prestige"])
        for i, e, k, v, ei, cp in zip(iters, evals, kls, vfs, eff_inv, chose_pre):
            writer.writerow([i, e, k, v, ei, cp])
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

    parser.add_argument(
        "--algo",
        type=str,
        default="PPO",
        choices=["PPO", "APPO"],
        help='RL algorithm to use: "PPO" or "APPO" (for DreamerV3 use train_dreamerv3.py)',
    )

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
    parser.add_argument("--n-agents", type=int, default=400)
    parser.add_argument("--start-agents", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--max-rewardless-steps", type=int, default=50)
    parser.add_argument("--n-groups", type=int, default=10)
    parser.add_argument("--max-peer-group-size", type=int, default=40)
    parser.add_argument("--n-projects-per-step", type=int, default=1)
    parser.add_argument("--max-projects-per-agent", type=int, default=8)
    parser.add_argument("--max-agent-age", type=int, default=750)

    # Reward knobs
    parser.add_argument("--acceptance-threshold", type=float, default=0.44)
    parser.add_argument("--reward-function", type=str, default="by_effort", choices=["multiply", "evenly", "by_effort"])

    # Heuristic thresholds (same as your simulation script)
    parser.add_argument("--prestige-threshold", type=float, default=0.29)
    parser.add_argument("--novelty-threshold", type=float, default=0.4)
    parser.add_argument("--effort-threshold", type=int, default=35)

    # Controlled agent
    parser.add_argument("--controlled-agent-id", type=str, default="agent_0")

    # Top-k Collaboration ablation
    parser.add_argument(
        "--topk", type=int, default=None,
        help="If set, restricts collaboration to top-k partners per step (default: None)",
    )
    parser.add_argument(
        "--topk-all-agents", action="store_true", default=False,
        help="If set, applies top-k also to heuristic agents (default: False)",
    )
    parser.add_argument(
        "--no-topk-all-agents", action="store_false", dest="topk_all_agents",
        help="If set, applies top-k ONLY to the controlled RL agent",
    )

    # Wandb options
    parser.add_argument("--wandb-project", type=str, default="RL in the Game of Science")
    parser.add_argument("--wandb-entity", type=str, default="rl_in_the_game_of_science")
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-run-id", type=str, default=None,
                        help="ID of an existing WandB run to resume.")

    # Action debug / info options
    parser.add_argument(
        "--info-action",
        action="store_true",
        default=False,
        help="Periodically prints the chosen action of the controlled agent (not enabled by default).",
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

    parser.add_argument(
        "--debug-effort",
        action="store_true",
        help="Enable detailed logging for action components (raw, scaled, rounded) in the wrapper.",
    )

    parser.add_argument("--train-batch-size", type=int, default=18000,
                        help="Number of env steps collected per training iteration.")

    parser.add_argument("--vf-share-layers", action="store_true", default=True,
                        help="Whether to share layers between the value function and the policy.")
    parser.add_argument("--no-vf-share-layers", action="store_false", dest="vf_share_layers",
                        help="Disable layer sharing between the value function and the policy.")

    parser.add_argument("--save-every-n-iters", type=int, default=10,
                        help="Save a periodic checkpoint every N iterations (0 = disabled).")
    parser.add_argument("--evaluation-interval", type=int, default=3,
                        help="Evaluate the policy every N iterations (0 = disabled).")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a checkpoint to restore from.")

    # Parallelization settings
    parser.add_argument("--num-workers", type=int, default=5,
                        help="Number of RLlib rollout workers.")
    parser.add_argument("--num-envs-per-worker", type=int, default=1,
                        help="Number of environments per worker.")
    parser.add_argument("--rollout-fragment-length", type=int, default=200,
                        help="Number of steps to collect per fragment.")

    # RL training hyperparameters
    parser.add_argument("--gamma", type=float, default=0.973060999938588)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.9585499239587636)
    parser.add_argument("--lr", type=float, default=0.00004299021945559274)
    parser.add_argument("--num-epochs", type=int, default=6)
    parser.add_argument("--entropy-coeff", type=float, default=0.0005)
    parser.add_argument("--vf-loss-coeff", type=float, default=2.5)
    parser.add_argument("--grad-clip", type=float, default=0.47456641063621474)

    args = parser.parse_args()

    main(
        algo=args.algo,
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
        topk_collab=args.topk,
        topk_apply_to_all_agents=args.topk_all_agents,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_mode=args.wandb_mode,
        wandb_run_id=args.wandb_run_id,
        info_action=args.info_action,
        info_intervall=args.info_intervall,
        debug_effort=args.debug_effort,
        train_batch_size=args.train_batch_size,
        save_every_n_iters=args.save_every_n_iters,
        evaluation_interval=args.evaluation_interval,
        gamma=args.gamma,
        lambda_=args.lambda_,
        lr=args.lr,
        num_epochs=args.num_epochs,
        entropy_coeff=args.entropy_coeff,
        vf_loss_coeff=args.vf_loss_coeff,
        grad_clip=args.grad_clip,
        vf_share_layers=args.vf_share_layers,
        num_workers=args.num_workers,
        num_envs_per_worker=args.num_envs_per_worker,
        rollout_fragment_length=args.rollout_fragment_length,
        checkpoint=args.checkpoint,
    )
