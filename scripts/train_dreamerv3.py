"""
train_dreamerv3.py

DreamerV3 training script for Game-of-Science environment with RLlib.

Features:
- GPU training with validation
- MultiDiscrete action space support
- Checkpoint management with configurable intervals
- WandB integration (optional)
- Smoke test mode for quick validation
- Fair comparison with PPO wrapper

DreamerV3 uses RLlib's new API stack (RLModule, EnvRunnerV2).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback

import numpy as np
import ray
import torch
import wandb
from lightning.pytorch import seed_everything
from ray import tune
from ray.rllib.algorithms.dreamerv3 import DreamerV3Config
from typing import Any, Callable, Dict, Optional

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent_policies import (
    create_mixed_policy_population,
    create_per_group_policy_population,
    do_nothing_policy,
    get_policy_function,
)
from src.env.peer_group_environment import PeerGroupEnvironment
from src.dreamerv3_wrapper import DreamerV3SingleAgentWrapper
from src.callbacks.papers_metrics_callback import PapersMetricsCallback

# Suppress Ray warnings
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

# Policy distribution presets
POLICY_CONFIGS = {
    "Balanced": {"careerist": 0.33, "orthodox_scientist": 0.33, "mass_producer": 0.34},
    "CareeristHeavy": {"careerist": 0.7, "orthodox_scientist": 0.15, "mass_producer": 0.15},
    "OrthodoxHeavy": {"careerist": 0.15, "orthodox_scientist": 0.7, "mass_producer": 0.15},
    "MassProducerHeavy": {"careerist": 0.15, "orthodox_scientist": 0.15, "mass_producer": 0.7},
}


def check_cuda(num_gpus: int, abort_on_missing: bool = True) -> bool:
    """
    Check CUDA availability and log device information.

    Args:
        num_gpus: Number of GPUs requested
        abort_on_missing: If True, raise error when CUDA unavailable but requested

    Returns:
        True if CUDA is available and can be used
    """
    cuda_available = torch.cuda.is_available()

    print("=" * 70)
    print("GPU / CUDA Check")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("No CUDA devices found.")

    print(f"Requested GPUs: {num_gpus}")
    print("=" * 70)

    if num_gpus > 0 and not cuda_available:
        msg = (
            f"\nERROR: Requested {num_gpus} GPU(s) but CUDA is not available!\n"
            "Please install CUDA-enabled PyTorch or set --num-gpus 0 for CPU training."
        )
        if abort_on_missing:
            raise RuntimeError(msg)
        else:
            print(f"WARNING: {msg}")
            return False

    return cuda_available


def wandb_sanitize(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize metrics for WandB logging (remove non-scalar values)."""
    safe: Dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, (dict, list, tuple, set)):
            continue
        if hasattr(v, "item"):
            v = v.item()
        if isinstance(v, (int, float, bool, np.number)):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                continue
            safe[k] = v
    return safe


def make_env_creator(
    *,
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
    policy_distribution: Dict[str, float],
    group_policy_homogenous: bool,
    prestige_threshold: float,
    novelty_threshold: float,
    effort_threshold: int,
    controlled_agent_id: str,
    topk_collab: Optional[int] = None,
    debug_effort: bool = False,
    use_light_policy_obs: bool = False,
    debug_actions: bool = False,
    action_space_type: str = "box",
) -> Callable[[Optional[Dict[str, Any]]], Any]:
    """
    Create environment factory for RLlib registration.

    Returns a callable that creates DreamerV3SingleAgentWrapper instances.
    """

    careerist_fn = get_policy_function("careerist")
    orthodox_fn = get_policy_function("orthodox_scientist")
    mass_prod_fn = get_policy_function("mass_producer")

    def _policy_from_name(policy_name: Optional[str]):
        if policy_name is None:
            return lambda nested_obs: do_nothing_policy(nested_obs["observation"], nested_obs["action_mask"])
        if policy_name == "careerist":
            return lambda nested_obs: careerist_fn(nested_obs["observation"], nested_obs["action_mask"], prestige_threshold)
        if policy_name == "orthodox_scientist":
            return lambda nested_obs: orthodox_fn(nested_obs["observation"], nested_obs["action_mask"], novelty_threshold)
        if policy_name == "mass_producer":
            return lambda nested_obs: mass_prod_fn(nested_obs["observation"], nested_obs["action_mask"], effort_threshold)
        return lambda nested_obs: do_nothing_policy(nested_obs["observation"], nested_obs["action_mask"])

    def _env_creator(env_config: Optional[Dict[str, Any]] = None):
        env_config = env_config or {}

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
        )

        if group_policy_homogenous:
            policy_names_list = create_per_group_policy_population(n_agents, policy_distribution)
        else:
            policy_names_list = create_mixed_policy_population(n_agents, policy_distribution, seed=seed)

        agent_policy_names = {
            f"agent_{i}": name for i, name in enumerate(policy_names_list)
        }

        other_policies: Dict[str, Callable[[Any], Any]] = {}
        for agent_id in env.possible_agents:
            if agent_id == controlled_agent_id:
                continue
            p_name = agent_policy_names.get(agent_id)
            other_policies[agent_id] = _policy_from_name(p_name)

        return DreamerV3SingleAgentWrapper(
            env,
            controlled_agent=controlled_agent_id,
            other_policies=other_policies,
            max_peer_group_size=env_config.get("max_peer_group_size", max_peer_group_size),
            topk_collab=topk_collab,
            debug_effort=debug_effort,
            use_light_policy_obs=use_light_policy_obs,
            debug_actions=debug_actions,
            action_space_type=env_config.get("action_space_type", action_space_type),
            env_config=env_config,
            debug_action_mask=env_config.get("debug_action_mask", False),
            debug_action_mask_steps=env_config.get("debug_action_mask_steps", 50),
            debug_action_mask_interval=env_config.get("debug_action_mask_interval", 100),
            debug_action_mask_jsonl=env_config.get("debug_action_mask_jsonl", None),
            invalid_action_penalty=env_config.get("invalid_action_penalty", 0.0),
        )

    return _env_creator


def smoke_test_env(args: argparse.Namespace) -> bool:
    """
    Run a quick smoke test to verify the environment wrapper works.

    Returns:
        True if smoke test passes, False otherwise
    """
    print("=" * 70)
    print("SMOKE TEST MODE")
    print("=" * 70)

    try:
        # Create environment
        env_creator = make_env_creator(
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
            seed=args.seed,
            policy_distribution=POLICY_CONFIGS[args.policy_config],
            group_policy_homogenous=args.group_policy_homogenous,
            prestige_threshold=args.prestige_threshold,
            novelty_threshold=args.novelty_threshold,
            effort_threshold=args.effort_threshold,
            controlled_agent_id=args.controlled_agent_id,
            topk_collab=args.topk_collab,
            debug_effort=args.debug_effort,
            use_light_policy_obs=args.use_light_policy_obs,
            debug_actions=args.debug_actions,
        )

        env = env_creator()

        print(f"[OK] Environment created successfully")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")

        # Reset environment
        obs, info = env.reset(seed=args.seed)
        print(f"[OK] Environment reset successfully")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Observation in space: {env.observation_space.contains(obs)}")

        # Sample and step through environment
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if not env.observation_space.contains(obs):
                print(f"[X] Step {step}: Observation not in observation_space!")
                return False

            if terminated or truncated:
                obs, info = env.reset(seed=args.seed + step + 1)
                print(f"  Episode ended at step {step}, reset environment")

        print(f"[OK] Completed 10 steps without errors")
        print(f"  Final observation shape: {obs.shape}")
        print(f"  Final reward: {reward}")

        env.close()
        print("[OK] Environment closed successfully")

        print("=" * 70)
        print("SMOKE TEST PASSED")
        print("=" * 70)
        return True

    except Exception as e:
        print("=" * 70)
        print("SMOKE TEST FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def validate_config(args: argparse.Namespace) -> None:
    """
    Validate configuration before starting training.

    Raises ValueError if configuration is invalid.
    """
    print("\n" + "=" * 70)
    print("CONFIGURATION VALIDATION")
    print("=" * 70)

    errors = []

    # Guard 1: max_peer_group_size must be divisible by 10
    if args.max_peer_group_size % 10 != 0:
        errors.append(
            f"[X] max_peer_group_size must be divisible by 10!\n"
            f"   Current value: {args.max_peer_group_size}\n"
            f"   Valid values: 10, 20, 30, 40, 50, 60, ..."
        )
    else:
        print(f"[OK] max_peer_group_size={args.max_peer_group_size} is divisible by 10")

    # Guard 2: average group size must not exceed max_peer_group_size
    avg_group_size = args.n_agents / args.n_groups
    if avg_group_size > args.max_peer_group_size:
        errors.append(
            f"[X] Average peer group size exceeds max_peer_group_size!\n"
            f"   n_agents / n_groups = {args.n_agents} / {args.n_groups} = {avg_group_size:.2f}\n"
            f"   max_peer_group_size = {args.max_peer_group_size}\n"
            f"   This would cause a broadcast error in the environment.\n"
            f"   Solutions:\n"
            f"     - Increase n_groups (e.g., {int(args.n_agents / args.max_peer_group_size) + 1}+)\n"
            f"     - Decrease n_agents (e.g., {int(args.n_groups * args.max_peer_group_size)})\n"
            f"     - Increase max_peer_group_size (e.g., {int((avg_group_size // 10 + 1) * 10)})"
        )
    else:
        print(f"[OK] avg_group_size={avg_group_size:.1f} <= max_peer_group_size={args.max_peer_group_size}")

    print("=" * 70)

    if errors:
        print("\n" + "=" * 70)
        print("CONFIGURATION ERRORS DETECTED")
        print("=" * 70)
        for error in errors:
            print(error)
            print()
        print("=" * 70)
        print("Training aborted due to invalid configuration.")
        print("=" * 70)
        sys.exit(1)

    print("[OK] Configuration is valid\n")


def flatten_numeric(prefix: str, obj: Any, out: Dict[str, float]) -> None:
    """Recursively flatten all scalar numeric values from nested dicts/lists."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}/{k}" if prefix else k
            flatten_numeric(new_prefix, v, out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            new_prefix = f"{prefix}/{i}" if prefix else str(i)
            flatten_numeric(new_prefix, v, out)
    elif isinstance(obj, (int, float, np.number, np.bool_)):
        val = float(obj)
        if not (math.isnan(val) or math.isinf(val)):
            out[prefix] = val


def collect_wandb_metrics(result: Dict[str, Any], iteration: int, total_env_steps: int) -> Optional[Dict[str, Any]]:
    """
    Collect comprehensive metrics for WandB logging.
    """
    env_runners_data = result.get("env_runners", {})
    learner_data = result.get("learners", {})
    custom_metrics = env_runners_data.get("custom_metrics", {})

    # Build metrics dict
    metrics = {
        "iteration": int(iteration),
        "total_env_steps": int(total_env_steps),
    }

    # Helper to find a metric value with various fallbacks
    def get_metric(key_list):
        for k in key_list:
            # Try original key in both places
            for source in [env_runners_data, custom_metrics]:
                if k in source:
                    val = source[k]
                    if isinstance(val, (int, float, np.number)) and not math.isnan(val):
                        return float(val)
            
            # Try with _mean suffix
            if not k.endswith("_mean"):
                k_mean = f"{k}_mean"
                for source in [env_runners_data, custom_metrics]:
                    if k_mean in source:
                        val = source[k_mean]
                        if isinstance(val, (int, float, np.number)) and not math.isnan(val):
                            return float(val)
            
            # Try without _mean suffix
            if k.endswith("_mean"):
                k_no_mean = k[:-5]
                for source in [env_runners_data, custom_metrics]:
                    if k_no_mean in source:
                        val = source[k_no_mean]
                        if isinstance(val, (int, float, np.number)) and not math.isnan(val):
                            return float(val)
        return float('nan')

    # Always add learner metrics if available
    if learner_data:
        flatten_numeric("dreamer", learner_data, metrics)

    # -------------------------------------------------------------------------
    # Episode return and reward statistics
    # -------------------------------------------------------------------------
    val = get_metric(["episode_return", "episode_reward"])
    if not math.isnan(val):
        metrics["episode_return_mean"] = val

    val = get_metric(["episode_len", "episode_length"])
    if not math.isnan(val):
        metrics["episode_len_mean"] = val

    val = get_metric(["num_episodes_lifetime"])
    if not math.isnan(val):
        metrics["num_episodes_lifetime"] = int(val)

    # -------------------------------------------------------------------------
    # Custom metrics from PapersMetricsCallback
    # -------------------------------------------------------------------------
    
    # Mapping table for new metric structure
    mapping = {
        # Reward
        "reward/env_return_mean": ["reward_components/env_reward", "episode_reward_components/env_reward_sum"],
        "reward/training_return_mean": ["reward_components/training_reward", "episode_reward_components/training_reward_sum"],
        "reward/invalid_penalty_sum_mean": ["episode_reward_components/invalid_action_penalty_sum"],
        
        # Action validity
        "action/raw_invalid_rate": ["action_validity_episode/raw_invalid_rate"],
        "action/raw_choose_project_invalid_rate": ["action_validity_episode/raw_choose_project_invalid_rate"],
        "action/raw_put_effort_invalid_rate": ["action_validity_episode/raw_put_effort_invalid_rate"],
        "action/raw_collaborate_with_invalid_rate": ["action_validity_episode/raw_collaborate_with_invalid_rate"],
        "action/repair_rate": ["action_validity_episode/repair_rate"],
        "action/repair_choose_project_rate": ["action_validity_episode/repair_choose_project_rate"],
        "action/repair_put_effort_rate": ["action_validity_episode/repair_put_effort_rate"],
        "action/repair_collaborate_with_rate": ["action_validity_episode/repair_collaborate_with_rate"],
        "action/final_invalid_rate": ["action_validity/final_all_valid"], # Will be inverted below
        
        # Action distribution
        "action/choose_project_none_rate": ["action_distribution_episode/choose_project_none_rate"],
        "action/choose_project_nonzero_rate": ["action_distribution_episode/choose_project_nonzero_rate"],
        "action/put_effort_none_rate": ["action_distribution_episode/put_effort_none_rate"],
        "action/put_effort_nonzero_rate": ["action_distribution_episode/put_effort_nonzero_rate"],
        "action/collab_count_mean": ["action_distribution_episode/collab_count_mean"],
        "action/collab_count_max": ["action_distribution_episode/collab_count_max"],
        "action/raw_cont_choose_project_mean": ["action_distribution/raw_cont_choose_project"],
        "action/raw_cont_put_effort_mean": ["action_distribution/raw_cont_put_effort"],
        "action/raw_cont_collab_mean": ["action_distribution/raw_cont_collab_mean"],
        
        # RL Agent projects
        "rl_agent/projects_active": ["rl_agent_projects/active_projects"],
        "rl_agent/projects_completed": ["rl_agent_projects/completed_projects"],
        "rl_agent/projects_accepted": ["rl_agent_projects/accepted_projects"],
        "rl_agent/projects_rejected": ["rl_agent_projects/rejected_projects"],
        
        # RL Agent detailed stats
        "rl_agent/episode_return": ["rl_agent_stats/episode_return"],
        "rl_agent/accumulated_reward": ["rl_agent_stats/accumulated_reward"],
        "rl_agent/started_projects": ["rl_agent_stats/started_projects"],
        "rl_agent/steps_until_first_reward": ["rl_agent_stats/steps_until_first_reward"],
        "rl_agent/reward_nonzero_frac": ["rl_agent_stats/reward_nonzero_frac"],
        "rl_agent/age_at_done": ["rl_agent_stats/age_at_done"],
        "rl_agent/rewardless_steps": ["rl_agent_stats/rewardless_steps"],
        "rl_agent/agent_removed": ["rl_agent_stats/agent_removed"],
        "rl_agent/termination_reason": ["rl_agent_stats/termination_reason"],
        
        # Decision quality (from effort_analysis)
        "rl_agent/effort_applied_sum": ["agent0_effort_applied_sum"],
        "rl_agent/active_projects_mean": ["agent0_active_projects_mean"],
        "rl_agent/chosen_remaining_time": ["effort/chosen_remaining_time"],
        "rl_agent/min_remaining_time": ["effort/min_remaining_time"],
        "rl_agent/chose_most_urgent": ["effort/chose_most_urgent"],
        "rl_agent/chosen_prestige": ["effort/chosen_prestige"],
        "rl_agent/max_prestige": ["effort/max_prestige"],
        "rl_agent/chose_max_prestige": ["effort/chose_max_prestige"],
        
        # Global projects
        "global/projects_active": ["global_projects/active_projects"],
        "global/projects_published": ["global_projects/published_projects"],
        "global/projects_rejected": ["global_projects/rejected_projects"],
        "global/projects_total": ["papers_total"],
        
        # Observation
        "obs/abs_max": ["observation_stats/abs_max"],
        "obs/mean": ["observation_stats/mean"],
        "obs/std": ["observation_stats/std"],
        "obs/nan_count": ["observation_stats/nan_count"],
        "obs/inf_count": ["observation_stats/inf_count"],
    }

    for wandb_key, rllib_keys in mapping.items():
        val = get_metric(rllib_keys)
        if not math.isnan(val):
            metrics[wandb_key] = val

    # Legacy mappings / Fallbacks
    if "action/final_invalid_rate" in metrics:
        # If it's final_all_valid, then 1.0 means 0.0 invalid
        metrics["action/final_invalid_rate"] = 1.0 - metrics["action/final_invalid_rate"]

    # Original papers metrics for backward compatibility
    for key in ["papers_published_count", "papers_rejected_count", "papers_total", "papers_active_mean"]:
        val = get_metric([key])
        if not math.isnan(val):
            metric_key = f"papers/{key.replace('papers_', '')}"
            if metric_key not in metrics:
                metrics[metric_key] = val

    return metrics


def main(args: argparse.Namespace):
    """Main training loop for DreamerV3."""

    # Validate configuration first (always, even for smoke test)
    validate_config(args)

    # Smoke test mode
    if args.smoke_test:
        success = smoke_test_env(args)
        sys.exit(0 if success else 1)

    # GPU validation
    check_cuda(args.num_gpus, abort_on_missing=True)

    # Seeding
    seed_everything(args.seed)
    ray.init(ignore_reinit_error=True)

    # Convert checkpoint dir to absolute path (required by Ray)
    if args.checkpoint_dir and not os.path.isabs(args.checkpoint_dir):
        args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)

    # Create checkpoint directory if it doesn't exist
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Register environment
    env_name = "peer_group_env_dreamerv3"
    env_creator = make_env_creator(
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
        seed=args.seed,
        policy_distribution=POLICY_CONFIGS[args.policy_config],
        group_policy_homogenous=args.group_policy_homogenous,
        prestige_threshold=args.prestige_threshold,
        novelty_threshold=args.novelty_threshold,
        effort_threshold=args.effort_threshold,
        controlled_agent_id=args.controlled_agent_id,
        topk_collab=args.topk_collab,
        debug_effort=args.debug_effort,
        use_light_policy_obs=args.use_light_policy_obs,
        debug_actions=args.debug_actions,
        action_space_type=args.action_space_type,
    )
    tune.register_env(env_name, env_creator)

    # DreamerV3 configuration
    env_config_dict = {
        "evaluation": False,
        "debug_action_mask": args.debug_action_mask,
        "debug_actions": args.debug_actions,
        "debug_action_mask_steps": args.debug_action_mask_steps,
        "debug_action_mask_interval": args.debug_action_mask_interval,
        "debug_action_mask_jsonl": args.debug_action_mask_jsonl,
        "invalid_action_penalty": args.invalid_action_penalty,
        "base_seed": args.seed,
    }

    config = (
        DreamerV3Config()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .framework("torch")
        .training(
            model_size=args.model_size,
            training_ratio=args.training_ratio,
            batch_size_B=args.batch_size_B,
            batch_length_T=args.batch_length_T,
            gamma=args.gamma,
            world_model_lr=args.lr,
            actor_lr=args.actor_lr if args.actor_lr is not None else args.lr,
            critic_lr=args.lr,
        )
        .environment(
            env=env_name,
            env_config=env_config_dict,
        )
        .env_runners(
            num_env_runners=args.num_workers,
            num_envs_per_env_runner=args.num_envs_per_worker,
            create_env_on_local_worker=True,
            sample_timeout_s=3600,
        )
        .learners(
            num_learners=0,  # 0 means use local learner
            num_gpus_per_learner=args.num_gpus,  # Assign GPU to the learner
        )
        .resources(
            num_gpus=0,  # Don't assign GPU at driver level, only to learner
        )
        .callbacks(PapersMetricsCallback)
        .debugging(seed=args.seed)
    )

    # WandB initialization
    use_wandb = args.wandb_mode != "disabled"
    if use_wandb:
        try:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                group=args.wandb_group,
                name=args.wandb_run_name,
                mode=args.wandb_mode,
                config=vars(args),
            )
            wandb.define_metric("total_env_steps")
            wandb.define_metric("*", step_metric="total_env_steps")
            print("[OK] WandB initialized successfully")
        except Exception as e:
            print(f"WARNING: WandB initialization failed: {e}")
            use_wandb = False

    # Build algorithm
    print("\n" + "=" * 70)
    print("Building DreamerV3 Algorithm")
    print("=" * 70)

    algo = config.build()

    print("[OK] Algorithm built successfully")
    print(f"  Model size: {args.model_size}")
    print(f"  Training ratio: {args.training_ratio}")
    print(f"  Batch size (B): {args.batch_size_B}")
    print(f"  Batch length (T): {args.batch_length_T}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Gamma: {args.gamma}")
    print(f"  Num workers: {args.num_workers}")
    print(f"  Envs per worker: {args.num_envs_per_worker}")
    print(f"  Num GPUs: {args.num_gpus}")

    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    print(f"Target env steps: {args.total_env_steps:,}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Checkpoint interval: {args.checkpoint_interval}")
    print("=" * 70 + "\n")

    total_env_steps = 0
    iteration = 0
    last_log_iteration = 0
    
    # Track last known metrics for stable console logging
    last_metrics = {
        "reward": float('nan'),
        "len": float('nan'),
        "invalid": float('nan'),
        "repair": float('nan')
    }

    try:
        while total_env_steps < args.total_env_steps:
            iteration += 1

            # Train one iteration
            result = algo.train()

            # Debug: Print available keys on first iteration (enable for debugging)
            if iteration == 1 and args.debug_metrics:
                print(f"\n[DEBUG] Available result keys: {list(result.keys())}")
                if "env_runners" in result:
                    env_keys = result['env_runners'].keys()
                    print(f"[DEBUG] env_runners keys: {list(env_keys)}")
                    # Print all keys that contain "paper" or "effort"
                    paper_keys = [k for k in env_keys if "paper" in k.lower()]
                    effort_keys = [k for k in env_keys if "effort" in k.lower()]
                    custom_keys = [k for k in env_keys if "custom" in k.lower()]
                    if paper_keys:
                        print(f"[DEBUG] Paper-related keys: {paper_keys}")
                    if effort_keys:
                        print(f"[DEBUG] Effort-related keys: {effort_keys}")
                    if custom_keys:
                        print(f"[DEBUG] Custom metrics keys: {custom_keys}")
                        # Print content of custom_metrics if available
                        for key in custom_keys:
                            print(f"[DEBUG] {key} = {result['env_runners'][key]}")
                if "learners" in result:
                    learner_keys = result.get("learners", {}).get("default_policy", {})
                    print(f"[DEBUG] learner keys: {list(learner_keys.keys())}")

            # Extract timesteps - try multiple keys for compatibility
            env_runners_data = result.get("env_runners", {})
            env_steps_sampled = env_runners_data.get("num_env_steps_sampled_lifetime", 0)
            if env_steps_sampled == 0:
                env_steps_sampled = result.get("num_env_steps_sampled_lifetime", 0)
            if env_steps_sampled == 0:
                env_steps_sampled = result.get("timesteps_total", 0)

            total_env_steps = int(env_steps_sampled)

            # Extract metrics - try multiple keys for compatibility
            episode_reward_mean = env_runners_data.get("episode_return_mean", float('nan'))
            if episode_reward_mean != episode_reward_mean:  # Check for NaN
                episode_reward_mean = result.get("episode_return_mean", float('nan'))
            if episode_reward_mean != episode_reward_mean:  # Still NaN?
                episode_reward_mean = env_runners_data.get("episode_reward_mean", float('nan'))

            episode_len_mean = env_runners_data.get("episode_len_mean", float('nan'))
            if episode_len_mean != episode_len_mean:  # Check for NaN
                episode_len_mean = result.get("episode_len_mean", float('nan'))

            # Update last known metrics
            if not math.isnan(episode_reward_mean):
                last_metrics["reward"] = episode_reward_mean
            if not math.isnan(episode_len_mean):
                last_metrics["len"] = episode_len_mean

            # Extract additional metrics
            num_episodes_lifetime = int(env_runners_data.get("num_episodes_lifetime", 0))
            papers_published = env_runners_data.get("papers_published_count", float('nan'))
            papers_total = env_runners_data.get("papers_total", float('nan'))
            
            # Fallback for papers from custom_metrics
            if papers_published != papers_published:
                papers_published = env_runners_data.get("custom_metrics", {}).get("papers_published_count_mean", float('nan'))
            if papers_total != papers_total:
                papers_total = env_runners_data.get("custom_metrics", {}).get("papers_total_mean", float('nan'))

            # Determine if we have new episode data (not NaN)
            has_new_episode = not math.isnan(episode_reward_mean)

            # Console logging: Every 10 or 25 iterations (or on first)
            log_interval = 25
            if iteration == 1 or iteration % log_interval == 0:
                # Use current or last known value
                display_reward = episode_reward_mean if not math.isnan(episode_reward_mean) else last_metrics["reward"]
                display_len = episode_len_mean if not math.isnan(episode_len_mean) else last_metrics["len"]

                reward_str = f"{display_reward:7.2f}" if not math.isnan(display_reward) else "   -   "
                len_str = f"{display_len:6.1f}" if not math.isnan(display_len) else "  -   "
                
                # Extract learner metrics for console
                learner_stats = result.get("learners", {}).get("default_policy", {})
                wm_loss = learner_stats.get("WORLD_MODEL_L_total", "-")
                critic_loss = learner_stats.get("CRITIC_L_total", "-")
                actor_entropy = learner_stats.get("ACTOR_action_entropy", "-")
                
                # Extract validity/repair rates
                env_runners_data = result.get("env_runners", {})
                custom_metrics = env_runners_data.get("custom_metrics", {})
                
                # Try new API stack keys (direct in env_runners)
                invalid_rate = env_runners_data.get("action_validity_episode/raw_invalid_rate", float('nan'))
                if math.isnan(invalid_rate):
                    # Try custom_metrics (old API stack or mapped)
                    invalid_rate = custom_metrics.get("action_validity_episode/raw_invalid_rate_mean", float('nan'))
                
                repair_rate = env_runners_data.get("action_validity_episode/repair_rate", float('nan'))
                if math.isnan(repair_rate):
                    # Try custom_metrics
                    repair_rate = custom_metrics.get("action_validity_episode/repair_rate_mean", float('nan'))
                
                # Update last known validity/repair
                if not math.isnan(invalid_rate):
                    last_metrics["invalid"] = invalid_rate
                if not math.isnan(repair_rate):
                    last_metrics["repair"] = repair_rate
                
                # Use last known if current is NaN
                display_inv = invalid_rate if not math.isnan(invalid_rate) else last_metrics["invalid"]
                display_rep = repair_rate if not math.isnan(repair_rate) else last_metrics["repair"]

                # Format values
                wm_str = f"{wm_loss:5.2f}" if isinstance(wm_loss, (int, float)) else f"{wm_loss:>5}"
                critic_str = f"{critic_loss:5.2f}" if isinstance(critic_loss, (int, float)) else f"{critic_loss:>5}"
                ent_str = f"{actor_entropy:5.1f}" if isinstance(actor_entropy, (int, float)) else f"{actor_entropy:>5}"
                inv_str = f"{display_inv:4.2f}" if isinstance(display_inv, (int, float)) and not math.isnan(display_inv) else "  - "
                rep_str = f"{display_rep:4.2f}" if isinstance(display_rep, (int, float)) and not math.isnan(display_rep) else "  - "

                print(f"Iter {iteration:4d} | "
                      f"Steps {total_env_steps:9,} / {args.total_env_steps:,} | "
                      f"Eps {num_episodes_lifetime:4d} | "
                      f"Return {reward_str} | "
                      f"Len {len_str} | "
                      f"WM {wm_str} | "
                      f"Critic {critic_str} | "
                      f"Ent {ent_str} | "
                      f"Invalid {inv_str} | "
                      f"Repair {rep_str}")

            # WandB logging: Always log (learner metrics are always available)
            if use_wandb:
                wandb_metrics = collect_wandb_metrics(result, iteration, total_env_steps)
                if wandb_metrics:
                    wandb.log(wandb_metrics, step=int(total_env_steps))
                    last_log_iteration = iteration

            # Checkpoint management
            if args.checkpoint_interval > 0 and iteration % args.checkpoint_interval == 0:
                checkpoint_path = algo.save(checkpoint_dir=args.checkpoint_dir)
                # print(f"  → Checkpoint saved: {checkpoint_path}")

        # Final checkpoint
        if args.checkpoint_dir:
            final_checkpoint = algo.save(checkpoint_dir=args.checkpoint_dir)
            print(f"\n[OK] Final checkpoint saved: {final_checkpoint}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    except Exception as e:
        print(f"\n\nERROR during training: {e}")
        traceback.print_exc()

    finally:
        # Cleanup
        print("\nCleaning up...")
        algo.stop()
        ray.shutdown()

        if use_wandb:
            wandb.finish()

        print("[OK] Training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DreamerV3 on Game-of-Science environment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Training parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--total-env-steps", type=int, default=300000, help="Total environment steps to train")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/dreamerv3", help="Checkpoint directory")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Save checkpoint every N iterations")

    # GPU / device
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs (0 for CPU)")

    # Environment parameters
    parser.add_argument("--policy-config", type=str, default="Balanced", choices=POLICY_CONFIGS.keys(), help="Policy distribution preset")
    parser.add_argument("--group-policy-homogenous", action="store_true", help="Use homogenous policies per group")
    parser.add_argument("--n-agents", type=int, default=400, help="Maximum number of agents")
    parser.add_argument("--start-agents", type=int, default=100, help="Starting number of agents")
    parser.add_argument("--max-steps", type=int, default=400, help="Maximum episode steps")
    parser.add_argument("--max-rewardless-steps", type=int, default=50, help="Max steps without reward before termination")
    parser.add_argument("--n-groups", type=int, default=10, help="Number of peer groups")
    parser.add_argument("--max-peer-group-size", type=int, default=40, help="Maximum peer group size")
    parser.add_argument("--n-projects-per-step", type=int, default=1, help="New projects per step")
    parser.add_argument("--max-projects-per-agent", type=int, default=8, help="Max active projects per agent")
    parser.add_argument("--max-agent-age", type=int, default=750, help="Maximum agent age")
    parser.add_argument("--acceptance-threshold", type=float, default=0.44, help="Project acceptance threshold")
    parser.add_argument("--reward-function", type=str, default="by_effort", choices=["multiply", "evenly", "by_effort"], help="Reward distribution mode")

    # Agent policy thresholds
    parser.add_argument("--prestige-threshold", type=float, default=0.29, help="Careerist prestige threshold")
    parser.add_argument("--novelty-threshold", type=float, default=0.4, help="Orthodox scientist novelty threshold")
    parser.add_argument("--effort-threshold", type=int, default=35, help="Mass producer effort threshold")
    parser.add_argument("--controlled-agent-id", type=str, default="agent_0", help="Controlled agent ID")

    # Collaboration ablation
    parser.add_argument("--topk-collab", type=int, default=None, help="Top-k collaboration constraint (None = disabled)")

    # Performance optimization
    parser.add_argument("--use-light-policy-obs", action="store_true", default=False, help="Use lightweight observations for non-controlled agents (performance optimization)")

    # DreamerV3 hyperparameters
    parser.add_argument("--model-size", type=str, default="S", choices=["XS", "S", "M", "L", "XL"], help="DreamerV3 model size")
    parser.add_argument("--training-ratio", type=int, default=64, help="Training steps per environment step (default: 64, original paper: 1024)")
    parser.add_argument("--batch-size-B", type=int, default=16, help="Batch size (B)")
    parser.add_argument("--batch-length-T", type=int, default=128, help="Batch sequence length (T, default: 16)")
    parser.add_argument("--gamma", type=float, default=0.997, help="Discount factor")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for world model and critic")
    parser.add_argument("--actor-lr", type=float, default=1e-4, help="Actor learning rate (defaults to --lr if not set)")

    # RLlib workers
    parser.add_argument("--num-workers", type=int, default=0, help="Number of environment workers")
    parser.add_argument("--num-envs-per-worker", type=int, default=1, help="Environments per worker")

    # WandB
    parser.add_argument("--wandb-project", type=str, default="RL in the Game of Science", help="WandB project name")
    parser.add_argument("--wandb-entity", type=str, default="rl_in_the_game_of_science", help="WandB entity")
    parser.add_argument("--wandb-group", type=str, default="DreamerV3", help="WandB group")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="WandB run name")
    parser.add_argument("--wandb-mode", type=str, default="disabled", choices=["online", "offline", "disabled"], help="WandB mode")

    # Debug
    parser.add_argument("--debug-effort", action="store_true", help="Enable effort allocation debugging")
    parser.add_argument("--debug-actions", action="store_true", help="Print observations and actions at each step for debugging")
    parser.add_argument("--debug-metrics", action="store_true", help="Print all available metrics on first iteration")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test and exit")

    # Action Space
    parser.add_argument("--action-space-type", type=str, default="box", choices=["box", "discrete"], help="Action space type for DreamerV3")

    # Action mask debugging
    parser.add_argument("--debug-action-mask", action="store_true", help="Enable action mask and repair debugging")
    parser.add_argument("--debug-action-mask-steps", type=int, default=50, help="Number of initial steps to always debug")
    parser.add_argument("--debug-action-mask-interval", type=int, default=100, help="Debug every N steps after initial period")
    parser.add_argument("--debug-action-mask-jsonl", type=str, default=None, help="Path to JSONL log file for action mask debugging")
    parser.add_argument("--invalid-action-penalty", type=float, default=0.0, help="Reward penalty per invalid action head (e.g., 0.1)")

    args = parser.parse_args()

    main(args)
