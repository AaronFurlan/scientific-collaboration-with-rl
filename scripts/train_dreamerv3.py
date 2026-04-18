"""
train_dreamerv3.py

Spezialisiertes Trainingsskript für DreamerV3 in RLlib.
DreamerV3 nutzt den neuen RLlib API-Stack (RLModule, EnvRunnerV2) und hat 
spezifische Anforderungen an die Konfiguration und die Umgebung.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
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

# Add the project root to sys.path so we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ray import tune
from typing import Any, Callable, Dict, Optional, List
from ray.rllib.algorithms.dreamerv3 import DreamerV3Config
from lightning.pytorch import seed_everything

from src.agent_policies import (
    create_mixed_policy_population,
    create_per_group_policy_population,
    do_nothing_policy,
    get_policy_function,
)

from src.env.peer_group_environment import PeerGroupEnvironment
from src.rllib_single_agent_wrapper import RLLibSingleAgentWrapper
from src.dreamerv3_wrapper import DreamerV3SingleAgentWrapper
from src.callbacks.papers_metrics_callback import PapersMetricsCallback

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

POLICY_CONFIGS = {
    "Balanced": {"careerist": 0.33, "orthodox_scientist": 0.33, "mass_producer": 0.34},
    "CareeristHeavy": {"careerist": 0.7, "orthodox_scientist": 0.15, "mass_producer": 0.15},
    "OrthodoxHeavy": {"careerist": 0.15, "orthodox_scientist": 0.7, "mass_producer": 0.15},
    "MassProducerHeavy": {"careerist": 0.15, "orthodox_scientist": 0.15, "mass_producer": 0.7},
}

def wandb_sanitize(metrics: Dict[str, Any]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, (dict, list, tuple, set)): continue
        if hasattr(v, "item"): v = v.item()
        if isinstance(v, (int, float, bool, np.number)):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): continue
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
    debug_effort: bool = False,
) -> Callable[[Optional[Dict[str, Any]]], Any]:

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
            debug_effort=debug_effort
        )
    return _env_creator

def main(**kwargs):
    seed = kwargs["seed"]
    seed_everything(seed)
    ray.init(ignore_reinit_error=True)

    env_name = "peer_group_env_dreamerv3"
    env_creator = make_env_creator(
        n_agents=kwargs["n_agents"],
        start_agents=kwargs["start_agents"],
        max_steps=kwargs["max_steps"],
        max_rewardless_steps=kwargs["max_rewardless_steps"],
        n_groups=kwargs["n_groups"],
        max_peer_group_size=kwargs["max_peer_group_size"],
        n_projects_per_step=kwargs["n_projects_per_step"],
        max_projects_per_agent=kwargs["max_projects_per_agent"],
        max_agent_age=kwargs["max_agent_age"],
        acceptance_threshold=kwargs["acceptance_threshold"],
        reward_function=kwargs["reward_function"],
        seed=seed,
        policy_distribution=POLICY_CONFIGS[kwargs["policy_config"]],
        group_policy_homogenous=kwargs["group_policy_homogenous"],
        prestige_threshold=kwargs["prestige_threshold"],
        novelty_threshold=kwargs["novelty_threshold"],
        effort_threshold=kwargs["effort_threshold"],
        controlled_agent_id=kwargs["controlled_agent_id"],
        debug_effort=kwargs["debug_effort"],
    )
    tune.register_env(env_name, env_creator)

    config = (
        DreamerV3Config()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .framework("torch")
        .training(
            model_size=kwargs.get("model_size", "XS"),
            training_ratio=kwargs.get("training_ratio", 512),
            batch_size_B=kwargs.get("batch_size_B", 16),
            batch_length_T=kwargs.get("batch_length_T", 64),
            gamma=kwargs["gamma"],
            world_model_lr=kwargs["lr"],
            actor_lr=kwargs["lr"],
            critic_lr=kwargs["lr"],
        )
        .environment(
            env=env_name,
            env_config={
                "evaluation": False,
            },
        )
        .env_runners(
            num_env_runners=kwargs["num_workers"],
            num_envs_per_env_runner=kwargs["num_envs_per_worker"],
            create_env_on_local_worker=True,
            sample_timeout_s=3600,
        )
        .callbacks(PapersMetricsCallback)
    )

    if kwargs["wandb_mode"] != "disabled":
        wandb.init(
            project=kwargs["wandb_project"],
            entity=kwargs["wandb_entity"],
            group=kwargs["wandb_group"],
            mode=kwargs["wandb_mode"],
            config=kwargs,
        )

    algo_instance = config.build_algo()
    
    for i in range(kwargs["iterations"]):
        result = algo_instance.train()
        print(f"Iter {i}: reward={result.get('env_runners', {}).get('episode_reward_mean', 'N/A')}")
        if kwargs["wandb_mode"] != "disabled":
            wandb.log(wandb_sanitize(result))

    algo_instance.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--policy-config", type=str, default="Balanced", choices=POLICY_CONFIGS.keys())
    parser.add_argument("--group-policy-homogenous", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-agents", type=int, default=400)
    parser.add_argument("--start-agents", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--max-rewardless-steps", type=int, default=50)
    parser.add_argument("--n-groups", type=int, default=10)
    parser.add_argument("--max-peer-group-size", type=int, default=10)
    parser.add_argument("--n-projects-per-step", type=int, default=1)
    parser.add_argument("--max-projects-per-agent", type=int, default=8)
    parser.add_argument("--max-agent-age", type=int, default=750)
    parser.add_argument("--acceptance-threshold", type=float, default=0.44)
    parser.add_argument("--reward-function", type=str, default="by_effort", choices=["multiply", "evenly", "by_effort"])
    parser.add_argument("--prestige-threshold", type=float, default=0.29)
    parser.add_argument("--novelty-threshold", type=float, default=0.4)
    parser.add_argument("--effort-threshold", type=int, default=35)
    parser.add_argument("--controlled-agent-id", type=str, default="agent_0")
    parser.add_argument("--wandb-project", type=str, default="RL in the Game of Science")
    parser.add_argument("--wandb-entity", type=str, default="rl_in_the_game_of_science")
    parser.add_argument("--wandb-group", type=str, default="DreamerV3")
    parser.add_argument("--wandb-mode", type=str, default="disabled")
    parser.add_argument("--debug-effort", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-envs-per-worker", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model-size", type=str, default="XS")
    parser.add_argument("--training-ratio", type=int, default=1024)
    parser.add_argument("--batch-size-B", type=int, default=16)
    parser.add_argument("--batch-length-T", type=int, default=32)

    args = parser.parse_args()
    main(**vars(args))
