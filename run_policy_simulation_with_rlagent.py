"""
run_policy_simulation_with_rlagent.py

Runs the PeerGroupEnvironment simulation with a trained RLlib PPO checkpoint
controlling agent_0, while all other agents follow fixed heuristic policies.

Closely mirrors run_policy_simulation.py but replaces agent_0's heuristic
policy with the trained RL policy loaded from a checkpoint (default: models/).

Architecture:
    - The RLLibSingleAgentWrapper is used as a helper for observation
      flattening, macro-action decoding, action-mask repair, and top-k
      collaboration — exactly the same utilities used during training.
    - The restored RLlib algorithm's RLModule produces action logits via
      forward_inference(), which we decode into environment actions.
    - We maintain a reference to the *unwrapped* environment for full
      multi-agent logging visibility (observations, actions, projects, stats).

Usage:
    python run_policy_simulation_with_rlagent.py
    python run_policy_simulation_with_rlagent.py --checkpoint models/ --seed 42
    python run_policy_simulation_with_rlagent.py --policy-config Balanced --reward-function by_effort
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import ray
from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm
from lightning.pytorch import seed_everything

from agent_policies import (
    create_mixed_policy_population,
    create_per_group_policy_population,
    do_nothing_policy,
    get_policy_function,
)
from env.peer_group_environment import PeerGroupEnvironment
from rllib_single_agent_wrapper import RLLibSingleAgentWrapper
from log_simulation import SimLog
from stats_tracker import SimulationStats


# ---------------------------------------------------------------------------
# Policy configs — shared with run_policy_simulation.py / train_ppo_rllib.py
# ---------------------------------------------------------------------------
POLICY_CONFIGS: Dict[str, Dict[str, float]] = {
    "All Careerist": {"careerist": 1.0, "orthodox_scientist": 0.0, "mass_producer": 0.0},
    "All Orthodox": {"careerist": 0.0, "orthodox_scientist": 1.0, "mass_producer": 0.0},
    "All Mass Producer": {"careerist": 0.0, "orthodox_scientist": 0.0, "mass_producer": 1.0},
    "Balanced": {"careerist": 1 / 3, "orthodox_scientist": 1 / 3, "mass_producer": 1 / 3},
    "Careerist Heavy": {"careerist": 0.5, "orthodox_scientist": 0.5, "mass_producer": 0.0},
    "Orthodox Heavy": {"careerist": 0.5, "orthodox_scientist": 0.0, "mass_producer": 0.5},
    "Mass Producer Heavy": {"careerist": 0.5, "orthodox_scientist": 0.0, "mass_producer": 0.5},
}


# ---------------------------------------------------------------------------
# Centralized config — single source of truth for all parameters
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    """All parameters for a single evaluation run.

    Parameters marked [must-match-training] must be identical to the values
    used during PPO training; a mismatch will silently produce wrong results.
    """

    # Checkpoint
    checkpoint_path: str = "models"

    # Environment  [must-match-training]
    n_agents: int = 64
    start_agents: int = 60
    max_steps: int = 500
    max_rewardless_steps: int = 500
    n_groups: int = 8
    max_peer_group_size: int = 8       # ← macro-action size depends on this
    n_projects_per_step: int = 1
    max_projects_per_agent: int = 6
    max_agent_age: int = 750
    acceptance_threshold: float = 0.5
    reward_function: str = "by_effort"

    # Heuristic thresholds  [must-match-training]
    prestige_threshold: float = 0.2
    novelty_threshold: float = 0.8
    effort_threshold: int = 22

    # Population  [must-match-training]
    policy_config_name: str = "Balanced"
    group_policy_homogenous: bool = False

    # Wrapper  [must-match-training]
    topk_collab: int = 3
    topk_apply_to_all_agents: bool = True

    # RL agent
    controlled_agent_id: str = "agent_0"
    deterministic: bool = True

    # Reproducibility
    seed: int = 42

    # Output
    output_file_prefix: str = "rl_agent_sim"

    @property
    def policy_distribution(self) -> Dict[str, float]:
        return POLICY_CONFIGS[self.policy_config_name]

    def print_summary(self) -> None:
        """Print a compact config summary at startup."""
        print(f"\n{'='*60}")
        print("EVALUATION CONFIG")
        print(f"{'='*60}")
        print(f"  checkpoint:          {self.checkpoint_path}")
        print(f"  seed:                {self.seed}")
        print(f"  deterministic:       {self.deterministic}")
        print(f"  controlled_agent:    {self.controlled_agent_id}")
        print(f"  policy_config:       {self.policy_config_name}")
        print(f"  reward_function:     {self.reward_function}")
        print(f"  env: n_agents={self.n_agents}, start={self.start_agents}, "
              f"steps={self.max_steps}, groups={self.n_groups}, "
              f"peer_size={self.max_peer_group_size}")
        print(f"  thresholds: prestige={self.prestige_threshold}, "
              f"novelty={self.novelty_threshold}, effort={self.effort_threshold}")
        print(f"  topk_collab={self.topk_collab}, "
              f"topk_all_agents={self.topk_apply_to_all_agents}")
        print(f"  output_prefix:       {self.output_file_prefix}")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Reusable builder helpers — eliminate duplication
# ---------------------------------------------------------------------------

def build_env(cfg: EvalConfig) -> PeerGroupEnvironment:
    """Create a raw PeerGroupEnvironment from config."""
    return PeerGroupEnvironment(
        start_agents=cfg.start_agents,
        max_agents=cfg.n_agents,
        max_steps=cfg.max_steps,
        n_groups=cfg.n_groups,
        max_peer_group_size=cfg.max_peer_group_size,
        n_projects_per_step=cfg.n_projects_per_step,
        max_projects_per_agent=cfg.max_projects_per_agent,
        max_agent_age=cfg.max_agent_age,
        max_rewardless_steps=cfg.max_rewardless_steps,
        acceptance_threshold=cfg.acceptance_threshold,
        reward_mode=cfg.reward_function,
    )


def build_heuristic_population(cfg: EvalConfig) -> List[str]:
    """Assign archetype names to all agents (same logic as training)."""
    if cfg.group_policy_homogenous:
        return create_per_group_policy_population(
            cfg.n_agents, cfg.policy_distribution
        )
    return create_mixed_policy_population(
        cfg.n_agents, cfg.policy_distribution, seed=cfg.seed
    )


def make_policy_callable(
    policy_name: Optional[str],
    cfg: EvalConfig,
) -> Callable:
    """Return a callable(nested_obs) -> action_dict for a given archetype name.

    Each callable expects the wrapper-style nested observation:
        {"observation": <dict>, "action_mask": <dict>}
    and returns a valid env action dict.
    """
    if policy_name is None:
        return lambda nested_obs: do_nothing_policy(
            nested_obs["observation"], nested_obs["action_mask"]
        )

    fn = get_policy_function(policy_name)

    # Each heuristic has its own threshold kwarg
    threshold_map = {
        "careerist": cfg.prestige_threshold,
        "orthodox_scientist": cfg.novelty_threshold,
        "mass_producer": cfg.effort_threshold,
    }
    threshold = threshold_map.get(policy_name)

    if threshold is not None:
        return lambda nested_obs, _fn=fn, _t=threshold: _fn(
            nested_obs["observation"], nested_obs["action_mask"], _t
        )
    # Fallback (e.g. maximally_collaborative or unknown)
    return lambda nested_obs, _fn=fn: _fn(
        nested_obs["observation"], nested_obs["action_mask"]
    )


def build_other_policies(
    env: PeerGroupEnvironment,
    agent_policies: List[str],
    cfg: EvalConfig,
) -> Dict[str, Callable]:
    """Build {agent_id: callable} for every agent except the controlled one."""
    other_policies: Dict[str, Callable] = {}
    for agent_id in env.possible_agents:
        if agent_id == cfg.controlled_agent_id:
            continue
        idx = env.agent_to_id[agent_id]
        other_policies[agent_id] = make_policy_callable(agent_policies[idx], cfg)
    return other_policies


def build_eval_wrapper(
    env: PeerGroupEnvironment,
    other_policies: Dict[str, Callable],
    cfg: EvalConfig,
) -> RLLibSingleAgentWrapper:
    """Wrap the raw env for single-agent RLlib interaction.

    This wrapper uses the same class and identical parameters as during
    training. This guarantees:
      - observation vector layout matches what the RL module expects
      - macro-action decoding is identical
      - action mask repair + top-k collaboration behave the same
    """
    return RLLibSingleAgentWrapper(
        env,
        controlled_agent=cfg.controlled_agent_id,
        other_policies=other_policies,
        force_episode_horizon=cfg.max_steps,
        topk_collab=cfg.topk_collab,
        topk_apply_to_all_agents=cfg.topk_apply_to_all_agents,
    )


def make_env_creator_from_config(cfg: EvalConfig) -> Callable:
    """Return an RLlib-compatible env creator closure.

    Required by Algorithm.from_checkpoint() so RLlib can reconstruct the
    environment when restoring the checkpoint.
    """
    def _env_creator(env_config=None):
        env = build_env(cfg)
        agent_policies = build_heuristic_population(cfg)
        other_policies = build_other_policies(env, agent_policies, cfg)
        return build_eval_wrapper(env, other_policies, cfg)
    return _env_creator


# ---------------------------------------------------------------------------
# RL agent inference helper
# ---------------------------------------------------------------------------

def compute_rl_action(
    rl_module: torch.nn.Module,
    obs_vec: np.ndarray,
    deterministic: bool,
) -> int:
    """Compute a discrete macro-action ID from a flattened observation vector.

    Uses the RLModule's forward_inference() — the recommended new-API-stack
    approach for checkpoint-restored algorithms. This is safer than
    algo.compute_single_action() which is deprecated on the new API stack
    and fails with 'SingleAgentEnvRunner has no attribute get_policy'.

    Args:
        rl_module: The RLModule obtained via algo.get_module().
        obs_vec: 1-D float32 numpy array (flattened observation + action mask).
        deterministic: If True, take the argmax (greedy). Otherwise sample.

    Returns:
        Integer macro-action ID that the wrapper can decode.
    """
    obs_tensor = torch.from_numpy(obs_vec).unsqueeze(0).float()
    with torch.no_grad():
        fwd_out = rl_module.forward_inference({"obs": obs_tensor})
    logits = fwd_out["action_dist_inputs"]

    if deterministic:
        return int(torch.argmax(logits, dim=-1).item())
    else:
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1).item())


# ---------------------------------------------------------------------------
# RL agent status tracking
# ---------------------------------------------------------------------------

@dataclass
class RLAgentStatus:
    """Mutable tracker for the controlled RL agent's per-step state."""

    agent_id: str
    agent_idx: int
    terminated_step: Optional[int] = None
    total_reward: float = 0.0

    # Snapshot fields (updated each step before env.step())
    is_active: bool = True
    age: int = 0
    rewardless_steps: int = 0
    n_active_projects: int = 0
    completed_projects: int = 0
    successful_projects: int = 0
    step_reward: float = 0.0

    def snapshot(self, env: PeerGroupEnvironment) -> None:
        """Capture agent state from the environment BEFORE env.step()."""
        idx = self.agent_idx
        self.is_active = bool(env.active_agents[idx])
        self.age = int(env.agent_steps[idx])
        self.rewardless_steps = int(env.rewardless_steps[idx])
        self.n_active_projects = len(env._get_active_projects(idx))
        self.completed_projects = int(env.agent_completed_projects[idx])
        self.successful_projects = len(env.agent_successful_projects[idx])

    def record_step_reward(self, reward: float) -> None:
        self.step_reward = reward
        self.total_reward += reward

    def record_termination(self, step: int) -> None:
        """Mark the agent as terminated. Only records the first termination."""
        if self.terminated_step is None:
            self.terminated_step = step

    @property
    def status_label(self) -> str:
        if self.is_active:
            return "ACTIVE"
        return f"TERMINATED(step={self.terminated_step})"

    def format_log_line(self, action_dict: Optional[dict] = None) -> str:
        """One-line status string for periodic logging."""
        parts = [
            f"RL {self.agent_id}: {self.status_label}",
            f"age={self.age}",
            f"rewardless={self.rewardless_steps}",
            f"projects(active={self.n_active_projects}, "
            f"done={self.completed_projects}, "
            f"published={self.successful_projects})",
            f"reward(step={self.step_reward:.4f}, total={self.total_reward:.4f})",
        ]
        if self.is_active and action_dict is not None:
            n_collab = int(np.sum(action_dict.get("collaborate_with", [])))
            parts.append(
                f"action(proj={action_dict['choose_project']}, "
                f"effort={action_dict['put_effort']}, collab={n_collab})"
            )
        return " | ".join(parts)

    def format_termination_banner(self) -> str:
        return (
            f"\n{'!'*60}\n"
            f"  RL AGENT ({self.agent_id}) TERMINATED at step {self.terminated_step}\n"
            f"  Age: {self.age} | Rewardless steps: {self.rewardless_steps}\n"
            f"  Completed projects: {self.completed_projects} | "
            f"Successful: {self.successful_projects}\n"
            f"  Total reward at termination: {self.total_reward:.4f}\n"
            f"{'!'*60}\n"
        )

    def final_summary(self, env: PeerGroupEnvironment) -> Dict:
        """Return a dict of final RL agent metrics for the results JSON."""
        idx = self.agent_idx
        return {
            "rl_agent_total_reward": float(self.total_reward),
            "rl_agent_terminated_step": self.terminated_step,
            "rl_agent_completed_projects": int(env.agent_completed_projects[idx]),
            "rl_agent_successful_projects": len(env.agent_successful_projects[idx]),
            "rl_agent_h_index": int(env.agent_h_indexes[idx]),
            "rl_agent_age": int(env.agent_steps[idx]),
        }


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------

def run_simulation_with_rl_agent(cfg: EvalConfig) -> dict:
    """Run one evaluation episode with the trained RL agent.

    Args:
        cfg: Fully specified evaluation config.

    Returns:
        Results dict (also saved to log/<prefix>_summary.json).
    """
    cfg.print_summary()

    # Seed all RNGs (random, numpy, torch, cuda) for reproducibility
    seed_everything(cfg.seed, workers=True)

    # Reproducibility: force deterministic PyTorch operations where possible
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---- 1) Init Ray & register env for checkpoint restoration ----
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        logging_level="WARNING",
    )

    env_name = "peer_group_single_agent_fixed_population"
    tune.register_env(env_name, make_env_creator_from_config(cfg))

    # ---- 2) Restore trained algorithm from checkpoint ----
    checkpoint_path = os.path.abspath(cfg.checkpoint_path)
    print(f"Restoring algorithm from checkpoint: {checkpoint_path}")
    algo = Algorithm.from_checkpoint(checkpoint_path)
    print("Algorithm restored successfully.")

    # Get the RLModule for inference.
    # Why forward_inference() instead of algo.compute_single_action()?
    #   - compute_single_action() is deprecated on the new RLlib API stack
    #     and raises AttributeError ('SingleAgentEnvRunner' has no 'get_policy').
    #   - forward_inference() is the officially recommended replacement:
    #     it runs the neural network in eval mode without connectors/exploration
    #     overhead, and returns raw logits we can decode ourselves.
    #   - This also gives us explicit control over deterministic vs stochastic
    #     action selection (argmax vs sampling from the logit distribution).
    rl_module = algo.get_module()
    rl_module.eval()

    # ---- 3) Build evaluation environment & population ----
    env = build_env(cfg)
    agent_policies = build_heuristic_population(cfg)

    dist_counts = dict(zip(*np.unique(agent_policies, return_counts=True)))
    print(f"Agent policy distribution: {dist_counts}")
    print(f"Controlled agent: {cfg.controlled_agent_id} -> RL Policy (PPO)")

    # ---- 4) Build helper wrapper for obs flattening & action decoding ----
    # The wrapper uses the same RLLibSingleAgentWrapper class and identical
    # parameters as during training. This guarantees:
    #   - observation vector layout matches what the RL module expects
    #   - macro-action decoding is identical
    #   - action mask repair + top-k collaboration behave the same
    other_policies = build_other_policies(env, agent_policies, cfg)
    helper_wrapper = build_eval_wrapper(env, other_policies, cfg)

    # ---- 5) Set up logging (same as run_policy_simulation.py) ----
    stats = SimulationStats()
    log = SimLog(
        "log",
        f"{cfg.output_file_prefix}_actions.jsonl",
        f"{cfg.output_file_prefix}_observations.jsonl",
        f"{cfg.output_file_prefix}_projects.json",
    )
    log.start()

    # ---- 6) Reset environment and run simulation ----
    observations, infos = env.reset(seed=cfg.seed)
    # Sync wrapper internal state so _flatten_to_vector template is built
    helper_wrapper._last_observations = observations

    rl_status = RLAgentStatus(
        agent_id=cfg.controlled_agent_id,
        agent_idx=env.agent_to_id[cfg.controlled_agent_id],
    )

    for step in range(cfg.max_steps):
        # ---- Snapshot RL agent state BEFORE stepping ----
        rl_status.snapshot(env)

        actions = {}
        for agent in env.agents:
            agent_idx = env.agent_to_id[agent]

            if agent == cfg.controlled_agent_id and env.active_agents[agent_idx] == 1:
                # ---- RL agent: flatten → infer → decode → mask-repair ----
                nested_obs = observations[agent]
                obs_vec = helper_wrapper._flatten_to_vector(nested_obs)
                action_id = compute_rl_action(rl_module, obs_vec, cfg.deterministic)
                decoded = helper_wrapper._decode_action(action_id)
                decoded = helper_wrapper._apply_action_mask(
                    decoded, nested_obs, agent_id=agent
                )
                actions[agent] = decoded
            else:
                # ---- Heuristic / inactive agent ----
                nested_obs = observations[agent]
                if env.active_agents[agent_idx] == 0:
                    action = do_nothing_policy(
                        nested_obs["observation"], nested_obs["action_mask"]
                    )
                else:
                    policy_fn = other_policies.get(agent)
                    if policy_fn is not None:
                        action = policy_fn(nested_obs)
                    else:
                        action = do_nothing_policy(
                            nested_obs["observation"], nested_obs["action_mask"]
                        )
                # Apply the same mask repair + top-k as during training
                action = helper_wrapper._apply_action_mask(
                    action, nested_obs, agent_id=agent
                )
                actions[agent] = action

        # ---- Step the environment ----
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # ---- Track RL agent reward ----
        rl_reward = rewards.get(cfg.controlled_agent_id, 0.0)
        rl_status.record_step_reward(rl_reward)

        # ---- Detect RL agent termination ----
        if terminations.get(cfg.controlled_agent_id, False):
            if rl_status.terminated_step is None:
                rl_status.record_termination(step)
                print(rl_status.format_termination_banner())

        # ---- Log observations & actions ----
        log.log_observation({
            a: obs if env.active_agents[env.agent_to_id[a]] == 1 else None
            for a, obs in observations.items()
        })
        log.log_action({
            a: (
                act | {
                    "archetype": (
                        "rl_agent"
                        if a == cfg.controlled_agent_id
                        else agent_policies[env.agent_to_id[a]]
                    )
                }
                if env.active_agents[env.agent_to_id[a]] == 1
                else None
            )
            for a, act in actions.items()
        })

        # ---- Update stats ----
        stats.update(env, observations, rewards, terminations, truncations)

        # ---- Periodic progress ----
        if step % 10 == 0:
            rl_action = actions.get(cfg.controlled_agent_id)
            print(
                f"Step {step:3d}: {stats.summary_line()}\n"
                f"          {rl_status.format_log_line(rl_action if rl_status.is_active else None)}"
            )

        # ---- Check if all agents are done ----
        if all(terminations.values()):
            print(f"Simulation ended at step {step}")
            break

    # ---- 7) Save results ----
    env.area.save(f"log/{cfg.output_file_prefix}_area.pickle")
    log.log_projects(env.projects.values())

    results = {
        "config": {
            k: v for k, v in asdict(cfg).items()
            if k != "checkpoint_path"  # don't persist machine-specific abs path
        },
        "config_checkpoint": cfg.checkpoint_path,
        "final_stats": stats.to_dict(),
        "agent_policies": agent_policies,
        "policy_distribution": cfg.policy_distribution,
        "controlled_agent": cfg.controlled_agent_id,
        **rl_status.final_summary(env),
    }

    summary_path = f"log/{cfg.output_file_prefix}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    # ---- Final report ----
    final = rl_status.final_summary(env)
    print(f"\n{'='*60}")
    print(f"SIMULATION RESULTS (RL Agent: {cfg.controlled_agent_id})")
    print(f"{'='*60}")
    print(f"Total Steps: {stats.total_steps}")
    print(f"Finished Projects (all): {stats.finished_projects_count}")
    print(f"Successful Projects (all): {stats.successful_projects_count}")
    print(
        f"Success Rate (all): "
        f"{stats.successful_projects_count / max(stats.finished_projects_count, 1):.3f}"
    )
    print(f"Total Rewards (all agents): {stats.total_rewards_distributed:.2f}")
    print(f"\n--- RL Agent ({cfg.controlled_agent_id}) ---")
    status_str = (
        f"TERMINATED at step {rl_status.terminated_step}"
        if rl_status.terminated_step is not None
        else "ACTIVE (survived full episode)"
    )
    print(f"Status: {status_str}")
    print(f"Total Reward: {final['rl_agent_total_reward']:.4f}")
    print(f"Completed Projects: {final['rl_agent_completed_projects']}")
    print(f"Successfully Published: {final['rl_agent_successful_projects']}")
    print(f"H-Index: {final['rl_agent_h_index']}")
    print(f"Agent Age (steps active): {final['rl_agent_age']}")
    print(f"{'='*60}")

    # ---- 8) Cleanup ----
    algo.stop()
    ray.shutdown()

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> EvalConfig:
    """Parse CLI arguments into an EvalConfig."""
    parser = argparse.ArgumentParser(
        description=(
            "Run simulation with a trained RL agent (agent_0) "
            "and heuristic policies for all others."
        )
    )

    # Checkpoint
    parser.add_argument(
        "--checkpoint", type=str, default="models",
        help="Path to the RLlib checkpoint directory (default: models/)",
    )

    # Policy config
    parser.add_argument(
        "--policy-config", type=str, default="Balanced",
        choices=list(POLICY_CONFIGS.keys()),
        help="Policy distribution for non-RL agents",
    )
    parser.add_argument(
        "--group-policy-homogenous", action="store_true",
        help="Assign same archetype per group (vs mixed)",
    )

    # Env knobs (defaults match train_ppo_rllib.py)
    parser.add_argument("--n-agents", type=int, default=64)
    parser.add_argument("--start-agents", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-rewardless-steps", type=int, default=500)
    parser.add_argument("--n-groups", type=int, default=8)
    parser.add_argument("--max-peer-group-size", type=int, default=8)
    parser.add_argument("--n-projects-per-step", type=int, default=1)
    parser.add_argument("--max-projects-per-agent", type=int, default=6)
    parser.add_argument("--max-agent-age", type=int, default=750)

    # Reward & thresholds
    parser.add_argument("--acceptance-threshold", type=float, default=0.5)
    parser.add_argument(
        "--reward-function", type=str, default="by_effort",
        choices=["multiply", "evenly", "by_effort"],
    )
    parser.add_argument("--prestige-threshold", type=float, default=0.2)
    parser.add_argument("--novelty-threshold", type=float, default=0.8)
    parser.add_argument("--effort-threshold", type=int, default=22)

    # Agent control
    parser.add_argument("--controlled-agent-id", type=str, default="agent_0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Use stochastic (exploratory) actions instead of greedy",
    )

    # Output
    parser.add_argument(
        "--output-prefix", type=str, default="rl_agent_sim",
        help="Prefix for log output files",
    )

    args = parser.parse_args()

    return EvalConfig(
        checkpoint_path=args.checkpoint,
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
        policy_config_name=args.policy_config,
        group_policy_homogenous=args.group_policy_homogenous,
        controlled_agent_id=args.controlled_agent_id,
        deterministic=not args.stochastic,
        seed=args.seed,
        output_file_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    config = parse_args()
    run_simulation_with_rl_agent(config)

