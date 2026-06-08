"""
Baseline evaluation with random agent controlling one agent.
Random agent samples from valid masked actions uniformly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Any
from datetime import datetime

import numpy as np
from lightning.pytorch import seed_everything

from src.agent_policies import (
    create_mixed_policy_population,
    create_per_group_policy_population,
    do_nothing_policy,
    get_policy_function,
)
from src.env.peer_group_environment import PeerGroupEnvironment
from scripts.log_simulation import SimLog
from src.stats_tracker import SimulationStats


POLICY_CONFIGS: Dict[str, Dict[str, float]] = {
    "Balanced": {"careerist": 1 / 3, "orthodox_scientist": 1 / 3, "mass_producer": 1 / 3}
}

@dataclass
class EvalConfig:
    """Configuration for random agent baseline evaluation."""

    # Environment
    n_agents: int = 400
    start_agents: int = 100
    max_steps: int = 600
    max_rewardless_steps: int = 50
    n_groups: int = 10
    max_peer_group_size: int = 40
    n_projects_per_step: int = 1
    max_projects_per_agent: int = 8
    max_agent_age: int = 750
    acceptance_threshold: float = 0.44
    reward_function: str = "by_effort"

    # Heuristic thresholds for other agents
    prestige_threshold: float = 0.29
    novelty_threshold: float = 0.4
    effort_threshold: int = 35

    # Population
    policy_config_name: str = "Balanced"
    group_policy_homogenous: bool = False

    # Controlled agent
    controlled_agent_id: str = "agent_0"

    # Reproducibility
    seed: int = 42

    # Output
    output_file_prefix: str = "random_agent_sim"
    output_dir: str = "test_results"

    # Debug
    debug_sim: bool = False

    @property
    def policy_distribution(self) -> Dict[str, float]:
        return POLICY_CONFIGS[self.policy_config_name]

    def copy_with(self, **kwargs) -> EvalConfig:
        """Create a copy with some fields updated."""
        from copy import deepcopy
        new_cfg = deepcopy(self)
        for k, v in kwargs.items():
            if not hasattr(new_cfg, k):
                raise AttributeError(f"EvalConfig has no attribute {k}")
            setattr(new_cfg, k, v)
        return new_cfg

    def print_summary(self) -> None:
        """Print configuration summary."""
        print(f"\n{'='*60}")
        print("RANDOM AGENT BASELINE EVALUATION")
        print(f"{'='*60}")
        print(f"  seed:                {self.seed}")
        print(f"  controlled_agent:    {self.controlled_agent_id} -> RANDOM POLICY")
        print(f"  policy_config:       {self.policy_config_name}")
        print(f"  reward_function:     {self.reward_function}")
        print(f"  env: n_agents={self.n_agents}, start={self.start_agents}, "
              f"steps={self.max_steps}, groups={self.n_groups}, "
              f"peer_size={self.max_peer_group_size}")
        print(f"  thresholds: prestige={self.prestige_threshold}, "
              f"novelty={self.novelty_threshold}, effort={self.effort_threshold}")
        print(f"  output_dir:          {self.output_dir}")
        print(f"  output_prefix:       {self.output_file_prefix}")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Agent status tracking
# ---------------------------------------------------------------------------
@dataclass
class RandomAgentStatus:
    """Track random agent's per-step state."""

    agent_id: str
    agent_idx: int
    terminated_step: Optional[int] = None
    total_reward: float = 0.0

    # Snapshot fields
    is_active: bool = True
    age: int = 0
    rewardless_steps: int = 0
    n_active_projects: int = 0
    completed_projects: int = 0
    successful_projects: int = 0
    step_reward: float = 0.0
    termination_reason: str = ""

    def snapshot(self, env: PeerGroupEnvironment) -> None:
        """Capture agent state from environment."""
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

    def record_termination(self, step: int, env: Optional[PeerGroupEnvironment] = None) -> None:
        """Mark agent as terminated."""
        if self.terminated_step is None:
            self.terminated_step = step
            if env is not None:
                idx = self.agent_idx
                rewardless_dist = self.rewardless_steps - env.max_rewardless_steps
                age_limit = env.agent_ages[idx]
                age_dist = self.age - age_limit

                reasons = []
                if rewardless_dist > -10:
                    reasons.append(f"REWARDLESS STEPS ({self.rewardless_steps}/{env.max_rewardless_steps})")
                if age_dist > -10:
                    reasons.append(f"AGE ({self.age}/{int(age_limit)})")

                if not reasons:
                    if rewardless_dist / env.max_rewardless_steps > age_dist / age_limit:
                        reasons.append(f"REWARDLESS STEPS (stochastic, {self.rewardless_steps}/{env.max_rewardless_steps})")
                    else:
                        reasons.append(f"AGE (stochastic, {self.age}/{int(age_limit)})")

                self.termination_reason = " & ".join(reasons)

    def final_summary(self, env: PeerGroupEnvironment) -> Dict:
        """Return final metrics."""
        idx = self.agent_idx
        return {
            "random_agent_total_reward": float(self.total_reward),
            "random_agent_terminated_step": self.terminated_step,
            "random_agent_completed_projects": int(env.agent_completed_projects[idx]),
            "random_agent_successful_projects": len(env.agent_successful_projects[idx]),
            "random_agent_h_index": int(env.agent_h_indexes[idx]),
            "random_agent_age": int(env.agent_steps[idx]),
        }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_env(cfg: EvalConfig) -> PeerGroupEnvironment:
    """Create environment from config."""
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
    """Assign archetype names to all agents."""
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
    """Return callable(nested_obs) -> action_dict for an archetype."""
    if policy_name is None:
        return lambda nested_obs: do_nothing_policy(
            nested_obs["observation"], nested_obs["action_mask"]
        )

    fn = get_policy_function(policy_name)

    # Map threshold kwargs
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
    # Fallback
    return lambda nested_obs, _fn=fn: _fn(
        nested_obs["observation"], nested_obs["action_mask"]
    )


def build_other_policies(
    env: PeerGroupEnvironment,
    agent_policies: List[str],
    cfg: EvalConfig,
) -> Dict[str, Callable]:
    """Build {agent_id: callable} for every agent except controlled one."""
    other_policies: Dict[str, Callable] = {}
    for agent_id in env.possible_agents:
        if agent_id == cfg.controlled_agent_id:
            continue
        idx = env.agent_to_id[agent_id]
        other_policies[agent_id] = make_policy_callable(agent_policies[idx], cfg)
    return other_policies


def apply_action_mask_repair(action: Dict[str, Any], action_mask: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Repair invalid actions using the action mask.

    This EXACTLY mirrors the behavior of RLLibSingleAgentWrapper._apply_action_mask()
    to ensure fair comparison between random baseline and RL agent.

    The RL agent samples from the full action space and then repairs invalid
    actions. The random agent should do the same.

    IMPORTANT: Invalid discrete actions (choose_project, put_effort) are set to 0,
    NOT to the first valid option. This matches the wrapper behavior.
    """
    if not isinstance(action_mask, dict):
        return action

    repaired = action.copy()

    # Repair choose_project (EXACT wrapper logic)
    cp_mask = np.asarray(action_mask.get("choose_project", []))
    if cp_mask.size > 0:
        cp = int(repaired.get("choose_project", 0))
        if cp < 0 or cp >= cp_mask.size or cp_mask[cp] <= 0:
            repaired["choose_project"] = 0  # ← Same as wrapper: always 0!

    # Repair put_effort (EXACT wrapper logic)
    pe_mask = np.asarray(action_mask.get("put_effort", []))
    if pe_mask.size > 0:
        pe = int(repaired.get("put_effort", 0))
        if pe < 0 or pe >= pe_mask.size or pe_mask[pe] <= 0:
            repaired["put_effort"] = 0  # ← Same as wrapper: always 0!

    # Repair collaborate_with (EXACT wrapper logic)
    c_mask = np.asarray(action_mask.get("collaborate_with", []))
    if c_mask.size > 0:
        c = np.asarray(repaired.get("collaborate_with", []), dtype=np.int8)
        # Ensure correct size
        if c.size != c_mask.size:
            c = np.zeros(c_mask.size, dtype=np.int8)

        # Apply mask: set invalid collaborations to 0
        allowed = (c_mask > 0)
        L = min(len(c), len(allowed))

        c_slice = c[:L]
        allowed_slice = allowed[:L]
        c_slice[~allowed_slice] = 0
        c[:L] = c_slice

        if len(c) > L:
            c[L:] = 0

        repaired["collaborate_with"] = c.astype(np.int8)

    return repaired


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_simulation_with_random_agent(cfg: EvalConfig) -> dict:
    """Run one evaluation episode with random agent as agent_0.

    Args:
        cfg: Evaluation config.

    Returns:
        Results dict (also saved to test_results/<prefix>_summary.json).
    """
    cfg.print_summary()

    # Seed all RNGs for reproducibility
    seed_everything(cfg.seed, workers=True)

    # Build environment
    env = build_env(cfg)
    agent_policies = build_heuristic_population(cfg)

    dist_counts = dict(zip(*np.unique(agent_policies, return_counts=True)))
    print(f"Agent policy distribution: {dist_counts}")
    print(f"Controlled agent: {cfg.controlled_agent_id} -> RANDOM POLICY")

    # Build other policies
    other_policies = build_other_policies(env, agent_policies, cfg)

    # Get random policy for agent_0
    random_policy_fn = get_policy_function("random")

    # Set up logging
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_output_dir = cfg.output_dir
    if not os.path.isabs(base_output_dir):
        base_output_dir = os.path.join(project_root, base_output_dir)

    run_dir_name = f"{cfg.output_file_prefix}_{timestamp}_s{cfg.seed}"
    output_dir = os.path.join(base_output_dir, run_dir_name)

    os.makedirs(output_dir, exist_ok=True)

    log = SimLog(
        output_dir,
        f"{cfg.output_file_prefix}_actions.jsonl",
        f"{cfg.output_file_prefix}_observations.jsonl",
        f"{cfg.output_file_prefix}_projects.json",
    )
    log.start()

    stats = SimulationStats()

    # Reset environment
    observations, infos = env.reset(seed=cfg.seed)

    random_status = RandomAgentStatus(
        agent_id=cfg.controlled_agent_id,
        agent_idx=env.agent_to_id[cfg.controlled_agent_id],
    )

    # Create a separate RNG for the random agent to ensure reproducibility
    random_rng = np.random.default_rng(cfg.seed)

    for step in range(cfg.max_steps):
        # Snapshot random agent state
        random_status.snapshot(env)

        actions = {}
        for agent in env.agents:
            agent_idx = env.agent_to_id[agent]
            nested_obs = observations[agent]

            if agent == cfg.controlled_agent_id and env.active_agents[agent_idx] == 1:
                # Random agent: sample uniformly from ALL actions (not just valid)
                action = random_policy_fn(
                    nested_obs["observation"],
                    nested_obs["action_mask"],
                    rng=random_rng
                )
                # Apply mask repair (just like RL agent in wrapper)
                action = apply_action_mask_repair(action, nested_obs["action_mask"])
                actions[agent] = action
            else:
                # Heuristic or inactive agent
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
                actions[agent] = action

        # Step environment
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Track random agent reward
        random_reward = rewards.get(cfg.controlled_agent_id, 0.0)
        random_status.record_step_reward(random_reward)

        # Detect random agent termination
        if terminations.get(cfg.controlled_agent_id, False):
            if random_status.terminated_step is None:
                random_status.record_termination(step, env=env)
                if cfg.debug_sim:
                    print(f"\nRandom agent terminated at step {step}")
                    print(f"  Reason: {random_status.termination_reason}")

        # Log observations & actions
        log.log_observation({
            a: obs if env.active_agents[env.agent_to_id[a]] == 1 else None
            for a, obs in observations.items()
        })
        log.log_action({
            a: (
                act | {
                    "archetype": (
                        "random"
                        if a == cfg.controlled_agent_id
                        else agent_policies[env.agent_to_id[a]]
                    )
                }
                if env.active_agents[env.agent_to_id[a]] == 1
                else None
            )
            for a, act in actions.items()
        })

        # Update stats
        stats.update(env, observations, rewards, terminations, truncations)

        # Periodic progress
        if step % 50 == 0 and cfg.debug_sim:
            n_active = int(np.sum(env.active_agents))
            print(
                f"Step {step:3d}: active agents: {n_active} | "
                f"Random {random_status.agent_id}: age={random_status.age}, "
                f"rewardless={random_status.rewardless_steps}, "
                f"projects={random_status.n_active_projects}, "
                f"reward={random_status.total_reward:.4f}"
            )

        # Check if all agents done
        if all(terminations.values()):
            if cfg.debug_sim:
                print(f"Simulation ended at step {step}")
            break

        # Safety: no active agents left
        if int(np.sum(env.active_agents)) == 0:
            if cfg.debug_sim:
                print(f"Simulation ended at step {step}: no active agents")
            break

    # Save results
    env.area.save(os.path.join(output_dir, f"{cfg.output_file_prefix}_area.pickle"))
    log.log_projects(env.projects.values())

    results = {
        "config": {
            k: v for k, v in asdict(cfg).items()
        },
        "final_stats": stats.to_dict(),
        "agent_policies": agent_policies,
        "policy_distribution": cfg.policy_distribution,
        "controlled_agent": cfg.controlled_agent_id,
        **random_status.final_summary(env),
    }

    summary_path = os.path.join(output_dir, f"{cfg.output_file_prefix}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    # Final report
    final = random_status.final_summary(env)
    print(f"\n{'='*60}")
    print(f"SIMULATION RESULTS (Random Agent: {cfg.controlled_agent_id})")
    print(f"{'='*60}")
    print(f"Total Steps: {stats.total_steps}")
    print(f"Finished Projects (all): {stats.finished_projects_count}")
    print(f"Successful Projects (all): {stats.successful_projects_count}")
    print(
        f"Success Rate (all): "
        f"{stats.successful_projects_count / max(stats.finished_projects_count, 1):.3f}"
    )
    print(f"Total Rewards (all): {stats.total_rewards_distributed:.2f}")
    print(f"\n--- Random Agent ({cfg.controlled_agent_id}) ---")
    status_str = (
        f"TERMINATED at step {random_status.terminated_step}"
        if random_status.terminated_step is not None
        else "ACTIVE (survived full episode)"
    )
    print(f"Status: {status_str}")
    print(f"Total Reward: {final['random_agent_total_reward']:.4f}")
    print(f"Completed Projects: {final['random_agent_completed_projects']}")
    print(f"Successfully Published: {final['random_agent_successful_projects']}")
    print(f"H-Index: {final['random_agent_h_index']}")
    print(f"Agent Age (steps): {final['random_agent_age']}")
    print(f"{'='*60}\n")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> tuple[EvalConfig, int, bool]:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run random agent baseline evaluation"
    )

    # Policy config (fixed to Balanced - no CLI arguments needed)

    # Env parameters
    parser.add_argument("--n-agents", type=int, default=400)
    parser.add_argument("--start-agents", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--max-rewardless-steps", type=int, default=50)
    parser.add_argument("--n-groups", type=int, default=10)
    parser.add_argument("--max-peer-group-size", type=int, default=40)
    parser.add_argument("--n-projects-per-step", type=int, default=1)
    parser.add_argument("--max-projects-per-agent", type=int, default=8)
    parser.add_argument("--max-agent-age", type=int, default=750)

    # Reward & thresholds
    parser.add_argument("--acceptance-threshold", type=float, default=0.44)
    parser.add_argument(
        "--reward-function", type=str, default="by_effort",
        choices=["multiply", "evenly", "by_effort"],
    )
    parser.add_argument("--prestige-threshold", type=float, default=0.29)
    parser.add_argument("--novelty-threshold", type=float, default=0.4)
    parser.add_argument("--effort-threshold", type=int, default=35)

    parser.add_argument("--controlled-agent-id", type=str, default="agent_0")
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--output-dir", type=str, default="test_results")
    parser.add_argument("--output-prefix", type=str, default="random_baseline")
    parser.add_argument("--debug-sim", action="store_true")

    # Automation
    parser.add_argument(
        "--num-seeds", type=int, default=1,
        help="Number of consecutive seeds to evaluate",
    )
    parser.add_argument(
        "--all-rewards", action="store_true",
        help="Evaluate all reward functions",
    )

    args = parser.parse_args()

    return EvalConfig(
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
        policy_config_name="Balanced",  # Fixed to Balanced
        group_policy_homogenous=False,  # Fixed to mixed distribution
        controlled_agent_id=args.controlled_agent_id,
        seed=args.seed,
        output_file_prefix=args.output_prefix,
        output_dir=args.output_dir,
        debug_sim=args.debug_sim,
    ), args.num_seeds, args.all_rewards


if __name__ == "__main__":
    base_config, num_seeds, all_rewards = parse_args()

    # Prepare reward functions
    if all_rewards:
        reward_functions = ["multiply", "evenly", "by_effort"]
    else:
        reward_functions = [base_config.reward_function]

    start_seed = base_config.seed

    for reward_fn in reward_functions:
        print(f"\n{'='*60}")
        print(f"STARTING EVALUATION FOR REWARD FUNCTION: {reward_fn}")
        print(f"{'='*60}\n")

        for i in range(num_seeds):
            current_seed = start_seed + i

            config = base_config.copy_with(
                reward_function=reward_fn,
                seed=current_seed,
                output_file_prefix=f"random_{reward_fn}_s{current_seed}"
            )

            print(f"\n--- Run {i+1}/{num_seeds} | Seed: {current_seed} | Reward: {reward_fn} ---")
            run_simulation_with_random_agent(config)

    print("\nRandom agent baseline evaluation completed.")
