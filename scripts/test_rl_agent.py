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
    python run_policy_simulation_with_rlagent.py --checkpoint checkpoints/ --seed 42
    python run_policy_simulation_with_rlagent.py --policy-config Balanced --reward-function by_effort
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Add the project root to sys.path so we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Union, Any

import numpy as np
import torch
import ray
from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm
from lightning.pytorch import seed_everything

from src.agent_policies import (
    create_mixed_policy_population,
    create_per_group_policy_population,
    do_nothing_policy,
    get_policy_function,
)
from src.env.peer_group_environment import PeerGroupEnvironment
from src.rllib_single_agent_wrapper import RLLibSingleAgentWrapper
from scripts.log_simulation import SimLog
from src.stats_tracker import SimulationStats


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
    checkpoint_path: str = "checkpoints"

    # Environment  [must-match-training]
    # We use None as default to force sync from checkpoint if not provided via CLI
    n_agents: Optional[int] = 80
    start_agents: Optional[int] = 20
    max_steps: Optional[int] = 600
    max_rewardless_steps: Optional[int] = 100
    n_groups: Optional[int] = 4
    max_peer_group_size: Optional[int] = 60       # ← macro-action size depends on this
    n_projects_per_step: Optional[int] = 1
    max_projects_per_agent: Optional[int] = 6
    max_agent_age: Optional[int] = 1000
    acceptance_threshold: Optional[float] = 0.5
    reward_function: Optional[str] = "multiply"

    # Heuristic thresholds  [must-match-training]
    prestige_threshold: float = 0.2
    novelty_threshold: float = 0.8
    effort_threshold: int = 22

    # Population  [must-match-training]
    policy_config_name: str = "Balanced"
    group_policy_homogenous: bool = False

    # Wrapper  [must-match-training]
    topk_collab: Optional[int] = None
    topk_apply_to_all_agents: bool = True
    
    # Compatibility
    expected_obs_size: Optional[int] = None

    # RL agent
    controlled_agent_id: str = "agent_0"
    deterministic: bool = True
    debug_effort: bool = False  # Set to True to see detailed action decoding/repair logs
    debug_rl: bool = False      # Set to True to see detailed RL observations/actions during simulation
    debug_sim: bool = False     # Set to True to see general simulation progress and status banners
    debug_freq: int = 50        # Debug log frequency in steps

    # Reproducibility
    seed: int = 42

    # Output
    output_file_prefix: str = "rl_agent_sim"

    @property
    def policy_distribution(self) -> Dict[str, float]:
        return POLICY_CONFIGS[self.policy_config_name]

    def copy_with(self, **kwargs) -> EvalConfig:
        """Create a copy of this config with some fields updated."""
        from copy import deepcopy
        new_cfg = deepcopy(self)
        for k, v in kwargs.items():
            if not hasattr(new_cfg, k):
                raise AttributeError(f"EvalConfig has no attribute {k}")
            setattr(new_cfg, k, v)
        return new_cfg

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
        print(f"  debug: effort={self.debug_effort}, rl={self.debug_rl}, sim={self.debug_sim}")
        print(f"  output_prefix:       {self.output_file_prefix}")
        print(f"{'='*60}\n")

    def update_from_algorithm_config(self, algo_config: dict) -> None:
        """Automatically update environment parameters from a loaded RLlib algorithm config or config.json."""
        env_cfg = algo_config.get("env_config", algo_config)
        if not env_cfg:
            print("[WARN] No env_config found in algorithm state. Using CLI/default values.")
            return
        
        # RLlib often nested env_config under "env_config" in the main config
        actual_env_cfg = env_cfg.get("env_config", env_cfg) if isinstance(env_cfg, dict) else env_cfg
        
        mapping = {
            "n_agents": "n_agents",
            "start_agents": "start_agents",
            "max_steps": "max_steps",
            "max_rewardless_steps": "max_rewardless_steps",
            "n_groups": "n_groups",
            "max_peer_group_size": "max_peer_group_size",
            "n_projects_per_step": "n_projects_per_step",
            "max_projects_per_agent": "max_projects_per_agent",
            "max_agent_age": "max_agent_age",
            "acceptance_threshold": "acceptance_threshold",
            "reward_function": "reward_function",
            "reward_mode": "reward_function", # alternative key in some configs
            "topk_collab": "topk_collab",
            "topk_apply_to_all_agents": "topk_apply_to_all_agents",
            "policy_config_name": "policy_config_name",
            "prestige_threshold": "prestige_threshold",
            "novelty_threshold": "novelty_threshold",
            "effort_threshold": "effort_threshold",
            "group_policy_homogenous": "group_policy_homogenous",
        }

        updated = []
        for env_key, attr_name in mapping.items():
            if env_key in actual_env_cfg:
                val = actual_env_cfg[env_key]
                old_val = getattr(self, attr_name)
                if val != old_val:
                    setattr(self, attr_name, val)
                    updated.append(f"{attr_name}: {old_val} -> {val}")


# ---------------------------------------------------------------------------
# Reusable builder helpers — eliminate duplication
# ---------------------------------------------------------------------------

def build_env(cfg: EvalConfig) -> PeerGroupEnvironment:
    """Create a raw PeerGroupEnvironment from config.
    
    Uses defaults if some values are still None after sync.
    """
    env = PeerGroupEnvironment(
        start_agents=cfg.start_agents or 30,
        max_agents=cfg.n_agents or 64,
        max_steps=cfg.max_steps or 500,
        n_groups=cfg.n_groups or 8,
        max_peer_group_size=cfg.max_peer_group_size or 8,
        n_projects_per_step=cfg.n_projects_per_step or 1,
        max_projects_per_agent=cfg.max_projects_per_agent or 6,
        max_agent_age=cfg.max_agent_age or 750,
        max_rewardless_steps=cfg.max_rewardless_steps or 500,
        acceptance_threshold=cfg.acceptance_threshold or 0.5,
        reward_mode=cfg.reward_function or "by_effort",
    )
    # Propagate compatibility setting
    if hasattr(cfg, "expected_obs_size"):
        setattr(env, "expected_obs_size", cfg.expected_obs_size)
    return env


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
    # CRITICAL FIX for older Wrapper versions:
    # We must ensure observation_space is ready BEFORE the wrapper is initialized.
    try:
        env.reset(seed=cfg.seed)
        if hasattr(env, "observation_space"):
            _ = env.observation_space(cfg.controlled_agent_id)
    except Exception as e:
        print(f"[WARN] Pre-initialization in build_eval_wrapper failed: {e}")

    return RLLibSingleAgentWrapper(
        env,
        controlled_agent=cfg.controlled_agent_id,
        other_policies=other_policies,
        force_episode_horizon=cfg.max_steps,
        topk_collab=cfg.topk_collab,
        topk_apply_to_all_agents=cfg.topk_apply_to_all_agents,
        debug_effort=cfg.debug_effort,
    )


def make_env_creator_from_config(cfg: EvalConfig) -> Callable:
    """Return an RLlib-compatible env creator closure.

    Required by Algorithm.from_checkpoint() so RLlib can reconstruct the
    environment when restoring the checkpoint.
    """
    def _env_creator(env_config=None):
        # Merge default config with env_config if provided (RLlib often passes it)
        run_cfg = cfg
        if env_config:
             # print(f"[DEBUG_CREATOR] Updating EvalConfig from env_config: {env_config}")
             run_cfg = cfg.copy_with() # avoid modifying base cfg if needed
             run_cfg.update_from_algorithm_config(env_config)
        
        env = build_env(run_cfg)
        
        # --- CRITICAL FIX FOR COMPATIBILITY ---
        # The rolled-back RLLibSingleAgentWrapper expects observation_space to be ready in __init__.
        # If it's None, it crashes with RuntimeError.
        # We MUST ensure it's populated.
        
        try:
            # 1. Force environment initialization
            env.reset(seed=run_cfg.seed)
            
            # 2. Warm up the observation space cache for agent_0
            # This is what the wrapper calls in line 163 (or similar)
            ref_agent = run_cfg.controlled_agent_id
            if hasattr(env, "observation_space"):
                obs_space = env.observation_space(ref_agent)
                if hasattr(obs_space, "shape"):
                     print(f"[INFO] Environment initialized. Raw observation vector size: {obs_space.shape[0]} (before wrapper flattening)")
        except Exception as e:
            print(f"[WARN] Warming up env failed: {e}")

        agent_policies = build_heuristic_population(run_cfg)
        other_policies = build_other_policies(env, agent_policies, run_cfg)
        return build_eval_wrapper(env, other_policies, run_cfg)
    return _env_creator


# ---------------------------------------------------------------------------
# RL agent inference helper
# ---------------------------------------------------------------------------

def compute_rl_action(
    model: torch.nn.Module,
    obs_vec: np.ndarray,
    deterministic: bool,
    action_space: Optional[gym.spaces.Dict] = None,
    debug: bool = False,
) -> Union[int, np.ndarray, Dict[str, Any]]:
    """Compute an action from a flattened observation vector.

    Returns:
        Action (dict, int ID, or Box array) that the wrapper can decode.
    """
    obs_tensor = torch.from_numpy(obs_vec).unsqueeze(0).float()
    with torch.no_grad():
        # TorchModelV2 expects a dict with "obs" and returns (logits/means, state)
        model_out, _ = model({"obs": obs_tensor})

    # 1. Handle Dict output (newer API stack or specific model wrappers)
    if isinstance(model_out, dict):
        actions = {}
        for k, v in model_out.items():
            if deterministic:
                actions[k] = torch.argmax(v, dim=-1).item()
            else:
                probs = torch.softmax(v, dim=-1)
                actions[k] = torch.multinomial(probs, 1).item()
        return actions

    # 2. Handle Multi-head flattened output (Old API stack with Dict action space)
    if action_space is not None and isinstance(action_space, gym.spaces.Dict):
        # We need to split model_out into segments for each action head.
        # RLLib flattens Dict action spaces in a specific order (alphabetical usually).
        # Our keys: "choose_project", "collaborate_with", "put_effort"
        # Sorted keys: "choose_project", "collaborate_with", "put_effort"
        
        # Determine sizes
        cp_size = action_space["choose_project"].n
        cb_size = action_space["collaborate_with"].shape[0]
        pe_size = action_space["put_effort"].n
        
        # Check if model_out is a flattened vector of all logits
        # RLlib often outputs (N_logits) or (2 * N_logits) if it includes log_std for Box components.
        # But for Discrete, it's just logits.
        total_logits = cp_size + cb_size * 2 + pe_size # 2*cb if MultiBinary is treated as MultiDiscrete([2]*CB)
        
        # Let's check the actual size
        actual_size = model_out.shape[-1]
        
        # MultiBinary in RLlib with Torch usually results in 2 logits per bit (0 or 1)
        if actual_size == (cp_size + 2 * cb_size + pe_size):
            # Segment the flattened tensor
            cp_logits = model_out[0, 0:cp_size]
            cb_logits = model_out[0, cp_size : cp_size + 2 * cb_size].view(cb_size, 2)
            pe_logits = model_out[0, cp_size + 2 * cb_size : cp_size + 2 * cb_size + pe_size]
            
            if deterministic:
                cp_act = torch.argmax(cp_logits).item()
                pe_act = torch.argmax(pe_logits).item()
                cb_act = torch.argmax(cb_logits, dim=-1).cpu().numpy().astype(np.int8)
            else:
                cp_act = torch.multinomial(torch.softmax(cp_logits, dim=-1), 1).item()
                pe_act = torch.multinomial(torch.softmax(pe_logits, dim=-1), 1).item()
                cb_act = torch.multinomial(torch.softmax(cb_logits, dim=-1), 1).squeeze(-1).cpu().numpy().astype(np.int8)
            
            return {
                "choose_project": cp_act,
                "collaborate_with": cb_act,
                "put_effort": pe_act
            }
        
        # Another common case: Box(0, 1) for MultiBinary (treated as continuous in some models)
        elif actual_size == (cp_size + cb_size + pe_size):
            cp_logits = model_out[0, 0:cp_size]
            cb_raw = model_out[0, cp_size : cp_size + cb_size]
            pe_logits = model_out[0, cp_size + cb_size : cp_size + cb_size + pe_size]
            
            if deterministic:
                cp_act = torch.argmax(cp_logits).item()
                pe_act = torch.argmax(pe_logits).item()
                cb_act = (cb_raw > 0).cpu().numpy().astype(np.int8) # Simple threshold for raw output
            else:
                cp_act = torch.multinomial(torch.softmax(cp_logits, dim=-1), 1).item()
                pe_act = torch.multinomial(torch.softmax(pe_logits, dim=-1), 1).item()
                # For sampling cb we'd need probs, but if they are raw, we can't easily without sigmoid.
                cb_act = (torch.sigmoid(cb_raw) > torch.rand_like(cb_raw)).cpu().numpy().astype(np.int8)
                
            return {
                "choose_project": cp_act,
                "collaborate_with": cb_act,
                "put_effort": pe_act
            }

    # Fallback/Legacy handling:
    out_size = model_out.shape[-1]
    
    # If it's the old Discrete case:
    if out_size > 100: # heuristic: probably discrete macro-action
        if deterministic:
            return int(torch.argmax(model_out, dim=-1).item())
        else:
            probs = torch.softmax(model_out, dim=-1)
            return int(torch.multinomial(probs, 1).item())
    
    # If it's the Box case (CB + 2 or 2*(CB+2)):
    action = model_out[0].cpu().numpy()
    return action


def debug_obs_diff(vec1, vec2):
    print("len1:", len(vec1), "len2:", len(vec2))
    min_len = min(len(vec1), len(vec2))
    if min_len == 0:
        print("mean diff:", 0.0)
        print("max diff:", 0.0)
        return
    diff = np.abs(vec1[:min_len] - vec2[:min_len])
    print("mean diff:", float(diff.mean()))
    print("max diff:", float(diff.max()))


def debug_eval_obs_pipeline(
    wrapper: RLLibSingleAgentWrapper,
    nested_obs: Dict[str, Any],
    obs_vec_from_flatten: np.ndarray,
    model_expected_size: int,
    *,
    verbose: bool,
) -> None:
    raw_obs = nested_obs.get("observation", {}) if isinstance(nested_obs, dict) else {}
    raw_mask = nested_obs.get("action_mask", {}) if isinstance(nested_obs, dict) else {}

    normalized_obs = wrapper._normalize_observation(raw_obs)
    obs_only_vec = wrapper._flatten_any_like_template(normalized_obs, wrapper._obs_template)
    mask_vec = wrapper._flatten_mask_like_template(raw_mask, wrapper._mask_template)
    recomposed = np.concatenate([obs_only_vec, mask_vec]).astype(np.float32, copy=False)

    cp_raw = np.asarray(raw_mask.get("choose_project", []), dtype=np.float32).ravel()
    cb_raw = np.asarray(raw_mask.get("collaborate_with", []), dtype=np.float32).ravel()
    pe_raw = np.asarray(raw_mask.get("put_effort", []), dtype=np.float32).ravel()

    cp_tmpl = np.asarray(wrapper._mask_template.get("choose_project", []), dtype=np.float32).ravel()
    cb_tmpl = np.asarray(wrapper._mask_template.get("collaborate_with", []), dtype=np.float32).ravel()
    pe_tmpl = np.asarray(wrapper._mask_template.get("put_effort", []), dtype=np.float32).ravel()

    print("EVAL obs_vec size:", len(obs_vec_from_flatten))
    print("MODEL expected size:", model_expected_size)
    print("WRAPPER expected size:", wrapper.expected_obs_size)
    print("EVAL observation size (ohne mask):", len(obs_only_vec))
    print("EVAL action_mask size:", len(mask_vec))
    print("EVAL total recomposed size:", len(recomposed))
    print("EVAL mask raw sizes: choose_project=", len(cp_raw), "collaborate_with=", len(cb_raw), "put_effort=", len(pe_raw))
    print(
        "EVAL mask template sizes: choose_project=",
        len(cp_tmpl),
        "collaborate_with=",
        len(cb_tmpl),
        "put_effort=",
        len(pe_tmpl),
    )

    running_projects = raw_obs.get("running_projects", {}) if isinstance(raw_obs, dict) else {}
    print("EVAL running_projects raw keys:", len(running_projects) if isinstance(running_projects, dict) else "not-dict")
    print("EVAL has put_effort mask key:", "put_effort" in raw_mask if isinstance(raw_mask, dict) else False)
    print("EVAL has choose_project mask key:", "choose_project" in raw_mask if isinstance(raw_mask, dict) else False)
    print("EVAL has collaborate_with mask key:", "collaborate_with" in raw_mask if isinstance(raw_mask, dict) else False)

    if verbose:
        print("EVAL diff flatten_to_vector vs recomposed:")
        debug_obs_diff(obs_vec_from_flatten, recomposed)

        # Feature-length mismatches against template (top-level keys)
        if isinstance(wrapper._obs_template, dict) and isinstance(normalized_obs, dict):
            mismatches = []
            for key in sorted(wrapper._obs_template.keys()):
                tmpl_val = wrapper._obs_template[key]
                obs_val = normalized_obs.get(key, tmpl_val)
                exp_len = int(wrapper._flatten_any_like_template(tmpl_val, tmpl_val).size)
                got_len = int(wrapper._flatten_any_like_template(obs_val, tmpl_val).size)
                if exp_len != got_len:
                    mismatches.append((key, exp_len, got_len))
            if mismatches:
                print("EVAL feature length mismatches (template vs runtime):")
                for key, exp_len, got_len in mismatches:
                    print(f"  - {key}: expected {exp_len}, got {got_len}, delta={exp_len - got_len}")


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
    termination_reason: str = ""

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

    def record_termination(self, step: int, env: Optional[PeerGroupEnvironment] = None) -> None:
        """Mark the agent as terminated. Only records the first termination."""
        if self.terminated_step is None:
            self.terminated_step = step
            if env is not None:
                idx = self.agent_idx
                # In env.step(), termination depends on these two probabilities
                # We check which one was higher or if they crossed a threshold
                rewardless_dist = self.rewardless_steps - env.max_rewardless_steps
                age_limit = env.agent_ages[idx]
                age_dist = self.age - age_limit
                
                reasons = []
                if rewardless_dist > -10: # Close to or over the limit
                    reasons.append(f"REWARDLESS STEPS ({self.rewardless_steps}/{env.max_rewardless_steps})")
                if age_dist > -10: # Close to or over the limit
                    reasons.append(f"AGE ({self.age}/{int(age_limit)})")
                
                if not reasons:
                    # If neither is close, it was a very lucky/unlucky stochastic termination
                    # Let's see which probability was higher
                    if rewardless_dist / env.max_rewardless_steps > age_dist / age_limit:
                        reasons.append(f"REWARDLESS STEPS (stochastic, {self.rewardless_steps}/{env.max_rewardless_steps})")
                    else:
                        reasons.append(f"AGE (stochastic, {self.age}/{int(age_limit)})")
                
                self.termination_reason = " & ".join(reasons)

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
        reason_str = f"  Reason: {self.termination_reason}\n" if self.termination_reason else ""
        return (
            f"\n{'!'*60}\n"
            f"  RL AGENT ({self.agent_id}) TERMINATED at step {self.terminated_step}\n"
            f"{reason_str}"
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
    # CRITICAL: Save requested overrides BEFORE any restoration or sync happens
    # We only override if the value was provided via CLI (not None)
    overrides = {
        "reward_function": cfg.reward_function,
        "max_steps": cfg.max_steps,
        "max_rewardless_steps": cfg.max_rewardless_steps,
        "n_agents": cfg.n_agents,
        "start_agents": cfg.start_agents,
        "n_groups": cfg.n_groups,
        "max_peer_group_size": cfg.max_peer_group_size,
        "n_projects_per_step": cfg.n_projects_per_step,
        "max_projects_per_agent": cfg.max_projects_per_agent,
        "max_agent_age": cfg.max_agent_age,
        "acceptance_threshold": cfg.acceptance_threshold,
    }

    cfg.print_summary()

    # Seed all RNGs (random, numpy, torch, cuda) for reproducibility
    seed_everything(cfg.seed, workers=True)

    # Reproducibility: force deterministic PyTorch operations where possible
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---- 1) Init Ray & register env for checkpoint restoration ----
    # Ensure the registered environment uses the SYNCED config
    env_name = "peer_group_single_agent_fixed_population"
    tune.register_env(env_name, make_env_creator_from_config(cfg))

    # To avoid "size mismatch" errors during from_checkpoint(), we try to 
    # extract the environment configuration from the checkpoint's algorithm_state.pkl
    # or the new config.json if it exists, before we register the environment.
    checkpoint_path = os.path.abspath(cfg.checkpoint_path)

    if not os.path.exists(checkpoint_path):
        # Check if it's a relative path from the checkpoints folder
        alt_path = os.path.join("checkpoints", cfg.checkpoint_path)
        if os.path.exists(alt_path):
            checkpoint_path = os.path.abspath(alt_path)
        else:
            raise ValueError(f"Checkpoint path not found: {checkpoint_path}")

    # Prioritize config.json if available (explicitly requested by user)
    config_json = os.path.join(checkpoint_path, "config.json")
    if os.path.exists(config_json):
        try:
            with open(config_json, "r") as f:
                config_data = json.load(f)
            print(f"\nLoading configuration from {config_json}")
            cfg.update_from_algorithm_config(config_data)
        except Exception as e:
            print(f" Could not load config from {config_json}: {e}")

    state_file = os.path.join(checkpoint_path, "algorithm_state.pkl")
    if os.path.exists(state_file):
        try:
            import pickle
            with open(state_file, "rb") as f:
                state = pickle.load(f)
            # RLlib stores the config in the state
            if "config" in state:
                # Some RLlib versions store it as a dict, others as a Config object
                config_to_sync = state["config"]
                if hasattr(config_to_sync, "to_dict"):
                    config_to_sync = config_to_sync.to_dict()
                
                # IMPORTANT: Deep sync from algorithm_state.pkl to avoid size mismatch
                print(f"[SYNC] Pre-syncing configuration from {state_file}...")
                cfg.update_from_algorithm_config(config_to_sync)
                
                # Print what we found in the env_config to help debug size mismatches
                env_cfg = config_to_sync.get("env_config", config_to_sync)
                if isinstance(env_cfg, dict):
                    print(f"  -> Found in checkpoint: max_projects_per_agent={env_cfg.get('max_projects_per_agent')}, max_peer_group_size={env_cfg.get('max_peer_group_size')}")
        except Exception as e:
            print(f"[WARN] Could not pre-sync config from {state_file}: {e}")

    # Ensure the registered environment uses the SYNCED config
    tune.register_env(env_name, make_env_creator_from_config(cfg))

    # ---- 2) Restore trained algorithm from checkpoint ----
    print(f"Restoring algorithm from checkpoint: {checkpoint_path}")
    
    # Identify expected observation size if possible
    expected_obs_size = None
    state_file = os.path.join(checkpoint_path, "algorithm_state.pkl")
    if os.path.exists(state_file):
        try:
            import pickle
            with open(state_file, "rb") as f:
                state = pickle.load(f)
            # Try to find observation space in the state
            if "worker" in state and "policy_states" in state["worker"]:
                p_states = state["worker"]["policy_states"]
                # RLlib often uses "default_policy"
                p_id = "default_policy"
                if p_id not in p_states and len(p_states) > 0:
                     p_id = next(iter(p_states.keys()))
                
                if p_id in p_states:
                    obs_space = p_states[p_id].get("observation_space")
                    if hasattr(obs_space, "shape"):
                         expected_obs_size = obs_space.shape[0]
                         # print(f"[SYNC] Detected expected observation size from checkpoint: {expected_obs_size}")
        except Exception as e:
            print(f"[WARN] Could not detect expected obs size: {e}")

    if expected_obs_size:
        # We pass this into the env creator via a special key in other_policies 
        # (which is currently just a dict of callables, but we can abuse it for config)
        # or we could add it to EvalConfig.
        # Let's add it to EvalConfig to be clean.
        if not hasattr(cfg, "expected_obs_size"):
             # Dynamic attribute addition if not in dataclass (though it is better to add it to the class)
             setattr(cfg, "expected_obs_size", expected_obs_size)
        else:
             cfg.expected_obs_size = expected_obs_size

    # Try to load algorithm from checkpoint
    try:
        algo = Algorithm.from_checkpoint(checkpoint_path)
        print("Algorithm restored successfully.")

        # Sync Environment Config from Checkpoint
        cfg.update_from_algorithm_config(algo.config.to_dict())
    except (AttributeError, TypeError, ModuleNotFoundError, ImportError) as e:
        print(f"[WARN] Could not load algorithm from checkpoint: {e}")
        print("[INFO] Using default configuration values from script")

        # Build algorithm from scratch using the config we have
        from ray.rllib.algorithms.ppo import PPOConfig

        # Create a basic PPO config with OLD API stack to match checkpoint
        algo_config = PPOConfig()
        algo_config = algo_config.api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        algo_config = algo_config.environment(env=env_name)

        # Try to restore just the model weights if possible
        try:
            algo = algo_config.build()

            # Attempt to restore model weights manually by loading state dict
            state_file = os.path.join(checkpoint_path, "algorithm_state.pkl")
            if os.path.exists(state_file):
                print("[INFO] Attempting to restore model weights from checkpoint...")
                # Note: This will likely fail due to missing 'callbacks' module
                # but we try anyway in case the issue is resolved
                import pickle
                with open(state_file, "rb") as f:
                    state = pickle.load(f)
                # Restore only the weights, not the full config
                if "worker" in state:
                    print("[INFO] Restoring model weights from checkpoint")
                    algo.__setstate__(state)
                print("[INFO] Algorithm initialized with default config and checkpoint weights")
        except Exception as restore_error:
            print(f"[WARN] Could not restore model weights: {restore_error}")
            print("[INFO] Creating fresh algorithm with default config (no checkpoint weights)")
            # algo is already built, just continue without weights

    # Restore overrides if they were set via CLI or automation loop
    # We use a forced override here because update_from_algorithm_config might have
    # reverted these to the checkpoint's values.
    for key, val in overrides.items():
        if val is not None:
            old_val = getattr(cfg, key)
            if old_val != val:
                print(f"[SYNC] Overriding synced {key} '{old_val}' -> '{val}'")
                setattr(cfg, key, val)

    # Now that all synchronizations are complete, print the FINAL configuration
    print("\n[OK] FINAL EVALUATION CONFIG (after synchronization):")
    cfg.print_summary()

    # ---- 3) Build evaluation environment ----
    # Re-registering after sync if the env creator needs synced parameters for internal RLlib use,
    # though our manual simulation (Step 6) uses build_env(cfg) directly.
    tune.register_env(env_name, make_env_creator_from_config(cfg))

    # Get the RLModule for inference.
    # Why algo.get_policy().model instead of algo.get_module()?
    #   - We are on the OLD API stack (enable_rl_module_and_learner=False).
    #   - On the old stack, algo.get_module() fails because the workers are RolloutWorkers, not EnvRunners.
    #   - algo.get_policy().model gives us the TorchModelV2 which we can use for inference.
    policy = algo.get_policy()
    rl_module = policy.model
    rl_module.eval()
    model_expected_size = 0
    try:
        if hasattr(policy, "observation_space") and getattr(policy.observation_space, "shape", None):
            model_expected_size = int(np.prod(policy.observation_space.shape))
    except Exception:
        model_expected_size = 0
    if model_expected_size <= 0:
        try:
            if (
                hasattr(rl_module, "_hidden_layers")
                and hasattr(rl_module._hidden_layers[0], "_model")
                and hasattr(rl_module._hidden_layers[0]._model[0], "in_features")
            ):
                model_expected_size = int(rl_module._hidden_layers[0]._model[0].in_features)
        except Exception:
            model_expected_size = 0

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
    eval_obs_debug_printed = False

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
                obs_vec = helper_wrapper._ensure_obs_vector_ok(obs_vec, where="eval")

                # --- VALIDATION: Check for observation size mismatch before inference ---
                if not eval_obs_debug_printed:
                    debug_eval_obs_pipeline(
                        helper_wrapper,
                        nested_obs,
                        obs_vec,
                        model_expected_size,
                        verbose=True,
                    )
                    eval_obs_debug_printed = True

                if model_expected_size > 0 and len(obs_vec) != model_expected_size:
                    debug_eval_obs_pipeline(
                        helper_wrapper,
                        nested_obs,
                        obs_vec,
                        model_expected_size,
                        verbose=True,
                    )
                    print(f"\n[CRITICAL ERROR] Observation size mismatch!")
                    print(f"  Model expects: {model_expected_size} features")
                    print(f"  Environment produces: {len(obs_vec)} features")
                    print(f"  Wrapper expected: {helper_wrapper.expected_obs_size} features")
                    print(f"  Current Config: max_peer_group_size={cfg.max_peer_group_size}, max_projects_per_agent={cfg.max_projects_per_agent}")

                    # Provide helpful suggestions for common mismatches
                    print("\n  POSSIBLE FIXES:")
                    if model_expected_size == 451 and cfg.max_projects_per_agent == 8 and cfg.max_peer_group_size == 40:
                        print("  -> Try setting max_projects_per_agent = 10 (gives 451 features)")
                    elif model_expected_size == 412 and cfg.max_projects_per_agent == 10 and cfg.max_peer_group_size == 40:
                        print("  -> Try setting max_projects_per_agent = 8 (gives 412 features)")

                    print(f"  Please adjust these parameters in the MANUAL OVERRIDES block at the end of the script.\n")
                    raise RuntimeError(f"Input mismatch: Model expects {model_expected_size}, got {len(obs_vec)}")

                action_id = compute_rl_action(rl_module, obs_vec, cfg.deterministic, action_space=helper_wrapper.action_space, debug=cfg.debug_effort)
                
                # DEBUG RL INFERENCE (detailed RL observations)
                if cfg.debug_rl and (step % cfg.debug_freq == 0 or (step < 100 and step % 20 == 0) or step < 3):
                    print(f"DEBUG Step {step}: === RL Agent Observation ===")

                    # Get all project slots (including empty ones) from environment
                    # agent_active_projects is a list of project IDs (or None for empty slots)
                    all_project_slots = env.agent_active_projects[agent_idx]

                    print(f"DEBUG Step {step}: all_slots={len(all_project_slots)} (active={len([p for p in all_project_slots if p is not None])})")

                    # Show each slot with its project information
                    for slot_idx, proj_id in enumerate(all_project_slots):
                        if proj_id is None:
                            print(f"DEBUG Step {step}: [Slot {slot_idx}] EMPTY")
                        else:
                            # proj_id is a string like 'project_20-0-2'
                            project = env.projects[proj_id]
                            required_effort = project.required_effort
                            time_left = max(0, project.time_window - (env.timestep - project.start_time))

                            # Get total effort contributed to this project from all agents
                            # agent_project_effort is a List[Dict[str, float]] where key is project_id
                            total_effort = sum(env.agent_project_effort[agent_i].get(proj_id, 0)
                                              for agent_i in range(env.n_agents))

                            print(f"DEBUG Step {step}: [Slot {slot_idx}] {proj_id} | effort={total_effort}/{required_effort} | deadline={time_left} steps")

                decoded = helper_wrapper._decode_action(action_id, agent_id=agent)
                
                decoded = helper_wrapper._apply_action_mask(
                    decoded, nested_obs, agent_id=agent
                )
                
                if cfg.debug_rl and (step % cfg.debug_freq == 0 or step < 3):
                    # Extract action components for debug display
                    choose_project = decoded.get('choose_project', 0) if isinstance(decoded, dict) else 0
                    put_effort = decoded.get('put_effort', 0) if isinstance(decoded, dict) else 0

                    # put_effort is 1-based slot index: 0=no project, 1=slot 0, 2=slot 1, etc.
                    effort_slot = put_effort - 1 if put_effort > 0 else -1

                    if put_effort > 0:
                        # Show which slot receives effort
                        all_project_slots = env.agent_active_projects[agent_idx]
                        if effort_slot < len(all_project_slots):
                            target_proj = all_project_slots[effort_slot]
                            if target_proj is not None:
                                print(f"DEBUG Step {step}: action=(choose_project={choose_project}, put_effort={put_effort} → [Slot {effort_slot}] {target_proj})")
                            else:
                                print(f"DEBUG Step {step}: action=(choose_project={choose_project}, put_effort={put_effort} → [Slot {effort_slot}] EMPTY)")
                        else:
                            print(f"DEBUG Step {step}: action=(choose_project={choose_project}, put_effort={put_effort})")
                    else:
                        print(f"DEBUG Step {step}: action=(choose_project={choose_project}, put_effort={put_effort})")

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
                rl_status.record_termination(step, env=env)
                if cfg.debug_sim:
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
            if cfg.debug_sim:
                n_active = int(np.sum(env.active_agents))
                print(
                    f"Step {step:3d}: total active agents: {n_active} "
                    f"RL {rl_status.agent_id}: age={rl_status.age} "
                    f"rewardless={rl_status.rewardless_steps} "
                    f"projects(active={rl_status.n_active_projects}, "
                    f"done={rl_status.completed_projects}, "
                    f"published={rl_status.successful_projects}) "
                    f"reward total = {rl_status.total_reward:.4f}"
                )

        # ---- Check if all agents are done ----
        if all(terminations.values()):
            if cfg.debug_sim:
                print(f"Simulation ended at step {step}")
            break

        # ---- Safety stop: no active agents left ----
        if int(np.sum(env.active_agents)) == 0:
            if cfg.debug_sim:
                print(f"Simulation ended at step {step}: no active agents left")
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
    # ray.shutdown() # Shutdown handled by the loop/main function

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
        "--checkpoint", type=str, default="checkpoints",
        help="Path to the RLlib checkpoint directory (default: checkpoints/)",
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
    parser.add_argument("--n-agents", type=int, default=None)
    parser.add_argument("--start-agents", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-rewardless-steps", type=int, default=None)
    parser.add_argument("--n-groups", type=int, default=None)
    parser.add_argument("--max-peer-group-size", type=int, default=None)
    parser.add_argument("--n-projects-per-step", type=int, default=None)
    parser.add_argument("--max-projects-per-agent", type=int, default=None)
    parser.add_argument("--max-agent-age", type=int, default=None)

    # Reward & thresholds
    parser.add_argument("--acceptance-threshold", type=float, default=None)
    parser.add_argument(
        "--reward-function", type=str, default=None,
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
    parser.add_argument(
        "--debug-effort", action="store_true",
        help="Enable detailed logging for action components (raw, scaled, rounded)",
    )
    parser.add_argument(
        "--debug-rl", action="store_true",
        help="Enable detailed RL observations/actions during simulation",
    )
    parser.add_argument(
        "--debug-sim", action="store_true", default=True,
        help="Enable general simulation progress and status banners",
    )
    parser.add_argument(
        "--debug-all", action="store_true",
        help="Enable all debug flags",
    )
    parser.add_argument(
        "--debug-freq", type=int, default=50,
        help="Debug log frequency in steps (default: 50). Set to 1 for every step, 10 for every 10 steps, etc.",
    )

    # Top-k Collaboration
    parser.add_argument(
        "--topk", type=int, default=None,
        help="If set, restricts collaboration to top-k partners per step (default: None)",
    )
    parser.add_argument(
        "--topk-all-agents", action="store_true", default=False,
        help="If set, applies top-k also to heuristic agents (default: True)",
    )
    parser.add_argument(
        "--no-topk-all-agents", action="store_false", dest="topk_all_agents",
        help="If set, applies top-k ONLY to the controlled RL agent",
    )
    
    # Automation
    parser.add_argument(
        "--num-seeds", type=int, default=1,
        help="Number of consecutive seeds to evaluate (default: 1)",
    )
    parser.add_argument(
        "--all-rewards", action="store_true",
        help="If set, evaluates for all reward functions ('multiply', 'evenly', 'by_effort')",
    )

    args = parser.parse_args()

    if args.debug_all:
        args.debug_effort = True
        args.debug_rl = True
        args.debug_sim = True

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
        topk_collab=args.topk,
        topk_apply_to_all_agents=args.topk_all_agents,
        debug_effort=args.debug_effort,
        debug_rl=args.debug_rl,
        debug_sim=args.debug_sim,
        debug_freq=args.debug_freq,
        controlled_agent_id=args.controlled_agent_id,
        deterministic=not args.stochastic,
        seed=args.seed,
        output_file_prefix=args.output_prefix,
    ), args.num_seeds, args.all_rewards


if __name__ == "__main__":
    base_config, num_seeds, all_rewards = parse_args()

    # --- MANUAL PARAMETER OVERRIDES ---
    # Hier können alle Parameter manuell gesetzt werden, um CLI-Argumente zu überschreiben.
    # Dies ist nützlich, wenn der Checkpoint-Sync fehlschlägt oder man schnell testen will.
    # HINWEIS: Wenn diese Werte gesetzt sind, werden sie bevorzugt gegenüber den Werten aus dem Checkpoint verwendet!
    
    # Checkpoint
    # base_config.checkpoint_path = "checkpoints/my_checkpoint"
    
    # Environment Parameter (Beispielwerte für 451 Features)
    ENABLE_MANUAL_OVERRIDES = False
    if ENABLE_MANUAL_OVERRIDES:
        base_config.n_agents = 60
        base_config.start_agents = 100
        base_config.max_steps = 600
        base_config.max_rewardless_steps = 50
        base_config.n_groups = 10
        base_config.max_peer_group_size = 10
        base_config.n_projects_per_step = 1
        base_config.max_projects_per_agent = 8
        base_config.max_agent_age = 750
        base_config.acceptance_threshold = 0.44

        # Heuristik-Schwellenwerte
        base_config.prestige_threshold = 0.29
        base_config.novelty_threshold = 0.4
        base_config.effort_threshold = 35

        # Simulation
        base_config.deterministic = True
        base_config.seed = 42
    
    # Debugging
    # base_config.debug_sim = True
    # base_config.debug_rl = True
    # base_config.debug_effort = True
    
    # ----------------------------------

    # Move ray.init() outside the simulation loop to avoid re-init issues
    # and improve performance.
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

    # Prepare reward functions
    # If all_rewards is set, use the list. Otherwise, use what was in base_config.
    if all_rewards:
        reward_functions = ["multiply", "evenly", "by_effort"]
    elif base_config.reward_function is not None:
        reward_functions = [base_config.reward_function]
    else:
        # Final fallback if neither CLI arg nor default is set
        reward_functions = ["multiply"]

    start_seed = base_config.seed

    for reward_fn in reward_functions:
        print(f"\n{'='*60}")
        print(f"STARTING EVALUATION FOR REWARD FUNCTION: {reward_fn}")
        print(f"{'='*60}\n")
        
        for i in range(num_seeds):
            current_seed = start_seed + i
            
            # Create a copy of the config for this specific run
            config = base_config.copy_with(
                reward_function=reward_fn,
                seed=current_seed,
                output_file_prefix=f"rl_ppo_{reward_fn}_s{current_seed}"
            )
            
            print(f"\n--- Run {i+1}/{num_seeds} | Seed: {current_seed} | Reward: {reward_fn} ---")
            run_simulation_with_rl_agent(config)

    # Cleanup Ray after all runs are completed
    ray.shutdown()

    print("\nEvaluation batch completed.")

