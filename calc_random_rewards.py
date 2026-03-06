"""
test_random_rewards.py

Sparse-reward diagnostic that mirrors the PPO training setting exactly:
  - agent_0 acts randomly (with action-mask awareness)
  - all other agents follow the SAME fixed-policy archetypes as during training
    (Balanced: 1/3 Careerist, 1/3 Orthodox, 1/3 Mass Producer)

ALL pipeline counters and conditional probabilities are agent_0-only.
Global env_metrics are shown separately for context but never mixed
into the agent_0 analysis.

Agent0Tracker tracks everything via Project-ID sets, with no dependency
on fragile integer counters like agent_completed_projects.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np

from agent_policies import (
    create_mixed_policy_population,
    do_nothing_policy,
    get_policy_function,
)
from env.peer_group_environment import PeerGroupEnvironment
from rllib_single_agent_wrapper import RLLibSingleAgentWrapper

# ═══════════════════════════════════════════════════════════════════════
# Configuration  (matches train_ppo_rllib.py argparse defaults exactly)
# ═══════════════════════════════════════════════════════════════════════
TOTAL_STEPS: int = 10_000
SEED: int = 42
BATCH_SIZE: int = 32_000  # PPO train_batch_size (for expected-reward calc)
DEBUG: bool = True         # enable reward-verification warnings

CONTROLLED_AGENT: str = "agent_0"
POLICY_CONFIG_NAME: str = "Balanced"
POLICY_DISTRIBUTION: Dict[str, float] = {
    "careerist": 1 / 3,
    "orthodox_scientist": 1 / 3,
    "mass_producer": 1 / 3,
}

# Heuristic thresholds (same as training defaults)
PRESTIGE_THRESHOLD: float = 0.2
NOVELTY_THRESHOLD: float = 0.8
EFFORT_THRESHOLD: int = 22

# Environment parameters (same as training argparse defaults)
ENV_KWARGS: Dict[str, Any] = dict(
    start_agents=60,
    max_agents=64,
    max_steps=500,
    max_peer_group_size=8,
    n_groups=8,
    n_projects_per_step=1,
    max_projects_per_agent=6,
    max_agent_age=750,
    max_rewardless_steps=500,
    growth_rate=0.04,
    acceptance_threshold=0.5,
    reward_mode="by_effort",
)


# ═══════════════════════════════════════════════════════════════════════
# Agent0Tracker – read-only, PID-set-based instrumentation
# ═══════════════════════════════════════════════════════════════════════

class Agent0Tracker:
    """Tracks agent_0-only project pipeline entirely via Project-ID sets.

    No dependency on env.agent_completed_projects or any integer delta
    logic.  All state transitions are detected by diffing PID sets
    against env.projects each step.

    Terminology:
      - involved_started: agent_0 appears in proj.contributors
        (there is no 'initiator' field, so we cannot distinguish
         initiator vs. collaborator)
      - finished: proj.finished == True (or equivalent)
      - published: finished AND final_reward > 0
        (cross-checked with env.agent_successful_projects when available)
      - rejected: finished AND NOT published
    """

    # Field-name candidates for defensive probing
    _CONTRIBUTORS_FIELDS = ("contributors", "agents", "members", "authors")
    _FINISHED_FIELDS = ("finished", "is_finished", "done", "completed")
    _REWARD_FIELDS = ("final_reward", "reward", "quality_reward")
    _ACCEPTED_FIELDS = ("accepted", "is_accepted", "published")

    def __init__(
        self,
        base_env: PeerGroupEnvironment,
        agent_id: str = "agent_0",
        debug: bool = True,
    ) -> None:
        self.env = base_env
        self.agent_id = agent_id
        self.debug = debug

        # Resolve integer index
        if not hasattr(base_env, "agent_to_id"):
            raise AttributeError(
                f"base_env has no 'agent_to_id'. "
                f"Available attrs: {sorted(a for a in dir(base_env) if not a.startswith('__'))}"
            )
        if agent_id not in base_env.agent_to_id:
            raise KeyError(f"'{agent_id}' not in agent_to_id")
        self.idx: int = base_env.agent_to_id[agent_id]

        if not hasattr(base_env, "projects"):
            raise AttributeError(
                f"base_env has no 'projects' dict. "
                f"Available attrs: {sorted(a for a in dir(base_env) if not a.startswith('__'))}"
            )

        # Track which field names we actually resolved (set on first probe)
        self._resolved_contributors_field: Optional[str] = None
        self._resolved_finished_field: Optional[str] = None
        self._resolved_reward_field: Optional[str] = None
        self._resolved_accepted_field: Optional[str] = None
        # One-shot debug warnings (don't spam)
        self._warned_contributors: bool = False
        self._warned_finished: bool = False
        self._warned_reward: bool = False
        self._warned_accepted: bool = False

        # Check if env has agent_successful_projects for cross-validation
        self._has_successful_projects: bool = hasattr(base_env, "agent_successful_projects")

        self.reset()

    # ── Helper: defensive field probing ───────────────────────────────

    def _proj_contributors(self, proj: Any) -> Set[int]:
        """Return set of contributor indices for a project."""
        # Fast path: already resolved
        if self._resolved_contributors_field is not None:
            val = getattr(proj, self._resolved_contributors_field, None)
            if val is not None:
                return set(val)

        # Probe candidates
        for field in self._CONTRIBUTORS_FIELDS:
            val = getattr(proj, field, None)
            if val is not None:
                self._resolved_contributors_field = field
                return set(val)

        # Fallback: warn once
        if self.debug and not self._warned_contributors:
            self._warned_contributors = True
            attrs = [a for a in dir(proj) if not a.startswith("_")]
            print(f"  [WARN] Agent0Tracker: no contributors field found on Project. "
                  f"Tried: {self._CONTRIBUTORS_FIELDS}. "
                  f"Available: {attrs}")
        return set()

    def _proj_finished(self, proj: Any) -> bool:
        """Return whether a project is finished."""
        if self._resolved_finished_field is not None:
            return bool(getattr(proj, self._resolved_finished_field, False))

        for field in self._FINISHED_FIELDS:
            val = getattr(proj, field, None)
            if val is not None:
                self._resolved_finished_field = field
                return bool(val)

        if self.debug and not self._warned_finished:
            self._warned_finished = True
            attrs = [a for a in dir(proj) if not a.startswith("_")]
            print(f"  [WARN] Agent0Tracker: no finished field found on Project. "
                  f"Tried: {self._FINISHED_FIELDS}. Available: {attrs}")
        return False

    def _proj_accepted(self, proj: Any) -> Optional[bool]:
        """Return whether a finished project was accepted/published.

        Priority:
          1) Explicit accepted/published boolean field
          2) final_reward > 0 means published, == 0 means rejected
        Returns None if undeterminable.
        """
        # Try explicit boolean first
        if self._resolved_accepted_field is not None:
            val = getattr(proj, self._resolved_accepted_field, None)
            if val is not None:
                return bool(val)

        for field in self._ACCEPTED_FIELDS:
            val = getattr(proj, field, None)
            if val is not None:
                self._resolved_accepted_field = field
                return bool(val)

        # Fallback: derive from final_reward
        if self._resolved_reward_field is not None:
            val = getattr(proj, self._resolved_reward_field, None)
            if val is not None:
                return float(val) > 0

        for field in self._REWARD_FIELDS:
            val = getattr(proj, field, None)
            if val is not None:
                self._resolved_reward_field = field
                return float(val) > 0

        if self.debug and not self._warned_accepted:
            self._warned_accepted = True
            attrs = [a for a in dir(proj) if not a.startswith("_")]
            print(f"  [WARN] Agent0Tracker: no accepted/reward field on Project. "
                  f"Tried: {self._ACCEPTED_FIELDS + self._REWARD_FIELDS}. "
                  f"Available: {attrs}")
        return None

    # ── Core API ──────────────────────────────────────────────────────

    def reset(self) -> None:
        """Call at episode start (before snapshot_reset)."""
        self._seen_pids: Set[str] = set()
        self._finished_pids: Set[str] = set()
        self._published_pids: Set[str] = set()
        self._rejected_pids: Set[str] = set()

        self.involved_started: int = 0
        self.finished_count: int = 0
        self.published_count: int = 0
        self.rejected_count: int = 0
        # due == finished in this env (no separate evaluation phase)
        self.due_count: int = 0

    def snapshot_reset(self) -> None:
        """Take baseline snapshot right after env.reset().

        Captures any projects agent_0 is already involved in
        (should normally be empty after a fresh reset).
        """
        self._seen_pids = self._scan_relevant_pids()
        # Mark any already-finished ones (defensive, normally empty)
        for pid in self._seen_pids:
            proj = self.env.projects.get(pid)
            if proj is not None and self._proj_finished(proj):
                self._finished_pids.add(pid)
                accepted = self._proj_accepted(proj)
                if accepted is True:
                    self._published_pids.add(pid)
                elif accepted is False:
                    self._rejected_pids.add(pid)

    def _scan_relevant_pids(self) -> Set[str]:
        """Return all PIDs where agent_0 is a contributor."""
        result: Set[str] = set()
        for pid, proj in self.env.projects.items():
            if self.idx in self._proj_contributors(proj):
                result.add(pid)
        return result

    def step(self) -> None:
        """Call AFTER env.step() to update all counters."""
        current_relevant = self._scan_relevant_pids()

        # ── New projects agent_0 got involved in ──
        new_pids = current_relevant - self._seen_pids
        self.involved_started += len(new_pids)
        self._seen_pids = current_relevant

        # ── Newly finished projects ──
        new_finished: Set[str] = set()
        for pid in current_relevant:
            if pid in self._finished_pids:
                continue  # already counted
            proj = self.env.projects.get(pid)
            if proj is not None and self._proj_finished(proj):
                new_finished.add(pid)

        self._finished_pids |= new_finished
        self.finished_count += len(new_finished)

        # ── Published vs rejected among newly finished ──
        # Primary: check proj-level accepted/reward fields
        new_published: Set[str] = set()
        new_rejected: Set[str] = set()

        for pid in new_finished:
            proj = self.env.projects.get(pid)
            if proj is None:
                continue
            accepted = self._proj_accepted(proj)
            if accepted is True:
                new_published.add(pid)
            elif accepted is False:
                new_rejected.add(pid)
            # else: undeterminable, don't count either way

        # Cross-validate with env.agent_successful_projects if available
        if self._has_successful_projects:
            env_successful = set(self.env.agent_successful_projects[self.idx])
            # Any PID in env_successful that we missed?
            for pid in new_finished:
                if pid in env_successful and pid not in new_published:
                    new_published.add(pid)
                    new_rejected.discard(pid)

        self._published_pids |= new_published
        self._rejected_pids |= new_rejected
        self.published_count += len(new_published)
        self.rejected_count += len(new_rejected)

        # due == finished (no separate evaluation phase in this env)
        self.due_count = self.finished_count

    @property
    def open_end(self) -> int:
        """Projects agent_0 is involved in that are NOT yet finished."""
        count = 0
        for pid in self._seen_pids:
            if pid in self._finished_pids:
                continue
            proj = self.env.projects.get(pid)
            if proj is not None and not self._proj_finished(proj):
                count += 1
        return count

    def verify_reward(self, wrapper_reward: float) -> Optional[str]:
        """Compare wrapper reward with base_env per-agent reward dict.

        Probes multiple possible reward-dict locations.
        Returns a warning string if mismatch, else None.
        """
        env_r: Optional[float] = None

        # Try sources in priority order
        for attr in ("rewards", "last_rewards", "_rewards"):
            d = getattr(self.env, attr, None)
            if isinstance(d, dict) and self.agent_id in d:
                env_r = float(d[self.agent_id])
                break

        if env_r is None:
            return None

        if abs(float(wrapper_reward) - env_r) > 1e-7:
            return (f"REWARD MISMATCH: wrapper={wrapper_reward:.8f} "
                    f"vs env={env_r:.8f}")
        return None

    def summary_dict(self) -> Dict[str, int]:
        return {
            "involved_started_0": self.involved_started,
            "finished_0": self.finished_count,
            "published_0": self.published_count,
            "rejected_0": self.rejected_count,
            "due_0": self.due_count,
            "open_end_0": self.open_end,
        }


# ═══════════════════════════════════════════════════════════════════════
# Build fixed-policy population  (identical to make_env_creator)
# ═══════════════════════════════════════════════════════════════════════

def _build_other_policies(
    env: PeerGroupEnvironment,
    controlled_agent_id: str,
) -> Dict[str, Callable[[Any], Any]]:
    careerist_fn = get_policy_function("careerist")
    orthodox_fn = get_policy_function("orthodox_scientist")
    mass_prod_fn = get_policy_function("mass_producer")

    def _policy_from_name(policy_name: Optional[str]) -> Callable:
        if policy_name == "careerist":
            def _fn(nested_obs: Any) -> Any:
                return careerist_fn(
                    nested_obs["observation"], nested_obs["action_mask"],
                    PRESTIGE_THRESHOLD)
            return _fn
        if policy_name == "orthodox_scientist":
            def _fn(nested_obs: Any) -> Any:
                return orthodox_fn(
                    nested_obs["observation"], nested_obs["action_mask"],
                    NOVELTY_THRESHOLD)
            return _fn
        if policy_name == "mass_producer":
            def _fn(nested_obs: Any) -> Any:
                return mass_prod_fn(
                    nested_obs["observation"], nested_obs["action_mask"],
                    EFFORT_THRESHOLD)
            return _fn
        def _fb(nested_obs: Any) -> Any:
            return do_nothing_policy(
                nested_obs["observation"], nested_obs["action_mask"])
        return _fb

    agent_policy_names = create_mixed_policy_population(
        env.n_agents, POLICY_DISTRIBUTION, seed=SEED,
    )

    other_policies: Dict[str, Callable] = {}
    for agent_id in env.possible_agents:
        if agent_id == controlled_agent_id:
            continue
        idx = env.agent_to_id[agent_id]
        other_policies[agent_id] = _policy_from_name(agent_policy_names[idx])

    return other_policies


# ═══════════════════════════════════════════════════════════════════════
# Masked random action sampling  (only for the controlled agent)
# ═══════════════════════════════════════════════════════════════════════

def _sample_masked_action(
    rng: np.random.RandomState,
    obs_vec: np.ndarray,
    wrapper: RLLibSingleAgentWrapper,
) -> int:
    """Sample a uniform random macro-action respecting the action mask."""
    try:
        cp_n = wrapper._CP
        cb_n = wrapper._CB
        pe_n = wrapper._PE
        mask_len = cp_n + cb_n + pe_n
        mask_flat = obs_vec[-mask_len:]

        cp_mask = mask_flat[:cp_n]
        cb_mask = mask_flat[cp_n:cp_n + cb_n]
        pe_mask = mask_flat[cp_n + cb_n:]

        valid_cp = np.where(cp_mask > 0)[0]
        valid_pe = np.where(pe_mask > 0)[0]
        if len(valid_cp) == 0:
            valid_cp = np.array([0])
        if len(valid_pe) == 0:
            valid_pe = np.array([0])

        collab_bits = np.zeros(cb_n, dtype=np.int8)
        for i in range(cb_n):
            if cb_mask[i] > 0:
                collab_bits[i] = rng.randint(0, 2)

        cp = int(rng.choice(valid_cp))
        pe = int(rng.choice(valid_pe))
        collab_code = sum(int(collab_bits[i]) << i for i in range(cb_n))

        action_id = ((cp * pe_n) + pe) * (1 << cb_n) + collab_code
        return int(np.clip(action_id, 0, wrapper.action_space.n - 1))
    except Exception:
        return int(rng.randint(0, wrapper.action_space.n))


# ═══════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════

def _ratio(a: float, b: float) -> str:
    return f"{a / b:.4f}" if b > 0 else "n/a"


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 65)
    print("SPARSE-REWARD DIAGNOSTIC  (agent_0 random, others heuristic)")
    print("=" * 65)
    print(f"  TOTAL_STEPS         : {TOTAL_STEPS}")
    print(f"  SEED                : {SEED}")
    print(f"  CONTROLLED_AGENT    : {CONTROLLED_AGENT}")
    print(f"  POLICY_CONFIG       : {POLICY_CONFIG_NAME}")
    print(f"  BATCH_SIZE (PPO)    : {BATCH_SIZE}")
    print(f"  DEBUG               : {DEBUG}")
    for k, v in ENV_KWARGS.items():
        print(f"  env.{k:22s}: {v}")
    print("=" * 65)

    # ── Build env (identical to training) ─────────────────────────────
    base_env = PeerGroupEnvironment(**ENV_KWARGS)
    other_policies = _build_other_policies(base_env, CONTROLLED_AGENT)

    env = RLLibSingleAgentWrapper(
        env=base_env,
        controlled_agent=CONTROLLED_AGENT,
        other_policies=other_policies,
        force_episode_horizon=ENV_KWARGS["max_steps"],
        topk_collab=3,
        topk_apply_to_all_agents=True,
    )

    tracker = Agent0Tracker(base_env, CONTROLLED_AGENT, debug=DEBUG)

    # ── Rollout ───────────────────────────────────────────────────────
    rng = np.random.RandomState(SEED)
    obs, info = env.reset(seed=SEED)
    tracker.reset()
    tracker.snapshot_reset()

    all_rewards: List[float] = []
    positive_count: int = 0
    negative_count: int = 0
    zero_count: int = 0
    episode_returns: List[float] = []
    ep_return: float = 0.0
    episodes: int = 0
    step: int = 0
    reward_mismatches: int = 0

    # Accumulated agent_0-only counters across all episodes
    total_involved_started_0: int = 0
    total_finished_0: int = 0
    total_published_0: int = 0
    total_rejected_0: int = 0
    total_open_end_0: int = 0
    total_due_0: int = 0

    # Accumulated global env_metrics (context only)
    acc_global: Dict[str, float] = {}

    while step < TOTAL_STEPS:
        action = _sample_masked_action(rng, obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        reward_f = float(reward)
        all_rewards.append(reward_f)
        ep_return += reward_f

        # Update agent_0 tracker (PID-set based, read-only)
        tracker.step()

        # Debug: verify reward matches base_env
        if DEBUG:
            warn = tracker.verify_reward(reward_f)
            if warn:
                reward_mismatches += 1
                if reward_mismatches <= 5:
                    print(f"  [DEBUG] {warn}")

        if reward_f > 0:
            positive_count += 1
        elif reward_f < 0:
            negative_count += 1
        else:
            zero_count += 1

        if terminated or truncated:
            episodes += 1
            episode_returns.append(ep_return)

            # Snapshot agent_0 counters for this episode
            s = tracker.summary_dict()
            total_involved_started_0 += s["involved_started_0"]
            total_finished_0 += s["finished_0"]
            total_published_0 += s["published_0"]
            total_rejected_0 += s["rejected_0"]
            total_open_end_0 += s["open_end_0"]
            total_due_0 += s["due_0"]

            # Global env_metrics (context only)
            em = info.get("env_metrics", {})
            for k, v in em.items():
                if isinstance(v, (int, float)):
                    acc_global[k] = acc_global.get(k, 0.0) + float(v)

            print(f"  ep {episodes:>3d}  |  return={ep_return:7.3f}  |  "
                  f"inv_start_0={s['involved_started_0']}  "
                  f"pub_0={s['published_0']}  "
                  f"rej_0={s['rejected_0']}  "
                  f"open_0={s['open_end_0']}")

            ep_return = 0.0
            obs, info = env.reset()
            tracker.reset()
            tracker.snapshot_reset()

    # ══════════════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════════════
    R = np.array(all_rewards)
    p = positive_count / max(1, TOTAL_STEPS)

    print()
    print("#" * 65)
    print(f"  SUMMARY  ({TOTAL_STEPS} steps, {episodes} episodes)")
    print(f"  All metrics below are agent_0-only unless marked GLOBAL.")
    print("#" * 65)

    # ── Reward sparsity (agent_0) ─────────────────────────────────────
    print("\n  ── Reward sparsity (agent_0) ──")
    print(f"    reward > 0  : {positive_count:>6}  ({100*p:.2f}%)")
    print(f"    reward == 0 : {zero_count:>6}  ({100*zero_count/TOTAL_STEPS:.2f}%)")
    print(f"    reward < 0  : {negative_count:>6}  ({100*negative_count/TOTAL_STEPS:.2f}%)")
    print(f"    sum         : {R.sum():.4f}")
    print(f"    mean/step   : {R.mean():.6f}")
    print(f"    std/step    : {R.std():.6f}")
    print(f"    max single  : {R.max():.6f}")
    if DEBUG and reward_mismatches > 0:
        print(f"    ⚠ reward mismatches: {reward_mismatches}")

    # ── Episode returns (agent_0) ─────────────────────────────────────
    if episode_returns:
        E = np.array(episode_returns)
        print("\n  ── Episode returns (agent_0) ──")
        print(f"    n_episodes  : {episodes}")
        print(f"    mean        : {E.mean():.4f}")
        print(f"    median      : {np.median(E):.4f}")
        print(f"    std         : {E.std():.4f}")
        print(f"    max         : {E.max():.4f}")
        print(f"    min         : {E.min():.4f}")

    # ── Project pipeline (agent_0 only) ───────────────────────────────
    #    "involved_started" = agent_0 is in contributors (not necessarily initiator)
    #    "due" = finished (env has no separate evaluation phase)
    print("\n  ── Project pipeline (agent_0 only) ──")
    print(f"    involved_started_0 : {total_involved_started_0}")
    print(f"    due_0 (=finished)  : {total_due_0}")
    print(f"    finished_0         : {total_finished_0}")
    print(f"    published_0        : {total_published_0}")
    print(f"    rejected_0         : {total_rejected_0}")
    print(f"    open_end_0         : {total_open_end_0}")

    # ── Conditional probabilities (agent_0 only) ──────────────────────
    print("\n  ── Conditional probabilities (agent_0 only) ──")
    print(f"    P(reward>0 | step)                    = {_ratio(positive_count, TOTAL_STEPS)}")
    if p > 0:
        print(f"    E[T] = 1/p                            = {1/p:.0f} steps between rewards")
    else:
        print(f"    E[T] = 1/p                            = inf  (never got reward)")
    print(f"    expected pos. per batch                = {BATCH_SIZE * p:.1f}  (batch={BATCH_SIZE})")
    print(f"    P(published_0 | finished_0)            = {_ratio(total_published_0, total_finished_0)}")
    print(f"    P(finished_0  | involved_started_0)    = {_ratio(total_finished_0, total_involved_started_0)}")
    print(f"    P(due_0       | involved_started_0)    = {_ratio(total_due_0, total_involved_started_0)}")
    print(f"    P(involved_started_0 | step)           = {_ratio(total_involved_started_0, TOTAL_STEPS)}")
    print(f"    P(open_end_0  | involved_started_0)    = {_ratio(total_open_end_0, total_involved_started_0)}")

    # ── GLOBAL env_metrics (context only, NOT used for agent_0 analysis)
    if acc_global and episodes > 0:
        print("\n  ── GLOBAL env_metrics / episode (context only) ──")
        for k in sorted(acc_global):
            print(f"    {k:40s}: {acc_global[k]/episodes:.2f}")

    print()
    print("#" * 65)


if __name__ == "__main__":
    main()

