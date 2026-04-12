"""
rllib_single_agent_wrapper.py

Single-agent Gymnasium wrapper around a PettingZoo ParallelEnv (PeerGroupEnvironment) that:

1) Flattens nested observations (including action_mask) into a single 1D Box(float32)
   -> RLlib can use standard encoders.

2) Flattens the env's Dict action-space into ONE Discrete macro-action
   -> PPO can output a simple categorical distribution.

3) Decodes the macro-action back into the env's expected dict action:
   {"choose_project": int, "collaborate_with": np.ndarray[int8], "put_effort": int}

4) Repairs (clips) invalid decoded actions using the env-provided action_mask
   -> prevents invalid actions from breaking env logic.

5) Supports fixed policies for non-controlled agents via `other_policies`.
   `other_policies[agent_id]` must accept the nested obs:
   {"observation": ..., "action_mask": ...} and return an env-valid action dict.

6) (Optional ablation) Can enforce a top-k constraint on the "collaborate_with" bit-vector
   using a score computed from each agent's observation (peer reputation, distance,
   same-group bonus). This is fully implemented inside the wrapper and does NOT
   modify the underlying environment.

IMPORTANT:
- This macro-action approach is only feasible for SMALL max_peer_group_size (e.g. <= 12).
  Because ACTION_N = CP * PE * 2^CB.
- If you set max_peer_group_size=100, this is impossible. Then you need a multi-head model.

Key fixes vs your version:
- BUGFIX: non-controlled agents must NOT sample env.action_space() (it's a Dict). Sample wrapper.action_space instead.
- More robust active_agents selection: use observation keys.
- Stable flattening using a template built from observation_space, avoids feature-order/length drift across steps.
- Safer NumPy masking assignment to avoid view/copy surprises.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

import logging
import sys
import os

from src.agent_policies import do_nothing_policy

logger = logging.getLogger(__name__)


NestedObs = Dict[str, Any]
ActionDict = Dict[str, Any]


class RLLibSingleAgentWrapper(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        env,
        controlled_agent: Optional[Union[str, Callable[[Dict[str, Any]], str]]] = None,
        other_policies: Optional[Dict[str, Callable[[Any], Any]]] = None,
        *,
        force_episode_horizon: Optional[int] = None,
        strict_space_check: bool = False,
        # Optional top-k collaboration ablation
        topk_collab: Optional[int] = None,
        topk_mode: str = "score",
        topk_seed: int = 0,
        topk_apply_to_all_agents: bool = True,
        w_rep: float = 1.0,
        w_dist: float = 1.0,
        w_same: float = 0.5,
        info_action: bool = False,
        info_interval: int = 50,
        is_evaluation: bool = False,
        debug_effort: bool = False,
        debug_effort_agent_only: bool = True,
    ):
        """Wrapper around a multi-agent env exposing a single-agent RLlib interface.

        Args:
            env: PettingZoo ParallelEnv (your PeerGroupEnvironment)
            controlled_agent: None -> pick first agent each reset,
                              str -> fixed agent id,
                              callable -> choose agent given observations dict.
            other_policies: dict(agent_id -> callable(nested_obs)-> action_dict)
                           For non-controlled agents.
            force_episode_horizon: if set, wrapper truncates after this many steps
                                  (guarantees RLlib gets episodes).
            strict_space_check: if True, run a runtime space.contains() check on reset/step observations
                               (useful for debugging; can be slow).
            topk_collab: If set to an int k, enforce that at most k collaboration
                partners are selected per step, based on a score derived from the
                agent's observation. If None, no top-k ablation is applied.
            topk_mode: Reserved for future modes. Currently only "score" is
                supported, which uses reputation, distance, and same-group bonus.
            topk_seed: Seed used for deterministic tie-breaking when multiple
                peers share the same score.
            topk_apply_to_all_agents: If True, apply the top-k constraint also
                to non-controlled agents (both policy-driven and random actions).
            w_rep / w_dist / w_same: Weights for the reputation, distance, and
                same-group components of the collaboration score.
        """
        super().__init__()
        self.env = env
        self._choose_controlled = controlled_agent
        self.other_policies = other_policies or {}
        self.strict_space_check = bool(strict_space_check)

        # Top-k ablation config
        self.topk_collab = topk_collab
        self.topk_mode = topk_mode
        self.topk_seed = int(topk_seed)
        self.topk_apply_to_all_agents = bool(topk_apply_to_all_agents)
        self.w_rep = float(w_rep)
        self.w_dist = float(w_dist)
        self.w_same = float(w_same)
        self.info_action = bool(info_action)
        self.info_interval = max(1, int(info_interval))
        self.is_evaluation = bool(is_evaluation)
        self.debug_effort = bool(debug_effort)
        self.debug_effort_agent_only = bool(debug_effort_agent_only)

        # Logging / tracking state for controlled agent (agent_0)
        self._effort_total_count = 0
        self._effort_invalid_count = 0
        self._effort_valid_count = 0
        
        # Simple debug counters (only used if top-k is enabled)
        self._topk_calls = 0
        self._topk_pruned = 0
        self._topk_selected_count_sum = 0

        self.current_controlled: Optional[str] = None
        self._last_observations: Dict[str, Any] = {}

        # Horizon enforcement
        if force_episode_horizon is None and hasattr(env, "n_steps"):
            self._force_horizon = int(getattr(env, "n_steps"))
            logger.debug("No force_episode_horizon provided; using env.n_steps=%s", self._force_horizon)
        else:
            self._force_horizon = force_episode_horizon

        self._t = 0

        # Choose a stable reference agent
        self._ref_agent = getattr(env, "possible_agents", [None])[0]
        if self._ref_agent is None:
            raise ValueError("env.possible_agents is missing/empty; cannot select ref agent.")

        # ---- Multi-head action-space (Dict) ----
        # These attributes exist in your PeerGroupEnvironment
        self._CP = int(self.env.n_projects_per_step + 1)      # choose_project
        self._PE = int(self.env.max_projects_per_agent + 1)   # put_effort
        self._CB = int(self.env.max_peer_group_size)          # collaborate bits (max_peer_slots)

        # Modeling discrete components as truly discrete to avoid continuous-to-discrete mismatch.
        # This prevents the RL agent from getting 'stuck' at intermediate values (like 3)
        # during deterministic evaluation, as the policy will now output discrete probabilities.
        self.action_space = gym.spaces.Dict({
            "choose_project": gym.spaces.Discrete(self._CP),
            "put_effort": gym.spaces.Discrete(self._PE),
            "collaborate_with": gym.spaces.Box(0, 1, shape=(self._CB,), dtype=np.float32)
        })

        # Internal state for slot mapping (identity mapping as requested)
        self._slot_to_peer_index = list(range(self._CB))
        
        # Compatibility settings
        self._expected_obs_size = getattr(env, "expected_obs_size", None)
        # If passed via other_policies for some reason (older creator)
        if self._expected_obs_size is None and isinstance(other_policies, dict) and "expected_obs_size" in other_policies:
             self._expected_obs_size = other_policies["expected_obs_size"]

        # ---- Build stable observation template from env.observation_space ----
        # We need the env's observation_space(agent) which corresponds to the "observation" part.
        # The nested obs is {"observation": <that dict>, "action_mask": <mask dict>}.
        obs_space = None
        try:
            obs_space = self.env.observation_space(self._ref_agent)
        except Exception:
            # Some envs only build stable spaces after reset
            try:
                self.env.reset(seed=0)
                obs_space = self.env.observation_space(self._ref_agent)
            except Exception:
                obs_space = None

        self._env_obs_space = obs_space  # may be None
        
        # Determine if we should include masks in the flattened vector (compatibility flag)
        # 258 is a known size for older versions (max_peer=10, no peer_valid_mask, potentially no action_mask?)
        # Let's check if we can detect the expected size from env_config or similar.
        self._include_peer_valid_mask = True
        self._include_action_mask = True
        
        # Build deterministic templates for stable flattening
        if obs_space is not None:
            self._obs_template = self._zeros_from_space(obs_space)  # matches "observation" dict
            self._mask_template = {
                "choose_project": np.zeros(self.env.n_projects_per_step + 1, dtype=np.int8),
                "collaborate_with": np.zeros(self.env.max_peer_group_size, dtype=np.int8),
                "put_effort": np.zeros(self.env.max_projects_per_agent + 1, dtype=np.int8),
            }
            sample_vec = self._flatten_to_vector({"observation": self._obs_template, "action_mask": self._mask_template})
        else:
            # Fallback: derive vector size from runtime reset
            observations, infos = self.env.reset(seed=0)
            sample_agent = self._ref_agent if self._ref_agent in observations else next(iter(observations.keys()))
            # With no declared space, we cannot build a template; will be less stable.
            # We still create a best-effort template from this sample.
            nested = observations[sample_agent]
            if not (isinstance(nested, dict) and "observation" in nested and "action_mask" in nested):
                raise TypeError(
                    "Env must return nested obs {'observation':..., 'action_mask':...} "
                    "for this wrapper to work."
                )
            self._obs_template = self._deep_copy_numeric(nested.get("observation", {}))
            self._mask_template = self._deep_copy_numeric(nested.get("action_mask", {}))
            sample_vec = self._flatten_to_vector({"observation": self._obs_template, "action_mask": self._mask_template})

        # AUTO-ADAPT if env_config suggests a specific expected size (e.g. 258)
        # We check both the passed __init__ arguments and the environment's config.
        # This is a bit of a hack but helps with backward compatibility.
        expected_size = self._expected_obs_size
        
        if expected_size:
            if expected_size == 287 and sample_vec.size == 297:
                print(f"[COMPAT] Detected expected_obs_size=287. Disabling peer_valid_mask to match.")
                self._include_peer_valid_mask = False
                sample_vec = self._flatten_to_vector({"observation": self._obs_template, "action_mask": self._mask_template})
            elif expected_size == 258 and sample_vec.size == 297:
                print(f"[COMPAT] Detected expected_obs_size=258. Attempting to match by disabling masks.")
                self._include_peer_valid_mask = False
                # Try disabling action mask too if 258 is even smaller
                sample_vec = self._flatten_to_vector({"observation": self._obs_template, "action_mask": self._mask_template})
                if sample_vec.size > expected_size:
                     self._include_action_mask = False
                     sample_vec = self._flatten_to_vector({"observation": self._obs_template, "action_mask": self._mask_template})
            elif expected_size == 268 and sample_vec.size == 297:
                print(f"[COMPAT] Detected expected_obs_size=268. Disabling masks to match.")
                self._include_peer_valid_mask = False
                self._include_action_mask = False
                sample_vec = self._flatten_to_vector({"observation": self._obs_template, "action_mask": self._mask_template})
            elif expected_size < sample_vec.size:
                # Last resort: truncate if we still don't match after disabling masks
                # (This is dangerous but might work for tiny mismatches)
                print(f"[COMPAT] Warning: expected_obs_size={expected_size} < current_size={sample_vec.size}. Will truncate during flattening.")

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(sample_vec.size),),
            dtype=np.float32,
        )
        
        # DEBUG LOGGING (helpful for size mismatch troubleshooting)
        # print(f"[DEBUG_WRAPPER] Observation Space initialized (size={sample_vec.size}):")
        # print(f"  - mpgs={self.env.max_peer_group_size}, mppa={self.env.max_projects_per_agent}, nps={self.env.n_projects_per_step}")
        # print(f"  - observation (nested): {self._flatten_any_like_template(self._obs_template, self._obs_template).size}")
        # print(f"  - action_mask: {self._flatten_mask_like_template(self._mask_template, self._mask_template).size}")
        # print(f"  - peer_valid_mask: {self._CB}")
        import sys
        sys.stdout.flush()

        # Don't overwrite last observations here; caller will call reset() before stepping.
        self._last_observations = {}
        self.current_controlled = self._ref_agent
        
    # Remove debug prints and injected info

    # -----------------------------
    # Action encoding/decoding
    # -----------------------------

    def _decode_action(self, a: Any, agent_id: Optional[str] = None) -> ActionDict:
        """Pass-through or convert action into env dict format.
        
        Note on 'collaborate_with': 
        - The raw action from the policy contains 'intents' (bits for each peer slot).
        - These are decoded here (n_requested_collaborators).
        - Subsequently, _apply_action_mask() removes bits for invalid/inactive slots.
        - Finally, the environment only starts a project if BOTH agents select it 
          AND have mutual collaboration bits set.
        
        Supports:
        1. Dict actions (preferred): {'choose_project': int, 'put_effort': int, 'collaborate_with': ndarray}
        2. Tuple/List actions (legacy/RLlib internal): [CP, PE, CB_bits]
        3. Integer actions (legacy/fallback): decoded using modulo/bitmasking
        """
        
        # 1. Dictionary format (Matches our new action_space)
        if isinstance(a, dict):
            # Already in dict format or close to it
            choose_project = int(a.get("choose_project", 0))
            put_effort = int(a.get("put_effort", 0))
            collab_bits = np.asarray(a.get("collaborate_with", np.zeros(self._CB, dtype=np.float32)), dtype=np.float32)
            # Threshold if using Box (which might be continuous)
            collab_bits = (collab_bits > 0.5).astype(np.int8)

            do_debug = self.debug_effort
            # Robust debug logic: check if the agent matches the one we want to control
            # or matches the currently selected controlled agent.
            target = self.current_controlled or str(self._choose_controlled)
            if self.debug_effort_agent_only and agent_id is not None and agent_id != target:
                do_debug = False

            if do_debug:
                print(f"[DEBUG_EFFORT][Single Agent Wrapper] _decode_action (Dict) for {agent_id or 'unknown'}:")
                print(f"  - choose_project: {choose_project}")
                print(f"  - put_effort: {put_effort}")
                n_requested = int(np.sum(collab_bits))
                print(f"  - n_requested_collaborators: {n_requested} (Intent before masking)")
                import sys
                sys.stdout.flush()

            return {
                "choose_project": choose_project,
                "put_effort": put_effort,
                "collaborate_with": collab_bits,
            }

        # 2. Box-based actions (Legacy/Fallback for older checkpoints or manual testing)
        # Note: We keep this for backward compatibility if we load an old model,
        # but the primary path will be via Dict.
        if isinstance(a, np.ndarray) and a.ndim == 1 and len(a) == (self._CB + 2):
            raw_cp = a[0]
            raw_pe = a[1]
            raw_cb = a[2:]
            
            choose_project = int(np.round(raw_cp))
            put_effort = int(np.round(raw_pe))
            collab_bits = np.round(raw_cb).astype(np.int8)
            
            do_debug = self.debug_effort
            target = self.current_controlled or str(self._choose_controlled)
            if self.debug_effort_agent_only and agent_id is not None and agent_id != target:
                do_debug = False

            if do_debug:
                print(f"[DEBUG_EFFORT][Single Agent Wrapper] _decode_action (Box/Legacy) for {agent_id or 'unknown'}:")
                print(f"  - raw_effort: {raw_pe:.4f} -> {put_effort}")
                print(f"  - raw_choose_project: {raw_cp:.4f} -> {choose_project}")
                n_requested = int(np.sum(collab_bits))
                print(f"  - n_requested_collaborators: {n_requested} (Intent before masking)")
                import sys
                sys.stdout.flush()

            return {
                "choose_project": choose_project,
                "put_effort": put_effort,
                "collaborate_with": collab_bits,
            }

        # 3. Tuple/List actions (Sometimes returned by RLlib's older internal mechanisms)
        if isinstance(a, (tuple, list)) and len(a) >= 3:
            return {
                "choose_project": int(a[0]),
                "put_effort": int(a[1]),
                "collaborate_with": np.asarray(a[2], dtype=np.int8),
            }
        
        # 3b. Numpy 1D array actions treated like tuple/list (len >= 3)
        if isinstance(a, np.ndarray) and a.ndim == 1 and len(a) >= 3:
            return {
                "choose_project": int(a[0]),
                "put_effort": int(a[1]),
                "collaborate_with": np.asarray(a[2], dtype=np.int8),
            }
        
        if isinstance(a, dict):
            # Already in dict format from our older action_space
            return {
                "choose_project": int(a.get("choose_project", 0)),
                "collaborate_with": np.asarray(a.get("collaborate_with", np.zeros(self._CB, dtype=np.int8)), dtype=np.int8),
                "put_effort": int(a.get("put_effort", 0)),
            }
        
        # 4. Numpy scalar or length-1 array representing a discrete action id
        if isinstance(a, np.ndarray) and (a.shape == () or a.size == 1):
            a = a.item()
        
        # Fallback for old integer actions (if any remain in tests/heuristics)
        a = int(a)
        collab_base = 1 << self._CB
        collab_code = a % collab_base
        a //= collab_base

        put_effort = a % self._PE
        a //= self._PE

        choose_project = a % self._CP

        collab_bits = np.array([(collab_code >> i) & 1 for i in range(self._CB)], dtype=np.int8)

        return {
            "choose_project": int(choose_project),
            "collaborate_with": collab_bits,
            "put_effort": int(put_effort),
        }

    def decode_action_id(self, action_id: Any, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Public API: decode an action into a human-readable dict.

        Now supports both dict actions and legacy integer IDs.
        """
        decoded = self._decode_action(action_id, agent_id=agent_id)
        collab = decoded["collaborate_with"]
        return {
            "choose_project": decoded["choose_project"],
            "put_effort": decoded["put_effort"],
            "collaborate_with": collab.tolist() if hasattr(collab, "tolist") else list(collab),
            "n_collaborators": int(np.sum(collab)),
        }

    def _apply_action_mask(self, decoded: ActionDict, nested_obs: NestedObs, agent_id: Optional[str] = None) -> ActionDict:
        """Repair invalid actions using the env-provided action_mask.

        Treat mask values >0 as allowed (your env sometimes uses 2).
        If top-k collaboration ablation is enabled, further prune the
        "collaborate_with" bit-vector to at most k partners using
        observation-derived scores.
        """
        mask = nested_obs.get("action_mask", {})
        if not isinstance(mask, dict):
            return decoded

        # choose_project
        cp_mask = np.asarray(mask.get("choose_project", []))
        if cp_mask.size:
            cp = int(decoded.get("choose_project", 0))
            is_agent = (agent_id == self.current_controlled)
            do_debug = self.debug_effort and (not self.debug_effort_agent_only or is_agent)
            
            if cp < 0 or cp >= cp_mask.size or cp_mask[cp] <= 0:
                old_cp = cp
                decoded["choose_project"] = 0
                if do_debug:
                    active_projects = self._get_active_projects_from_obs(nested_obs, agent_id=agent_id)
                    n_running = sum(1 for v in active_projects.values() if v is not None)
                    reason = ""
                    if old_cp > 0:
                        if n_running >= self.env.max_projects_per_agent:
                            reason = " (Reason: max projects limit reached)"
                        elif old_cp < cp_mask.size:
                            reason = " (Reason: project opportunity stochastic masking)"
                        else:
                            reason = f" (Reason: index {old_cp} out of range for n_projects_per_step {self.env.n_projects_per_step})"
                    print(f"[DEBUG_EFFORT][Single Agent Wrapper] _apply_action_mask for {agent_id or 'unknown'}: choose_project {old_cp} -> 0 (masked). Active projects: {n_running}/{self.env.max_projects_per_agent}{reason}")
            elif do_debug:
                active_projects = self._get_active_projects_from_obs(nested_obs, agent_id=agent_id)
                n_running = sum(1 for v in active_projects.values() if v is not None)
                print(f"[DEBUG_EFFORT][Single Agent Wrapper] _apply_action_mask for {agent_id or 'unknown'}: choose_project {cp} is ALLOWED. Active projects: {n_running}/{self.env.max_projects_per_agent}")

        # put_effort
        pe_mask = np.asarray(mask.get("put_effort", []))
        if pe_mask.size:
            pe = int(decoded.get("put_effort", 0))
            is_agent = (agent_id == self.current_controlled)
            do_debug = self.debug_effort and (not self.debug_effort_agent_only or is_agent)
            
            if is_agent:
                self._effort_total_count += 1
            
            if pe < 0 or pe >= pe_mask.size or pe_mask[pe] <= 0:
                old_pe = pe
                decoded["put_effort"] = 0
                
                if is_agent:
                    self._effort_invalid_count += 1
                
                if do_debug:
                    print(f"[DEBUG_EFFORT][Single Agent Wrapper] _apply_action_mask for {agent_id or 'unknown'}: put_effort {old_pe} -> 0 (masked)")
            else:
                if is_agent:
                    self._effort_valid_count += 1
                
                if do_debug:
                    print(f"[DEBUG_EFFORT][Single Agent Wrapper] _apply_action_mask for {agent_id or 'unknown'}: put_effort {pe} is ALLOWED")
            
            if do_debug:
                running = self._get_active_projects_from_obs(nested_obs, agent_id=agent_id)
                if not any(v is not None for v in running.values()):
                    print(f"  - No active projects to work on.")
                else:
                    # In put_effort 0 is "no effort", 1..N maps to slots in running_projects
                    # The slots are project_0, project_1, etc. and correspond to 1..N
                    for i in range(self.env.max_projects_per_agent):
                        pid_str = f"project_{i}"
                        pinfo = running.get(pid_str)
                        if pinfo is not None:
                            eff = float(pinfo.get("current_effort", [0])[0])
                            req = float(pinfo.get("required_effort", [0])[0])
                            
                            # Extract time_left (deadline)
                            time_left = 0
                            if "time_left" in pinfo:
                                tl = pinfo["time_left"]
                                if isinstance(tl, (list, np.ndarray)) and len(tl) > 0:
                                    time_left = int(tl[0])
                                else:
                                    time_left = int(tl)
                            
                            # Extract internal project_id for duplicate detection
                            # PeerGroupEnvironment often puts it in the dict or we can try to find it
                            internal_id = "unknown"
                            if "project_id" in pinfo:
                                internal_id = pinfo["project_id"]
                            elif "id" in pinfo:
                                internal_id = pinfo["id"]
                            
                            mark = " <--- SELECTED" if pe == i + 1 else ""
                            print(f"  - Slot {i+1} [Project {pid_str} | ID: {internal_id}]: effort={eff:.2f}/{req:.2f}, deadline={time_left}{mark}")
                        else:
                            mark = " <--- SELECTED" if pe == i + 1 else ""
                            # If SELECTED but empty, it's actually masked anyway by pe_mask[pe] <= 0
                            print(f"  - Slot {i+1} [Empty]:{mark}")
                            
                    if pe == 0:
                        print(f"  - Slot 0 [No project]: SELECTED")

        # collaborate_with
        c_mask = np.asarray(mask.get("collaborate_with", []))
        if c_mask.size:
            c = decoded.get("collaborate_with")
            
            # Ensure 'c' is a 1D numpy array
            if isinstance(c, (int, np.integer)):
                # If it's a single integer, convert to bit-vector
                c = np.array([(c >> i) & 1 for i in range(self._CB)], dtype=np.int8)
            elif not isinstance(c, np.ndarray):
                c = np.asarray(c if c is not None else np.zeros(self._CB, dtype=np.int8), dtype=np.int8)
            else:
                c = c.copy().astype(np.int8)

            allowed = (c_mask > 0)

            # Safety check: if c is somehow still 0-d, make it 1-d
            if c.ndim == 0:
                c = c.reshape(1)
            
            # Ensure it matches _CB length (PeerGroupEnvironment expects consistent shapes)
            if len(c) < self._CB:
                new_c = np.zeros(self._CB, dtype=np.int8)
                new_c[:len(c)] = c
                c = new_c
            elif len(c) > self._CB:
                c = c[:self._CB]

            L = min(len(c), len(allowed))
            c_slice = c[:L]
            pruned_count = np.sum(c_slice[~allowed[:L]])
            c_slice[~allowed[:L]] = 0
            c[:L] = c_slice

            if self.debug_effort and (not self.debug_effort_agent_only or agent_id == self.current_controlled):
                n_collab = int(np.sum(c))
                print(f"[DEBUG_EFFORT][Single Agent Wrapper] _apply_action_mask for {agent_id or 'unknown'}: collaborate_with masked {pruned_count} bits, {n_collab} remain")
                if pruned_count > 0:
                    # In _apply_action_mask, nested_obs has {"observation": {...}, "action_mask": {...}}
                    obs_dict = nested_obs.get("observation", {}) if isinstance(nested_obs, dict) else {}
                    if not obs_dict and isinstance(nested_obs, dict) and "peer_group" in nested_obs:
                        obs_dict = nested_obs

                    peer_group = np.atleast_1d(obs_dict.get("peer_group", [])) # array of 1 (active) or 0 (inactive)
                    # peer_group from observation is a binary array (1=active, 0=inactive/empty)
                    # c_mask also has >0 for active, 0 for inactive/empty
                    
                    # We can't easily distinguish 'EMPTY' from 'INACTIVE' just from peer_group,
                    # as both are 0. But we know that slots up to the actual number of peers
                    # in the group are 'INACTIVE' if not active, and slots beyond that are 'EMPTY'.
                    # However, the environment's peer_group in obs is already a bit-mask.
                    
                    # To be more precise, we'd need the actual peer IDs or group size.
                    # In PeerGroupEnvironment, peer_group in obs is indeed just the bitmask of active agents.
                    
                    inactive_indices = np.where((c_mask[:L] == 0) & (decoded.get("collaborate_with")[:L] > 0))[0]
                    if len(inactive_indices) > 0:
                        # Let's try to find out if it's a padding slot or an inactive agent
                        # The environment doesn't provide the raw peer_group IDs in the observation, 
                        # only the bitmask of active ones.
                        print(f"  - Masked reasons: {len(inactive_indices)} peer slots are INACTIVE (agent exists but not active) or EMPTY (no agent in this slot)")
                        # for idx in inactive_indices:
                        #     print(f"    * Slot {idx}: Masked (Agent not active or slot empty)")
                    
                    # Also explain 2 vs 0 in c_mask
                    active_peers = np.where(c_mask > 0)[0]
                    if len(active_peers) == 0:
                        print(f"  - NO peers are currently active/available for collaboration.")

            if len(c) > L:
                c[L:] = 0

            # Optional top-k ablation (after mask repair)
            if self.topk_collab is not None and self.topk_collab >= 0:
                # Only apply to controlled agent or optionally to all agents
                if agent_id is None or self.topk_apply_to_all_agents:
                    c = self._apply_topk_collaboration(
                        c,
                        c_mask,
                        nested_obs,
                        agent_id=agent_id,
                        k=self.topk_collab,
                    )

            decoded["collaborate_with"] = c.astype(np.int8, copy=False)

        return decoded

    def _get_active_projects_from_obs(self, nested_obs: NestedObs, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Extract active projects from nested observation, including empty slots.
        
        Returns a dictionary mapping 'project_{i}' keys (0..max_projects-1) to their project info
        or None if the slot is empty.
        """
        obs_dict = nested_obs.get("observation", {}) if isinstance(nested_obs, dict) else {}
        
        if not obs_dict and isinstance(nested_obs, dict):
            if "running_projects" in nested_obs or "peer_group" in nested_obs:
                obs_dict = nested_obs
        
        running = obs_dict.get("running_projects", {})
        if not isinstance(running, dict):
            return {f"project_{i}": None for i in range(self.env.max_projects_per_agent)}
        
        # Determine the agent index to access the environment's ground truth for slot mapping
        target = agent_id or self.current_controlled or str(self._choose_controlled)
        agent_idx = self.env.agent_to_id.get(target)
        
        all_slots = {}
        for i in range(self.env.max_projects_per_agent):
            key = f"project_{i}"
            
            # Ground truth: What project ID is in this slot?
            p_id = None
            if agent_idx is not None:
                p_id = self.env.agent_active_projects[agent_idx][i]
            
            # Try to find the project in the observation.
            # 1. Try by slot key (project_0, project_1...) as per observation_space
            v = running.get(key)
            
            # 2. Try by project ID (project_{p_id}) as per _get_running_projects_obs
            if v is None and p_id is not None:
                v = running.get(f"project_{p_id}")
            
            is_active = False
            if isinstance(v, dict):
                # Check for either required_effort or prestige (both present in valid obs)
                if "required_effort" in v or "prestige" in v:
                    re = v.get("required_effort", v.get("prestige", [0]))
                    if isinstance(re, (list, np.ndarray)) and len(re) > 0:
                        val = float(re[0])
                    else:
                        val = float(re)
                    if val > 0:
                        is_active = True
            
            if is_active:
                # Add the project ID to the info dict for easier debugging/identification
                # if it's not already there.
                if isinstance(v, dict):
                    if "project_id" not in v and "id" not in v:
                        v["project_id"] = p_id if p_id is not None else "unknown"
                all_slots[key] = v
            else:
                all_slots[key] = None
        return all_slots

    # -----------------------------
    # Top-k collaboration helpers
    # -----------------------------

    def _extract_peer_feature_array(self, obs: Any, candidate_keys: Iterable[str], length: int, default_value: Any = 0.0) -> Optional[np.ndarray]:
        """Search nested obs dict for a 1D array under any of candidate_keys.

        The observation structure may change; we try a flexible search:
        - Direct dict lookup by key
        - One-level nested dict lookup
        If not found, return a constant array of default_value.
        If default_value is None, return None if not found.
        The result is always float32 length `length`, padded/clipped as needed.
        """
        if not isinstance(obs, dict):
            if default_value is None: return None
            return np.full((length,), float(default_value), dtype=np.float32)

        found = None

        # Direct lookup
        for key in candidate_keys:
            if key in obs:
                found = obs[key]
                break

        # One-level nested lookup
        if found is None:
            for v in obs.values():
                if isinstance(v, dict):
                    for key in candidate_keys:
                        if key in v:
                            found = v[key]
                            break
                if found is not None:
                    break

        if found is None:
            if default_value is None: return None
            return np.full((length,), float(default_value), dtype=np.float32)

        arr = np.asarray(found, dtype=np.float32).ravel()
        if arr.size < length:
            if default_value is None:
                # If we need a specific length but only found a shorter array, 
                # we pad with 0.0 even if default_value is None, to avoid partial failure.
                pad = np.zeros((length - arr.size,), dtype=np.float32)
            else:
                pad = np.full((length - arr.size,), float(default_value), dtype=np.float32)
            arr = np.concatenate([arr, pad])
        elif arr.size > length:
            arr = arr[:length]
        return arr

    def _extract_agent_group_id(self, obs: Any) -> Optional[int]:
        """Extract the current agent's own group id from nested observation.

        Looks for common key patterns like 'group_id', 'my_group', 'own_group'.
        Returns None if not found.
        """
        if not isinstance(obs, dict):
            return None

        own_keys = ["group_id", "my_group", "own_group"]

        for key in own_keys:
            if key in obs:
                try:
                    return int(np.asarray(obs[key]).item())
                except Exception:
                    continue

        for v in obs.values():
            if isinstance(v, dict):
                for key in own_keys:
                    if key in v:
                        try:
                            return int(np.asarray(v[key]).item())
                        except Exception:
                            continue

        return None

    def _extract_peer_group_ids(self, obs: Any, length: int) -> Optional[np.ndarray]:
        """Extract peer group IDs array (length `length`) or None if not found."""
        if not isinstance(obs, dict):
            return None

        peer_keys = ["peer_group", "peer_groups", "group_ids", "groups"]

        found = None
        for key in peer_keys:
            if key in obs:
                found = obs[key]
                break

        if found is None:
            for v in obs.values():
                if isinstance(v, dict):
                    for key in peer_keys:
                        if key in v:
                            found = v[key]
                            break
                if found is not None:
                    break

        if found is None:
            return None

        arr = np.asarray(found, dtype=np.int32).ravel()
        if arr.size < length:
            pad = np.full((length - arr.size,), -1, dtype=np.int32)
            arr = np.concatenate([arr, pad])
        elif arr.size > length:
            arr = arr[:length]
        return arr

    def _compute_peer_scores(self, nested_obs: NestedObs, agent_id: Optional[str], k: int, seed: int) -> np.ndarray:
        """Compute per-peer collaboration scores of length self._CB.

        Components (if available):
          - Reputation: higher is better.
          - Distance: smaller is better.
          - Same-group bonus: +w_same if peer is in same group as agent.

        All components are min-max normalized (with epsilon) before combining:
          score = w_rep * norm_rep - w_dist * norm_dist + w_same * same_group_bonus
        """
        if k <= 0 or self._CB <= 0:
            return np.zeros((self._CB,), dtype=np.float32)

        obs = nested_obs.get("observation", {})

        # Reputation features
        rep_keys = ["peer_reputation", "peer_reputations", "peers_reputation", "reputation_peers"]
        rep = self._extract_peer_feature_array(obs, rep_keys, self._CB, default_value=0.0)

        # Distance features (smaller is better)
        # We try to find precomputed distances or compute them from centroids
        dist_keys = ["peer_distance", "peer_distances", "distance_to_peers", "peers_distance"]
        dist = self._extract_peer_feature_array(obs, dist_keys, self._CB, default_value=None)

        if dist is None:
            # Fallback: compute Euclidean distance from centroids if available
            p_centroids = self._extract_peer_feature_array(obs, ["peer_centroids"], self._CB * 2, default_value=0.0)
            s_centroid = self._extract_peer_feature_array(obs, ["self_centroid", "self_centroids"], 2, default_value=0.0)
            
            p_centroids = p_centroids.reshape(self._CB, 2)
            s_centroid = s_centroid.reshape(1, 2)
            
            # Distance: sqrt(sum((p - s)^2))
            dist = np.sqrt(np.sum((p_centroids - s_centroid)**2, axis=1)).astype(np.float32)

        # Same-group bonus
        own_gid = self._extract_agent_group_id(obs)
        peer_gids = self._extract_peer_group_ids(obs, self._CB)
        if own_gid is not None and peer_gids is not None:
            same_group = (peer_gids == own_gid).astype(np.float32)
        else:
            same_group = np.zeros((self._CB,), dtype=np.float32)

        def _min_max_normalize(x: np.ndarray) -> np.ndarray:
            if x.size == 0:
                return x.astype(np.float32, copy=False)
            x = x.astype(np.float32, copy=False)
            x_min = float(np.min(x))
            x_max = float(np.max(x))
            if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max - x_min < 1e-8:
                return np.zeros_like(x, dtype=np.float32)
            return (x - x_min) / (x_max - x_min + 1e-8)

        rep_n = _min_max_normalize(rep)
        dist_n = _min_max_normalize(dist)

        scores = (
            self.w_rep * rep_n
            - self.w_dist * dist_n
            + self.w_same * same_group
        ).astype(np.float32, copy=False)

        # Deterministic tie-breaking: tiny, seeded noise
        rng = np.random.RandomState(seed=self.topk_seed if seed is None else seed)
        noise = rng.uniform(low=-1e-6, high=1e-6, size=scores.shape).astype(np.float32)
        scores = scores + noise

        return scores

    def _apply_topk_collaboration(
        self,
        collab_bits: np.ndarray,
        collab_mask: np.ndarray,
        nested_obs: NestedObs,
        *,
        agent_id: Optional[str],
        k: int,
    ) -> np.ndarray:
        """Apply top-k pruning to the collaboration bit-vector.

        Only positions with collab_bits[i] == 1 and collab_mask[i] > 0 are
        considered candidates. If more than k candidates exist, keep only the
        k peers with highest score (deterministic tie-breaking). Otherwise
        return collab_bits unchanged (up to allowed-mask cleanup).
        """
        self._topk_calls += 1

        c = np.asarray(collab_bits, dtype=np.int8).copy()
        mask = np.asarray(collab_mask, dtype=np.int8)

        if c.size != self._CB:
            if c.size < self._CB:
                pad = np.zeros((self._CB - c.size,), dtype=np.int8)
                c = np.concatenate([c, pad])
            else:
                c = c[: self._CB]

        if mask.size < self._CB:
            pad = np.zeros((self._CB - mask.size,), dtype=np.int8)
            mask = np.concatenate([mask, pad])
        elif mask.size > self._CB:
            mask = mask[: self._CB]

        allowed = (mask > 0)
        c[~allowed] = 0

        candidates = np.where((c == 1) & allowed)[0]
        if k is None or k < 0 or candidates.size <= k:
            self._topk_selected_count_sum += int(candidates.size)
            return c

        scores = self._compute_peer_scores(nested_obs, agent_id, k, seed=self.topk_seed)
        # Select top-k among candidates: sort by score desc
        cand_scores = scores[candidates]
        order = np.argsort(-cand_scores, kind="mergesort")  # stable, deterministic
        keep_idx = candidates[order[:k]]

        new_c = np.zeros_like(c, dtype=np.int8)
        new_c[keep_idx] = 1

        if candidates.size > k:
            self._topk_pruned += 1
        self._topk_selected_count_sum += int(keep_idx.size)

        return new_c

    # -----------------------------
    # Observation flattening (stable)
    # -----------------------------

    def _flatten_to_vector(self, nested_obs: Any) -> np.ndarray:
        """
        Env returns per-agent nested obs:
          {"observation": <dict>, "action_mask": <dict>}
        We flatten both in a stable, template-driven order.
        We also append a peer_valid_mask indicating which slots contain real peers.
        """
        if not (isinstance(nested_obs, dict) and "observation" in nested_obs):
            raise TypeError("Expected nested obs: {'observation': ..., 'action_mask': ...}")

        obs_part = nested_obs.get("observation", {})
        mask_part = nested_obs.get("action_mask", {})

        obs_vec = self._flatten_any_like_template(obs_part, self._obs_template)
        
        parts = [obs_vec]
        
        if self._include_action_mask:
            mask_vec = self._flatten_mask_like_template(mask_part, self._mask_template)
            parts.append(mask_vec)

        if self._include_peer_valid_mask:
            # Build peer_valid_mask: 1 if peer_group[i] > 0 (or peer is present), 0 otherwise.
            # peer_group in PeerGroupEnvironment uses 0 for no agent (agents are 1-indexed in IDs or positive).
            peer_group = np.asarray(obs_part.get("peer_group", np.zeros(self._CB)))
            peer_valid_mask = (peer_group > 0).astype(np.float32)
            if peer_valid_mask.size < self._CB:
                pad = np.zeros(self._CB - peer_valid_mask.size, dtype=np.float32)
                peer_valid_mask = np.concatenate([peer_valid_mask, pad])
            elif peer_valid_mask.size > self._CB:
                peer_valid_mask = peer_valid_mask[:self._CB]
            parts.append(peer_valid_mask)

        # Compatibility check for older checkpoints that didn't have the peer_valid_mask.
        # If the observation size doesn't match, we might need to adjust it.
        # The observation_space is set in __init__ based on sample_vec.size.
        
        # Build the final flattened vector
        out = np.concatenate(parts).astype(np.float32, copy=False)
        
        # Compatibility truncation / padding if size mismatch was detected
        if self._expected_obs_size is not None:
             expected = self._expected_obs_size
             if out.size != expected:
                  if out.size > expected:
                       out = out[:expected]
                  else:
                       pad = np.zeros(expected - out.size, dtype=np.float32)
                       out = np.concatenate([out, pad])
        
        # Handle cases where the model expects a different size (if it was trained with an older version)
        # However, RLlib will usually complain during setup if there's a mismatch.
        
        return out

    def _flatten_any_like_template(self, x: Any, tmpl: Any) -> np.ndarray:
        """
        Flatten x using tmpl structure and ordering.
        Missing keys/items are replaced by zeros from tmpl.
        Extra keys in x are ignored (keeps feature vector stable).
        """
        # Template dict: recurse in deterministic (template) order.
        if isinstance(tmpl, dict):
            parts: List[np.ndarray] = []
            if not isinstance(x, dict):
                x = {}
            for k in sorted(tmpl.keys()):
                parts.append(self._flatten_any_like_template(x.get(k, tmpl[k]), tmpl[k]))
            return np.concatenate(parts) if parts else np.zeros((0,), dtype=np.float32)

        # Template array/scalar: flatten numeric
        arr = np.asarray(x)
        if arr.dtype == object:
            raise TypeError(f"Non-numeric object in observation: {type(x)}")
        return arr.astype(np.float32, copy=False).ravel()

    def _flatten_mask_like_template(self, mask: Any, tmpl_mask: Any) -> np.ndarray:
        """
        Flatten action_mask in template order. Convert to 0/1 float.
        """
        if isinstance(tmpl_mask, dict):
            parts: List[np.ndarray] = []
            if not isinstance(mask, dict):
                mask = {}
            for k in sorted(tmpl_mask.keys()):
                v = np.asarray(mask.get(k, tmpl_mask[k]))
                v01 = (v > 0).astype(np.float32).ravel()
                parts.append(v01)
            return np.concatenate(parts) if parts else np.zeros((0,), dtype=np.float32)

        v = np.asarray(mask)
        return (v > 0).astype(np.float32).ravel()

    # -----------------------------
    # Gymnasium API
    # -----------------------------

    def reset(self, *, seed=None, options=None):
        logger.debug("Wrapper.reset() called; delegating to env.reset(seed=%s)", seed)
        observations, infos = self.env.reset(seed=seed, options=options)
        self._last_observations = observations
        self._t = 0
        self._effort_total_count = 0
        self._effort_invalid_count = 0
        self._effort_valid_count = 0

        # RL Reproducibility: seed wrapper's own action_space so that
        # fallback sampling (for agents without policies) is deterministic
        if seed is not None:
            self.action_space.seed(seed)

        # choose controlled agent
        if callable(self._choose_controlled):
            self.current_controlled = self._choose_controlled(observations)
        elif isinstance(self._choose_controlled, str):
            self.current_controlled = self._choose_controlled
        else:
            self.current_controlled = next(iter(observations.keys()))

        if self.current_controlled not in observations:
            raise RuntimeError(f"controlled agent {self.current_controlled} not in env observations")

        nested = observations[self.current_controlled]
        obs_vec = self._flatten_to_vector(nested)
        info = infos.get(self.current_controlled, {})

        obs_vec = self._ensure_obs_vector_ok(obs_vec, where="reset")

        # Optional strict space check (debug)
        if self.strict_space_check and self._env_obs_space is not None:
            obs_part = nested.get("observation", {})
            if not self._env_obs_space.contains(obs_part):
                logger.warning("Env observation NOT contained in env.observation_space(%s) at reset()", self._ref_agent)

        logger.info("Episode START for %s (obs_len=%d, force_horizon=%s)", self.current_controlled, obs_vec.size, self._force_horizon)
        # print(f"[WRAPPER] Episode START for {self.current_controlled} (obs_len={obs_vec.size}, force_horizon={self._force_horizon})")
        return obs_vec, info

    def _summarize_obs(self, nested_obs: NestedObs) -> Dict[str, Any]:
        """Create a compact summary of the observation for logging."""
        obs = nested_obs.get("observation", {})
        
        # Peer group size
        peer_group = np.asarray(obs.get("peer_group", []))
        n_peers = int(np.sum(peer_group > 0))
        
        # Projects available
        opps = obs.get("project_opportunities", {})
        n_opps = len(opps)
        
        # Active projects
        running = obs.get("running_projects", {})
        n_running = len(running)
        
        # Self stats
        age = int(obs.get("age", [0])[0])
        acc_rew = float(obs.get("accumulated_rewards", [0])[0])
        
        return {
            "n_peers": n_peers,
            "n_opps": n_opps,
            "n_running": n_running,
            "age": age,
            "acc_rew": round(acc_rew, 2)
        }

    def step(self, action):
        if self.current_controlled is None:
            raise RuntimeError("Wrapper not reset() before step().")

        # Optional logging of the observation BEFORE taking the action
        if self.info_action and (self._t % self.info_interval == 0):
            try:
                nested_obs = self._last_observations.get(self.current_controlled, {})
                summary = self._summarize_obs(nested_obs)
                tag = "[EVAL]" if self.is_evaluation else "[TRAIN]"
                # Use t+1 for the upcoming step matching the action log
                print(f"{tag} [OBS] t={self._t + 1} summary={summary}")
            except Exception as e:
                tag = "[EVAL]" if self.is_evaluation else "[TRAIN]"
                print(f"{tag} [OBS] t={self._t + 1} (error: {e})")

        # Optional logging of the action taken by the controlled agent
        if self.info_action and (self._t % self.info_interval == 0):
            try:
                decoded = self.decode_action_id(action, agent_id=self.current_controlled)
                # Just print decoded to avoid multi-line array noise
                tag = "[EVAL]" if self.is_evaluation else "[TRAIN]"
                print(f"{tag} [ACTION] t={self._t + 1} decoded={decoded} | rew=...")
            except Exception as e:
                tag = "[EVAL]" if self.is_evaluation else "[TRAIN]"
                print(f"{tag} [ACTION] t={self._t + 1} action_type={type(action)} | rew=... (decode error: {e})")

        active_agents: List[str] = list(self._last_observations.keys())
        actions: Dict[str, Any] = {}

        for ag in active_agents:
            if ag == self.current_controlled:
                nested_obs = self._last_observations[ag]
                decoded = self._decode_action(action, agent_id=ag)
                decoded = self._apply_action_mask(decoded, nested_obs, agent_id=ag)
                actions[ag] = decoded
            else:
                policy = self.other_policies.get(ag)
                if policy is not None:
                    # policy must return an env-valid dict action; we still pass it
                    # through _apply_action_mask so that top-k ablation can be
                    # applied consistently (mask repair remains a no-op for
                    # already valid actions).
                    nested_obs = self._last_observations[ag]
                    try:
                        a = policy(nested_obs)
                    except Exception as e:
                        logger.error(f"Policy for agent {ag} failed: {e}")
                        a = do_nothing_policy(nested_obs["observation"], nested_obs["action_mask"])
                    
                    if isinstance(a, dict):
                        a = self._apply_action_mask(a, nested_obs, agent_id=ag)
                    actions[ag] = a
                else:
                    # Sample from this wrapper's action_space.
                    raw = self.action_space.sample()
                    decoded = self._decode_action(raw, agent_id=ag)
                    nested_obs = self._last_observations.get(ag, {})
                    decoded = self._apply_action_mask(decoded, nested_obs, agent_id=ag)
                    actions[ag] = decoded

        # print(f"Wrapper.step(t={self._t}) with {len(actions)} actions")
        
        # Suppress "No more agents to activate!" print from the environment
        # as requested by the user.
        with open(os.devnull, 'w') as fnull:
            old_stdout = sys.stdout
            sys.stdout = fnull
            try:
                observations, rewards, terminations, truncations, infos = self.env.step(actions)
            finally:
                sys.stdout = old_stdout

        self._last_observations = observations

        # Falls der kontrollierte Agent nicht mehr in den Beobachtungen auftaucht
        # (z.B. durch interne Env-Logik entfernt wurde), behandeln wir dies wie
        # eine Termination des Agents und beenden die Episode für RLlib.
        if self.current_controlled not in observations:
            reward = float(rewards.get(self.current_controlled, 0.0))
            terminated = bool(terminations.get(self.current_controlled, True))
            truncated = bool(truncations.get(self.current_controlled, False))
            info = infos.get(self.current_controlled, {})
            # Inject paper stats even when agent was eliminated
            try:
                agent_idx = self.env.agent_to_id.get(self.current_controlled)
                if agent_idx is not None:
                    completed = int(self.env.agent_completed_projects[agent_idx])
                    accepted_pids = self.env.agent_successful_projects[agent_idx]
                    accepted = len(accepted_pids)
                    info["papers_completed"] = completed
                    info["papers_accepted"] = accepted
                    info["papers_rejected"] = completed - accepted
            except Exception:
                pass
            # Inject episode-level env_metrics for horizon/truncation analysis
            try:
                if hasattr(self.env, "get_episode_metrics"):
                    info["env_metrics"] = self.env.get_episode_metrics()
            except Exception:
                pass
            logger.info(
                "Episode END for %s (t=%s) -> terminated=%s truncated=%s reward=%s (agent missing)",
                self.current_controlled,
                self._t + 1,
                terminated,
                truncated,
                reward,
            )
            # Printet auch in der Konsole, damitman es sofort sieht, ohne ins Log zu schauen.
            # print(
            #     f"[WRAPPER] Episode END for {self.current_controlled} (t={self._t + 1}) -> "
            #     f"terminated={terminated} truncated={truncated} reward={reward} (agent missing)"
            # )
            self._t = 0
            # Beobachtung für RLlib ist in diesem Fall egal; wir geben die letzte zurück
            # und signalisieren Done.
            last_nested = next(iter(observations.values())) if observations else {}
            last_obs_vec = self._flatten_to_vector(last_nested) if isinstance(last_nested, dict) else np.zeros_like(
                self.observation_space.low,
                dtype=np.float32,
            )
            last_obs_vec = self._ensure_obs_vector_ok(last_obs_vec, where="step-missing-agent")
            return last_obs_vec, reward, True, truncated, info

        nested = observations[self.current_controlled]
        obs_vec = self._flatten_to_vector(nested)
        reward = float(rewards.get(self.current_controlled, 0.0))
        terminated = bool(terminations.get(self.current_controlled, False))
        truncated = bool(truncations.get(self.current_controlled, False))
        info = infos.get(self.current_controlled, {})
        # paper_stats & debug_effort are already in info (set by env.step())

        # Inject paper stats & effort diagnostics
        info["effort_total_count"] = self._effort_total_count
        info["effort_invalid_count"] = self._effort_invalid_count
        info["effort_valid_count"] = self._effort_valid_count

        # Detailed effort analysis for prestige and deadlines
        try:
            agent_idx = self.env.agent_to_id.get(self.current_controlled)
            if agent_idx is not None:
                active_p_ids = self.env._get_active_projects(agent_idx)
                if active_p_ids:
                    prestigies = [self.env.projects[pid].prestige for pid in active_p_ids]
                    deadlines = [self.env.projects[pid].get_time_remaining(self.env.timestep) for pid in active_p_ids]
                    
                    max_prestige = max(prestigies)
                    min_deadline = min(deadlines)
                    
                    effort_idx = int(actions[self.current_controlled].get("put_effort", 0))
                    # Slot 0 is 'no effort', slots 1..N correspond to active projects
                    # In PeerGroupEnvironment, the order of active projects matches the observation slots.
                    # RLLibSingleAgentWrapper assumes this mapping.
                    
                    chosen_prestige = 0.0
                    chosen_deadline = 0
                    
                    # Correctly map effort_idx to the specific project slot in the environment
                    if 1 <= effort_idx <= self.env.max_projects_per_agent:
                        # self.env.agent_active_projects contains the project IDs (or None) at each slot
                        # effort_idx 1 corresponds to slot 0
                        p_id = self.env.agent_active_projects[agent_idx][effort_idx - 1]
                        if p_id is not None:
                            project = self.env.projects[p_id]
                            chosen_prestige = project.prestige
                            chosen_deadline = project.get_time_remaining(self.env.timestep)
                    
                    info["effort_analysis"] = {
                        "chosen_prestige": chosen_prestige,
                        "max_prestige": max_prestige,
                        "chose_max_prestige": 1 if (chosen_prestige >= max_prestige > 0) else 0,
                        "chosen_remaining_time": chosen_deadline,
                        "min_remaining_time": min_deadline,
                        "chose_most_urgent": 1 if (chosen_deadline <= min_deadline and effort_idx > 0 and chosen_deadline > 0) else 0,
                        "effort_idx": effort_idx,
                    }
        except Exception as e:
            logger.debug(f"Effort analysis failed: {e}")

        # Force horizon if requested
        self._t += 1
        
        # Optional logging of the result of the action
        if self.info_action and (self._t % self.info_interval == 0):
            tag = "[EVAL]" if self.is_evaluation else "[TRAIN]"
            print(f"{tag} [RESULT] t={self._t} rew={reward}")

        # Inject the actual action taken by the controlled agent for debugging/callbacks
        info["last_action"] = action
        self.last_action = action

        if self._force_horizon is not None and self._t >= int(self._force_horizon):
            truncated = True

        obs_vec = self._ensure_obs_vector_ok(obs_vec, where="step")

        # ------------------------------------------------------------------
        # Inject paper acceptance stats for the controlled agent into info
        # ------------------------------------------------------------------
        try:
            agent_idx = self.env.agent_to_id.get(self.current_controlled)
            if agent_idx is not None:
                completed = int(self.env.agent_completed_projects[agent_idx])
                accepted_pids = self.env.agent_successful_projects[agent_idx]
                accepted = len(accepted_pids)
                rejected = completed - accepted
                info["papers_completed"] = completed
                info["papers_accepted"] = accepted
                info["papers_rejected"] = rejected
        except Exception:
            pass  # never crash training for stats
        if self.strict_space_check and self._env_obs_space is not None:
            obs_part = nested.get("observation", {})
            if not self._env_obs_space.contains(obs_part):
                logger.warning("Env observation NOT contained in env.observation_space(%s) at step()", self._ref_agent)

        if terminated or truncated:
            logger.info(
                "Episode END for %s (t=%s) -> terminated=%s truncated=%s reward=%s",
                self.current_controlled,
                self._t,
                terminated,
                truncated,
                reward,
            )
            # print(f"[WRAPPER] Episode END for {self.current_controlled} (t={self._t}) -> terminated={terminated} truncated={truncated} reward={reward}")

            # Inject episode-level env_metrics for horizon/truncation analysis
            try:
                if hasattr(self.env, "get_episode_metrics"):
                    info["env_metrics"] = self.env.get_episode_metrics()
            except Exception:
                pass  # never crash training for stats

            self._t = 0

        return obs_vec, reward, terminated, truncated, info

    def render(self, mode="human"):
        return getattr(self.env, "render", lambda *a, **k: None)(mode)

    def close(self):
        # Optional debug info for top-k ablation
        if self.topk_collab is not None and self._topk_calls > 0:
            prune_rate = self._topk_pruned / float(self._topk_calls)
            avg_selected = self._topk_selected_count_sum / float(self._topk_calls)
            logger.info(
                "Top-k collab stats: k=%s, calls=%d, pruned=%d (rate=%.3f), avg_selected=%.2f",
                self.topk_collab,
                self._topk_calls,
                self._topk_pruned,
                prune_rate,
                avg_selected,
            )
        return getattr(self.env, "close", lambda *a, **k: None)()

    # -----------------------------
    # Helpers
    # -----------------------------

    def _ensure_obs_vector_ok(self, obs_vec: Any, *, where: str) -> np.ndarray:
        """Ensure observation is a 1D float32 vector matching observation_space length."""
        if not isinstance(obs_vec, np.ndarray):
            logger.warning("%s returned non-ndarray obs: %s", where, type(obs_vec))
            print(f"[WRAPPER] WARNING: {where} returned non-ndarray obs: {type(obs_vec)}")
            obs_vec = np.asarray(obs_vec, dtype=np.float32)

        if obs_vec.ndim != 1:
            logger.warning("%s flattened obs not 1D, reshaping: shape=%s", where, getattr(obs_vec, "shape", None))
            print(f"[WRAPPER] WARNING: {where} flattened obs not 1D, reshaping: shape={getattr(obs_vec,'shape',None)}")
            obs_vec = obs_vec.ravel()

        obs_vec = obs_vec.astype(np.float32, copy=False)

        expected_len = int(self.observation_space.shape[0])
        if obs_vec.size != expected_len:
            # With template-driven flattening, this should not happen. Keep as a last-resort safeguard.
            logger.warning("%s obs length mismatch: got %s expected %s; padding/truncating", where, obs_vec.size, expected_len)
            print(f"[WRAPPER] WARNING: {where} obs length mismatch: got {obs_vec.size} expected {expected_len}; padding/truncating")
            if obs_vec.size < expected_len:
                pad = np.zeros((expected_len - obs_vec.size,), dtype=np.float32)
                obs_vec = np.concatenate([obs_vec, pad])
            else:
                obs_vec = obs_vec[:expected_len]

        return obs_vec

    def _zeros_from_space(self, space):
        """Create a nested zero-valued observation matching a Gym space (Dict/Box/MultiBinary/Discrete)."""
        from gymnasium.spaces import Box as GymBox, Dict as GymDict, MultiBinary, Discrete

        if isinstance(space, GymDict):
            return {k: self._zeros_from_space(s) for k, s in space.spaces.items()}

        if isinstance(space, GymBox):
            return np.zeros(space.shape, dtype=space.dtype)

        if isinstance(space, MultiBinary):
            n = space.n if isinstance(space.n, tuple) else (space.n,)
            return np.zeros(n, dtype=np.int8)

        if isinstance(space, Discrete):
            return np.array(0, dtype=np.int64)

        # Fallback for unknown spaces
        try:
            sample = space.sample()
            arr = np.asarray(sample)
            return np.zeros_like(arr)
        except Exception:
            return np.zeros((0,), dtype=np.float32)

    def _deep_copy_numeric(self, x: Any) -> Any:
        """Deep-copy dict/arrays/scalars into numpy containers (best effort)."""
        if isinstance(x, dict):
            return {k: self._deep_copy_numeric(v) for k, v in x.items()}
        arr = np.asarray(x)
        if arr.dtype == object:
            raise TypeError(f"Non-numeric object encountered while copying: {type(x)}")
        return arr.copy()

