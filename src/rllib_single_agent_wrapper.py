"""
rllib_single_agent_wrapper.py

Single-agent Gymnasium wrapper around a PettingZoo ParallelEnv that:

1) Exposes one controlled agent through a standard Gymnasium API.
2) Uses a multi-head Dict action space:
   {"choose_project": Discrete, "put_effort": Discrete, "collaborate_with": Box}
3) Decodes and repairs actions using the env-provided action masks.
4) Supports fixed policies for all non-controlled agents.
5) Optionally applies a top-k collaboration ablation.
6) Flattens nested observations into a stable 1D float32 vector for RLlib.
7) Re-encodes running project slots into a compact, slot-stable representation.

Observation design for running project slots:
- explicit activity flag (`is_active`)
- progress features (`current_effort`, `remaining_effort`, `progress_ratio`)
- urgency features (`time_left`, `urgency`)
- project quality features (`prestige`, `novelty`, optional `societal_value`)
- summarized collaboration features instead of raw peer vectors
- explicit slot identity features (`slot_index`, normalized index, one-hot)

This keeps the flattened observation size stable across steps and makes empty vs.
active project slots semantically distinguishable.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

import logging

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

        # Simple debug counters (only used if top-k is enabled)
        self._topk_calls = 0
        self._topk_pruned = 0
        self._topk_selected_count_sum = 0

        self.current_controlled: Optional[str] = None
        self._last_observations: Dict[str, Any] = {}

        self.expected_obs_size = 0

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
        self._CP = int(self.env.n_projects_per_step + 1)  # choose_project
        self._PE = int(self.env.max_projects_per_agent + 1)  # put_effort
        self._CB = int(self.env.max_peer_group_size)  # collaborate bits (max_peer_slots)

        # Define discrete action spaces that match the environment
        # RLlib will automatically create separate heads for each discrete component
        # choose_project: Discrete(0, 1, 2, ...)
        # put_effort: Discrete(0, 1, 2, ..., 8)
        # collaborate_with: Box([0, 1]^CB) for continuous collaboration intent
        self.action_space = gym.spaces.Dict({
            "choose_project": gym.spaces.Discrete(self._CP),
            "put_effort": gym.spaces.Discrete(self._PE),
            "collaborate_with": gym.spaces.Box(0, 1, shape=(self._CB,), dtype=np.float32)
        })


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

        # Build deterministic templates for stable flattening
        # The template is derived from observation_space and defines the full feature structure
        # with all 8 project slots. At runtime, observations may have fewer projects (0-8),
        # and RLlib automatically handles padding/truncating the observation vector.

        if obs_space is not None:
            raw_obs_template = self._zeros_from_space(obs_space)
            self._mask_template = {
                "choose_project": np.zeros(self.env.n_projects_per_step + 1, dtype=np.int8),
                "collaborate_with": np.zeros(self.env.max_peer_group_size, dtype=np.int8),
                "put_effort": np.zeros(self.env.max_projects_per_agent + 1, dtype=np.int8),
            }
            # Normalize template to get correct structure with ALL slots
            self._obs_template = self._create_normalized_obs_template(raw_obs_template)

            # Create sample vector - pass a flag to _flatten_to_vector to indicate this is a template
            nested_template = {"observation": self._obs_template, "action_mask": self._mask_template, "_is_template": True}
            sample_vec = self._flatten_to_vector(nested_template)
        else:
            raise RuntimeError("Cannot create template: observation_space is None")
            nested_template = {"observation": self._obs_template, "action_mask": self._mask_template, "_is_template": True}
            sample_vec = self._flatten_to_vector(nested_template)
            self._obs_template = self._deep_copy_numeric(nested.get("observation", {}))
            self._mask_template = self._deep_copy_numeric(nested.get("action_mask", {}))
            # Normalize template BEFORE flattening
            normalized_obs_template = self._create_normalized_obs_template(self._obs_template)
            # Use the normalized template for all future flattening operations
            self._obs_template = normalized_obs_template
            sample_vec = self._flatten_to_vector(
                {"observation": normalized_obs_template, "action_mask": self._mask_template})

        # Set observation_space
        # IMPORTANT: Use the ACTUAL flattened size from sample_vec
        # This is the real size that _flatten_to_vector() produces at runtime
        # NOT the template-based size which can differ when running_projects normalization changes sizes
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(sample_vec.size),),
            dtype=np.float32,
        )

        # Set expected_obs_size ONCE during initialization - NEVER change it later!
        self.expected_obs_size = int(sample_vec.size)

        # Don't overwrite last observations here; caller will call reset() before stepping.
        self._last_observations = {}
        self.current_controlled = self._ref_agent

    # -----------------------------
    # Action encoding/decoding
    # -----------------------------

    def _decode_action(self, a: Any, agent_id: Optional[str] = None) -> ActionDict:
        """Pass-through or convert action into env dict format.

        Now expects Dict actions with discrete choose_project and put_effort:
        {'choose_project': int, 'put_effort': int, 'collaborate_with': ndarray or float}
        """

        # 1. Dictionary format (PRIMARY - from RLlib policy)
        if isinstance(a, dict):
            choose_project = int(a.get("choose_project", 0))
            put_effort = int(a.get("put_effort", 0))
            collab = a.get("collaborate_with", np.zeros(self._CB, dtype=np.float32))

            # Threshold collaboration intent to binary bits
            collab_bits = np.asarray(collab, dtype=np.float32)
            collab_bits = (collab_bits > 0.5).astype(np.int8)

            return {
                "choose_project": choose_project,
                "put_effort": put_effort,
                "collaborate_with": collab_bits,
            }

        # 2. Fallback for legacy tuple/list actions
        if isinstance(a, (tuple, list)) and len(a) >= 3:
            return {
                "choose_project": int(a[0]),
                "put_effort": int(a[1]),
                "collaborate_with": np.asarray(a[2], dtype=np.int8),
            }

        # 3. Fallback for legacy flat Box actions (should not happen with new action_space)
        if isinstance(a, np.ndarray) and a.ndim == 1 and len(a) == (self._CB + 2):
            choose_project = int(np.round(a[0]))
            put_effort = int(np.round(a[1]))
            collab_bits = np.round(a[2:]).astype(np.int8)
            return {
                "choose_project": choose_project,
                "put_effort": put_effort,
                "collaborate_with": collab_bits,
            }

        # 4. Fallback for old integer actions (legacy support)
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

    def decode_action_id(self, action_id: Any) -> Dict[str, Any]:
        """Public API: decode an action into a human-readable dict.

        Now supports both dict actions and legacy integer IDs.
        """
        decoded = self._decode_action(action_id)
        collab = decoded["collaborate_with"]
        return {
            "choose_project": decoded["choose_project"],
            "put_effort": decoded["put_effort"],
            "collaborate_with": collab.tolist() if hasattr(collab, "tolist") else list(collab),
            "n_collaborators": int(np.sum(collab)),
        }

    def _apply_action_mask(self, decoded: ActionDict, nested_obs: NestedObs,
                           agent_id: Optional[str] = None) -> ActionDict:
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
            if cp < 0 or cp >= cp_mask.size or cp_mask[cp] <= 0:
                decoded["choose_project"] = 0

        # put_effort
        pe_mask = np.asarray(mask.get("put_effort", []))
        if pe_mask.size:
            pe = int(decoded.get("put_effort", 0))
            if pe < 0 or pe >= pe_mask.size or pe_mask[pe] <= 0:
                decoded["put_effort"] = 0

        # collaborate_with
        c_mask = np.asarray(mask.get("collaborate_with", []))
        if c_mask.size:
            c = np.asarray(decoded.get("collaborate_with", np.zeros(self._CB, dtype=np.int8)), dtype=np.int8).copy()
            allowed = (c_mask > 0)

            L = min(len(c), len(allowed))
            c_slice = c[:L]
            c_slice[~allowed[:L]] = 0
            c[:L] = c_slice

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

    # -----------------------------
    # Top-k collaboration helpers
    # -----------------------------

    def _extract_peer_feature_array(self, obs: Any, candidate_keys: Iterable[str], length: int,
                                    default_value: Any = 0.0) -> Optional[np.ndarray]:
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
            s_centroid = self._extract_peer_feature_array(obs, ["self_centroid", "self_centroids"], 2,
                                                          default_value=0.0)

            p_centroids = p_centroids.reshape(self._CB, 2)
            s_centroid = s_centroid.reshape(1, 2)

            # Distance: sqrt(sum((p - s)^2))
            dist = np.sqrt(np.sum((p_centroids - s_centroid) ** 2, axis=1)).astype(np.float32)

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


    def _safe_scalar(self, value: Any, default: float = 0.0, dtype=float):
        """Best-effort extraction of a scalar from numpy/scalar/list-like input."""
        try:
            arr = np.asarray(value)
            if arr.size == 0:
                return dtype(default)
            return dtype(arr.reshape(-1)[0])
        except Exception:
            return dtype(default)

    def _make_slot_identity_features(self, slot_idx: int) -> Dict[str, Any]:
        """Explicit slot identification features for flat MLP policies."""
        one_hot = np.zeros(self.env.max_projects_per_agent, dtype=np.float32)
        if 0 <= slot_idx < self.env.max_projects_per_agent:
            one_hot[slot_idx] = 1.0

        denom = max(self.env.max_projects_per_agent - 1, 1)
        return {
            "slot_index": np.array([slot_idx], dtype=np.int32),
            "slot_index_normalized": np.array([slot_idx / denom], dtype=np.float32),
            "slot_one_hot": one_hot,
        }

    def _encode_project_slot(self, project_obs: Optional[Dict[str, Any]], slot_idx: int, *, is_active: bool) -> Dict[str, Any]:
        """
        Convert a raw running-project observation into a compact, slot-stable encoding.

        Design goals:
        - Explicitly distinguish active vs. dummy slots via `is_active`
        - Give the policy semantically useful project progress features
        - Provide explicit slot identity (scalar + one-hot) for flat MLPs
        - Avoid high-dimensional, noisy peer vectors by summarizing them
        """
        if not is_active or not isinstance(project_obs, dict):
            project_obs = {}

        required_effort = self._safe_scalar(project_obs.get("required_effort", 0), default=0.0, dtype=float)
        current_effort = self._safe_scalar(project_obs.get("current_effort", 0.0), default=0.0, dtype=float)
        time_left = self._safe_scalar(project_obs.get("time_left", 0), default=0.0, dtype=float)
        prestige = self._safe_scalar(project_obs.get("prestige", 0.0), default=0.0, dtype=float)
        novelty = self._safe_scalar(project_obs.get("novelty", 0.0), default=0.0, dtype=float)

        societal_value = self._safe_scalar(
            project_obs.get("societal_value", project_obs.get("societal_value_score", 0.0)),
            default=0.0,
            dtype=float,
        )

        contributors = np.asarray(project_obs.get("contributors", np.zeros(self._CB, dtype=np.int8)), dtype=np.float32).ravel()
        contributor_effort = np.asarray(
            project_obs.get("contributor_effort", np.zeros(self._CB, dtype=np.float32)),
            dtype=np.float32,
        ).ravel()
        peer_fit = np.asarray(project_obs.get("peer_fit", np.zeros(self._CB, dtype=np.float32)), dtype=np.float32).ravel()

        if contributors.size < self._CB:
            contributors = np.pad(contributors, (0, self._CB - contributors.size))
        else:
            contributors = contributors[:self._CB]

        if contributor_effort.size < self._CB:
            contributor_effort = np.pad(contributor_effort, (0, self._CB - contributor_effort.size))
        else:
            contributor_effort = contributor_effort[:self._CB]

        if peer_fit.size < self._CB:
            peer_fit = np.pad(peer_fit, (0, self._CB - peer_fit.size))
        else:
            peer_fit = peer_fit[:self._CB]

        if is_active:
            remaining_effort = max(required_effort - current_effort, 0.0)
            progress_ratio = current_effort / max(required_effort, 1.0)
            urgency = 1.0 / max(time_left, 1.0)
            num_contributors = float(np.sum(contributors > 0.0))

            active_peer_fit = peer_fit[contributors > 0.0]
            if active_peer_fit.size == 0:
                active_peer_fit = peer_fit[peer_fit != 0.0]

            peer_fit_mean = float(np.mean(active_peer_fit)) if active_peer_fit.size else 0.0
            peer_fit_max = float(np.max(active_peer_fit)) if active_peer_fit.size else 0.0
            total_contributor_effort = float(np.sum(contributor_effort))
        else:
            required_effort = 0.0
            current_effort = 0.0
            remaining_effort = 0.0
            progress_ratio = 0.0
            time_left = 0.0
            urgency = 0.0
            prestige = 0.0
            novelty = 0.0
            societal_value = 0.0
            num_contributors = 0.0
            peer_fit_mean = 0.0
            peer_fit_max = 0.0
            total_contributor_effort = 0.0

        encoded = {
            "is_active": np.array([1 if is_active else 0], dtype=np.int8),
            "required_effort": np.array([required_effort], dtype=np.float32),
            "current_effort": np.array([current_effort], dtype=np.float32),
            "remaining_effort": np.array([remaining_effort], dtype=np.float32),
            "progress_ratio": np.array([progress_ratio], dtype=np.float32),
            "time_left": np.array([time_left], dtype=np.float32),
            "urgency": np.array([urgency], dtype=np.float32),
            "prestige": np.array([prestige], dtype=np.float32),
            "novelty": np.array([novelty], dtype=np.float32),
            "societal_value": np.array([societal_value], dtype=np.float32),
            "num_contributors": np.array([num_contributors], dtype=np.float32),
            "peer_fit_mean": np.array([peer_fit_mean], dtype=np.float32),
            "peer_fit_max": np.array([peer_fit_max], dtype=np.float32),
            "total_contributor_effort": np.array([total_contributor_effort], dtype=np.float32),
        }
        encoded.update(self._make_slot_identity_features(slot_idx))

        return encoded

    def _create_normalized_obs_template(self, obs_template: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a normalized observation template that ensures running_projects
        always has ALL max_projects_per_agent slots in the new compact encoding.
        """
        obs_copy = self._deep_copy_numeric(obs_template) if obs_template else {}

        running_projects = obs_copy.get("running_projects", {})
        if not isinstance(running_projects, dict):
            running_projects = {}

        normalized_running = {}
        for i in range(self.env.max_projects_per_agent):
            project_key = f"project_{i}"
            raw_project = running_projects.get(project_key)
            normalized_running[project_key] = self._encode_project_slot(
                raw_project,
                i,
                is_active=bool(isinstance(raw_project, dict) and len(raw_project) > 0),
            )

        obs_copy["running_projects"] = normalized_running
        return obs_copy

    def _flatten_to_vector(self, nested_obs: Any) -> np.ndarray:
        """
        Flatten nested per-agent observations in a template-driven, size-stable way.

        Before flattening, raw `running_projects` observations are normalized into the
        compact per-slot encoding so that all project slots share the same structure.

        NOTE: If _is_template flag is present and True, don't normalize (template is already normalized).
        """
        if not (isinstance(nested_obs, dict) and "observation" in nested_obs):
            raise TypeError("Expected nested obs: {'observation': ..., 'action_mask': ...}")

        # Check if this is a template (already normalized during init)
        is_template = nested_obs.pop("_is_template", False)

        obs_part = nested_obs.get("observation", {})
        mask_part = nested_obs.get("action_mask", {})

        # Only normalize if NOT a template
        if not is_template:
            obs_part = self._normalize_observation(obs_part)

        obs_vec = self._flatten_any_like_template(obs_part, self._obs_template)
        mask_vec = self._flatten_mask_like_template(mask_part, self._mask_template)

        out = np.concatenate([obs_vec, mask_vec]).astype(np.float32, copy=False)

        # Log size mismatch only if it's substantial (suppress for now since env naturally has variable sizes)
        if len(out) != self.expected_obs_size and self.expected_obs_size > 0:
            # Only log if mismatch is more than 5% or if this is initialization phase
            pct_diff = abs(len(out) - self.expected_obs_size) / float(self.expected_obs_size) * 100
            if pct_diff > 5:
                logger.debug(f"Observation size mismatch: got {len(out)} expected {self.expected_obs_size} ({pct_diff:.1f}% diff)")

        return out

    def _normalize_observation(self, obs_part: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize observation to ensure consistent size across episodes.

        Running-project slots are converted from raw env observations into a compact
        per-slot encoding with:
        - explicit activity flag
        - progress / remaining effort / urgency
        - summarized collaborator information
        - explicit slot identity features
        """
        obs_copy = obs_part.copy() if isinstance(obs_part, dict) else {}

        raw_running = obs_copy.get("running_projects", {})
        if not isinstance(raw_running, dict):
            raw_running = {}

        normalized_running = {}
        for i in range(self.env.max_projects_per_agent):
            project_key = f"project_{i}"
            raw_project = raw_running.get(project_key)
            is_active = bool(isinstance(raw_project, dict) and len(raw_project) > 0)
            normalized_running[project_key] = self._encode_project_slot(raw_project, i, is_active=is_active)

        obs_copy["running_projects"] = normalized_running
        return obs_copy



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

        logger.info("Episode START for %s (obs_len=%d, force_horizon=%s)", self.current_controlled, obs_vec.size,
                    self._force_horizon)
        # print(f"[WRAPPER] Episode START for {self.current_controlled} (obs_len={obs_vec.size}, force_horizon={self._force_horizon})")
        return obs_vec, info

    def step(self, action):
        if self.current_controlled is None:
            raise RuntimeError("Wrapper not reset() before step().")

        # IMPORTANT: use keys from last observations (guaranteed the set of agents we can act for)
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
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
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

        # Force horizon if requested
        self._t += 1
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
            obs_vec = np.asarray(obs_vec, dtype=np.float32)

        if obs_vec.ndim != 1:
            logger.warning("%s flattened obs not 1D, reshaping: shape=%s", where, getattr(obs_vec, "shape", None))
            obs_vec = obs_vec.ravel()

        obs_vec = obs_vec.astype(np.float32, copy=False)

        expected_len = int(self.observation_space.shape[0])

        if obs_vec.size != expected_len:
            # Variable observation size is expected behavior (0-8 projects at any given time)
            # RLlib handles padding/truncating automatically
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

