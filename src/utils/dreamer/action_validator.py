"""
DreamerActionValidator: Validierung, Repair und Top-K Collaboration für Aktionen.

Diese Klasse kümmert sich um:
- Action Mask Validierung (sind Aktionen gültig?)
- Action Repair (ungültige Aktionen korrigieren)
- Top-K Collaboration Pruning (begrenze Collaboration auf K beste)
- Peer Scoring (Reputation, Distance, Same-Group)
"""

import numpy as np
from typing import Any, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

NestedObs = Dict[str, Any]
ActionDict = Dict[str, Any]


class DreamerActionValidator:
    """
    Validiert und repariert Aktionen gegen Action Masks.

    Implementiert:
    - Mask-basierte Validierung für jeden Action Head
    - Automatische Repair von ungültigen Aktionen (same logic as PPO)
    - Top-K Collaboration Pruning mit Peer Scoring
    """

    def __init__(
        self,
        max_peer_group_size: int,
        n_projects_per_step: int,
        max_projects_per_agent: int,
        topk_collab: Optional[int] = None,
        topk_mode: str = "score",
        topk_seed: int = 0,
        w_rep: float = 1.0,
        w_dist: float = 1.0,
        w_same: float = 0.5,
        debug_steps: int = 50,
    ):
        """
        Args:
            max_peer_group_size: CB - Größe der Peer Group
            n_projects_per_step: CP - Anzahl Projekte zum Starten
            max_projects_per_agent: PE - Max aktive Projekte
            topk_collab: Top-K Collaboration Limit (None = keine Limitierung)
            topk_mode: "score" oder andere Scoring-Modi
            topk_seed: Seed für Tie-Breaking
            w_rep / w_dist / w_same: Gewichte für Scoring
            debug_steps: Erste N Steps für Debug-Logging
        """
        self._CB = int(max_peer_group_size)
        self._CP = int(n_projects_per_step + 1)
        self._PE = int(max_projects_per_agent + 1)
        self.topk_collab = topk_collab
        self.topk_mode = topk_mode
        self.topk_seed = int(topk_seed)
        self.w_rep = float(w_rep)
        self.w_dist = float(w_dist)
        self.w_same = float(w_same)
        self.debug_steps = int(debug_steps)

        # Counters
        self._topk_calls = 0
        self._topk_pruned = 0
        self._topk_selected_count_sum = 0

    # =========================================================================
    # Action Mask Validation
    # =========================================================================

    def validate_action_against_mask(
        self,
        action: ActionDict,
        mask: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        Validate all action heads against mask.

        Returns dict with per-head validation results:
        {
            "choose_project_valid": bool,
            "put_effort_valid": bool,
            "collaborate_with_valid": bool,
            "all_valid": bool
        }
        """
        cp_valid = self._is_choose_project_valid(action, mask)
        pe_valid = self._is_put_effort_valid(action, mask)
        c_valid = self._is_collaborate_with_valid(action, mask)

        return {
            "choose_project_valid": cp_valid,
            "put_effort_valid": pe_valid,
            "collaborate_with_valid": c_valid,
            "all_valid": cp_valid and pe_valid and c_valid
        }

    def _is_choose_project_valid(self, action: ActionDict, mask: Dict[str, Any]) -> bool:
        """Check if choose_project action is valid against mask."""
        cp_mask = np.asarray(mask.get("choose_project", []))
        if cp_mask.size == 0:
            return True  # No mask available, assume valid

        cp = int(action.get("choose_project", 0))
        if cp < 0 or cp >= cp_mask.size:
            return False

        # Interpret mask values >0 as valid (handles both 0/1 and 0/2 masks)
        return cp_mask[cp] > 0

    def _is_put_effort_valid(self, action: ActionDict, mask: Dict[str, Any]) -> bool:
        """Check if put_effort action is valid against mask."""
        pe_mask = np.asarray(mask.get("put_effort", []))
        if pe_mask.size == 0:
            return True  # No mask available, assume valid

        pe = int(action.get("put_effort", 0))
        if pe < 0 or pe >= pe_mask.size:
            return False

        # Interpret mask values >0 as valid (handles both 0/1 and 0/2 masks)
        return pe_mask[pe] > 0

    def _is_collaborate_with_valid(self, action: ActionDict, mask: Dict[str, Any]) -> bool:
        """Check if collaborate_with action is valid against mask."""
        c_mask = np.asarray(mask.get("collaborate_with", []))
        if c_mask.size == 0:
            return True  # No mask available, assume valid

        c = np.asarray(action.get("collaborate_with", np.zeros(self._CB, dtype=np.int8)), dtype=np.int8)
        if c.size != self._CB:
            return False

        # Check if any collaboration bit is set for disallowed peers
        allowed = (c_mask > 0)  # Interpret mask values >0 as valid
        L = min(len(c), len(allowed))

        # Check if any disallowed bit is set
        if np.any(c[:L] & ~allowed[:L]):
            return False

        # Check if any bit beyond mask length is set
        if len(c) > L and np.any(c[L:]):
            return False

        return True

    # =========================================================================
    # Action Repair
    # =========================================================================

    def repair_action(
        self,
        decoded: ActionDict,
        nested_obs: NestedObs,
        agent_id: Optional[str] = None,
        debug_step: int = 0,
    ) -> tuple[ActionDict, bool, List[str]]:
        """
        Repair invalid actions using environment-provided action masks.

        Same logic as PPO wrapper for fair comparison.

        Args:
            decoded: Action dict to repair
            nested_obs: Nested observation (for mask access)
            agent_id: Agent ID (for logging)
            debug_step: Current step (for debug logging)

        Returns:
            (repaired_action, repair_applied, heads_repaired)
        """
        mask = nested_obs.get("action_mask", {})
        if not isinstance(mask, dict):
            # Ensure collaborate_with exists and is correctly sized
            if "collaborate_with" not in decoded or not hasattr(decoded["collaborate_with"], "size"):
                decoded["collaborate_with"] = np.zeros(self._CB, dtype=np.int8)
            return decoded, False, []

        repaired = False
        heads_repaired = []

        # Repair choose_project
        cp_mask = np.asarray(mask.get("choose_project", []))
        if cp_mask.size:
            cp = int(decoded.get("choose_project", 0))
            if cp < 0 or cp >= cp_mask.size or cp_mask[cp] <= 0:
                if debug_step < self.debug_steps:
                    valid_indices = [i for i, v in enumerate(cp_mask) if v > 0]
                    logger.debug(
                        f"[Mask] choose_project: {cp} invalid, mask={cp_mask.tolist()}, "
                        f"valid={valid_indices}, repairing→0"
                    )
                decoded["choose_project"] = 0
                heads_repaired.append("choose_project")
                repaired = True

        # Repair put_effort
        pe_mask = np.asarray(mask.get("put_effort", []))
        if pe_mask.size:
            pe = int(decoded.get("put_effort", 0))
            if pe < 0 or pe >= pe_mask.size or pe_mask[pe] <= 0:
                if debug_step < self.debug_steps:
                    valid_indices = [i for i, v in enumerate(pe_mask) if v > 0]
                    logger.debug(
                        f"[Mask] put_effort: {pe} invalid, mask={pe_mask.tolist()}, "
                        f"valid={valid_indices}, repairing→0"
                    )
                decoded["put_effort"] = 0
                heads_repaired.append("put_effort")
                repaired = True

        # Repair collaborate_with
        c_mask = np.asarray(mask.get("collaborate_with", []))
        c = np.asarray(decoded.get("collaborate_with", np.zeros(self._CB, dtype=np.int8)), dtype=np.int8).copy()

        if c.size != self._CB:
            c = np.zeros(self._CB, dtype=np.int8)
            repaired = True

        if c_mask.size:
            allowed = (c_mask > 0)
            L = min(len(c), len(allowed))

            # Zero out disallowed bits
            c_slice = c[:L]
            allowed_slice = allowed[:L]
            invalid_collabs = c_slice & ~allowed_slice
            if np.any(invalid_collabs):
                if debug_step < self.debug_steps:
                    invalid_indices = np.where(invalid_collabs)[0].tolist()
                    valid_indices = np.where(allowed[:L])[0].tolist()
                    logger.debug(
                        f"[Mask] collaborate: selected={np.sum(c_slice)} "
                        f"invalid_indices={invalid_indices}, valid_options={len(valid_indices)}/{L}"
                    )
                heads_repaired.append("collaborate_with")
                repaired = True
            c_slice[~allowed_slice] = 0
            c[:L] = c_slice

            if len(c) > L:
                if np.any(c[L:]):
                    heads_repaired.append("collaborate_with")
                    repaired = True
                c[L:] = 0

            # Apply top-k collaboration ablation if enabled
            if self.topk_collab is not None and self.topk_collab >= 0:
                c = self._apply_topk_collaboration(c, c_mask, nested_obs, agent_id=agent_id, k=self.topk_collab)

        decoded["collaborate_with"] = c.astype(np.int8, copy=False)

        return decoded, repaired, heads_repaired

    # =========================================================================
    # Top-K Collaboration
    # =========================================================================

    def _apply_topk_collaboration(
        self,
        collab_bits: np.ndarray,
        collab_mask: np.ndarray,
        nested_obs: NestedObs,
        *,
        agent_id: Optional[str],
        k: int,
    ) -> np.ndarray:
        """Apply top-k pruning to collaboration bit-vector (same as PPO wrapper)."""
        self._topk_calls += 1

        c = np.asarray(collab_bits, dtype=np.int8).copy()
        mask = np.asarray(collab_mask, dtype=np.int8)

        if c.size != self._CB:
            if c.size < self._CB:
                pad = np.zeros((self._CB - c.size,), dtype=np.int8)
                c = np.concatenate([c, pad])
            else:
                c = c[:self._CB]

        if mask.size < self._CB:
            pad = np.zeros((self._CB - mask.size,), dtype=np.int8)
            mask = np.concatenate([mask, pad])
        elif mask.size > self._CB:
            mask = mask[:self._CB]

        allowed = (mask > 0)
        c[~allowed] = 0

        candidates = np.where((c == 1) & allowed)[0]
        if k is None or k < 0 or candidates.size <= k:
            self._topk_selected_count_sum += int(candidates.size)
            return c

        scores = self._compute_peer_scores(nested_obs, agent_id, k, seed=self.topk_seed)
        cand_scores = scores[candidates]
        order = np.argsort(-cand_scores, kind="mergesort")
        keep_idx = candidates[order[:k]]

        new_c = np.zeros_like(c, dtype=np.int8)
        new_c[keep_idx] = 1

        if candidates.size > k:
            self._topk_pruned += 1
        self._topk_selected_count_sum += int(keep_idx.size)

        return new_c

    def _compute_peer_scores(
        self,
        nested_obs: NestedObs,
        agent_id: Optional[str],
        k: int,
        seed: int
    ) -> np.ndarray:
        """
        Compute per-peer collaboration scores.

        Components:
        - Reputation: higher is better (normalized)
        - Distance: smaller is better (normalized, negated)
        - Same-group bonus: +w_same if peer is in same group

        Returns array of shape (CB,) with scores for each peer slot.
        """
        if k <= 0 or self._CB <= 0:
            return np.zeros((self._CB,), dtype=np.float32)

        obs = nested_obs.get("observation", {})

        # Reputation
        rep_keys = ["peer_reputation", "peer_reputations", "peers_reputation", "reputation_peers"]
        rep = self._extract_peer_feature_array(obs, rep_keys, self._CB, default_value=0.0)

        # Distance
        dist_keys = ["peer_distance", "peer_distances", "distance_to_peers", "peers_distance"]
        dist = self._extract_peer_feature_array(obs, dist_keys, self._CB, default_value=None)

        if dist is None:
            # Fallback: compute from centroids
            p_centroids = self._extract_peer_feature_array(obs, ["peer_centroids"], self._CB * 2, default_value=0.0)
            s_centroid = self._extract_peer_feature_array(obs, ["self_centroid", "self_centroids"], 2, default_value=0.0)
            p_centroids = p_centroids.reshape(self._CB, 2)
            s_centroid = s_centroid.reshape(1, 2)
            dist = np.sqrt(np.sum((p_centroids - s_centroid) ** 2, axis=1)).astype(np.float32)

        # Same-group bonus
        own_gid = self._extract_agent_group_id(obs)
        peer_gids = self._extract_peer_group_ids(obs, self._CB)
        if own_gid is not None and peer_gids is not None:
            same_group = (peer_gids == own_gid).astype(np.float32)
        else:
            same_group = np.zeros((self._CB,), dtype=np.float32)

        # Normalize
        def _min_max_normalize(x: np.ndarray) -> np.ndarray:
            x = x.astype(np.float32, copy=False)
            x_min = float(np.min(x))
            x_max = float(np.max(x))
            if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max - x_min < 1e-8:
                return np.zeros_like(x, dtype=np.float32)
            return (x - x_min) / (x_max - x_min + 1e-8)

        rep_n = _min_max_normalize(rep)
        dist_n = _min_max_normalize(dist)

        scores = (self.w_rep * rep_n - self.w_dist * dist_n + self.w_same * same_group).astype(np.float32, copy=False)

        # Deterministic tie-breaking
        rng = np.random.RandomState(seed=self.topk_seed if seed is None else seed)
        noise = rng.uniform(low=-1e-6, high=1e-6, size=scores.shape).astype(np.float32)
        scores = scores + noise

        return scores

    def _extract_peer_feature_array(
        self,
        obs: Any,
        candidate_keys: List[str],
        length: int,
        default_value: Any = 0.0
    ) -> Optional[np.ndarray]:
        """Search nested obs dict for a 1D array under candidate keys."""
        if not isinstance(obs, dict):
            if default_value is None:
                return None
            return np.full((length,), float(default_value), dtype=np.float32)

        found = None

        # Direct lookup
        for key in candidate_keys:
            if key in obs:
                found = obs[key]
                break

        # Nested lookup
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
            if default_value is None:
                return None
            return np.full((length,), float(default_value) if default_value is not None else 0.0, dtype=np.float32)

        arr = np.asarray(found, dtype=np.float32).ravel()
        if arr.size < length:
            pad = np.full((length - arr.size,), float(default_value) if default_value is not None else 0.0, dtype=np.float32)
            arr = np.concatenate([arr, pad])
        elif arr.size > length:
            arr = arr[:length]

        return arr

    def _extract_agent_group_id(self, obs: Any) -> Optional[int]:
        """Extract agent's own group ID from observation."""
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
        """Extract peer group IDs array."""
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

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_topk_stats(self) -> Dict[str, Any]:
        """Return Top-K collaboration statistics."""
        if self._topk_calls == 0:
            return {}

        prune_rate = self._topk_pruned / float(self._topk_calls)
        avg_selected = self._topk_selected_count_sum / float(self._topk_calls)

        return {
            "topk_calls": self._topk_calls,
            "topk_pruned": self._topk_pruned,
            "topk_prune_rate": prune_rate,
            "topk_avg_selected": avg_selected,
        }

