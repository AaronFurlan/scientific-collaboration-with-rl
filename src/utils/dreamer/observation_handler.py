"""
DreamerObservationHandler: Vektorisierung und Normalisierung von Beobachtungen.

Diese Klasse kümmert sich um:
- Beobachtungsvorlage-Erstellung und -verwaltung
- Normalisierung von Beobachtungen (konsistente Größe über Episoden)
- Project Slot Encoding (Normalisierung von aktiven/inaktiven Projekten)
- Flattening zu 1D-Vektoren (stabil und typsicher)
- Action Mask Flattening
"""

import numpy as np
import gymnasium as gym
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DreamerObservationHandler:
    """
    Vektorisiert und normalisiert komplexe nested Beobachtungen für DreamerV3.

    Stellt sicher, dass Beobachtungen konsistent dimensioniert sind unabhängig
    von der Anzahl aktiver Projekte.
    """

    def __init__(self, env, max_projects_per_agent: int, max_peer_group_size: int):
        """
        Args:
            env: PeerGroupEnvironment
            max_projects_per_agent: Maximale Anzahl gleichzeitiger Projekte pro Agent
            max_peer_group_size: Maximale Größe einer Peer Group (für Collaboration)
        """
        self.env = env
        self.max_projects_per_agent = max_projects_per_agent
        self.max_peer_group_size = max_peer_group_size
        self._obs_template = None
        self._mask_template = None

    def set_templates(self, obs_template: Dict[str, Any], mask_template: Dict[str, Any]):
        """Setze die Vorlage-Strukturen für konsistentes Flattening."""
        self._obs_template = obs_template
        self._mask_template = mask_template

    def flatten_to_vector(self, nested_obs: Any) -> np.ndarray:
        """
        Flatten nested per-agent observations in template-driven, size-stable way.

        Erwartet dict mit "observation" und "action_mask" Keys.
        Kann auch Templates mit "_is_template": True verarbeiten.

        Args:
            nested_obs: {'observation': {...}, 'action_mask': {...}}

        Returns:
            np.ndarray of dtype float32, flattened Beobachtung + Action Mask
        """
        if not (isinstance(nested_obs, dict) and "observation" in nested_obs):
            raise TypeError("Expected nested obs: {'observation': ..., 'action_mask': ...}")

        is_template = nested_obs.pop("_is_template", False)

        obs_part = nested_obs.get("observation", {})
        mask_part = nested_obs.get("action_mask", {})

        if not is_template:
            obs_part = self.normalize_observation(obs_part)

        obs_vec = self._flatten_any_like_template(obs_part, self._obs_template)
        mask_vec = self._flatten_mask_like_template(mask_part, self._mask_template)

        # Ensure concatenation uses float32 dtype explicitly
        out = np.concatenate([obs_vec, mask_vec], dtype=np.float32)
        return out

    def normalize_observation(self, obs_part: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize observation to ensure consistent size across episodes.

        Garantiert, dass alle max_projects_per_agent Slots vorhanden sind,
        auch wenn manche leer/inaktiv sind.
        """
        obs_copy = obs_part.copy() if isinstance(obs_part, dict) else {}

        raw_running = obs_copy.get("running_projects", {})
        if not isinstance(raw_running, dict):
            raw_running = {}

        normalized_running = {}
        for i in range(self.max_projects_per_agent):
            project_key = f"project_{i}"
            raw_project = raw_running.get(project_key)
            is_active = bool(isinstance(raw_project, dict) and len(raw_project) > 0)
            normalized_running[project_key] = self.encode_project_slot(raw_project, i, is_active=is_active)

        obs_copy["running_projects"] = normalized_running
        return obs_copy

    def encode_project_slot(self, project_obs: Optional[Dict[str, Any]], slot_idx: int, *, is_active: bool) -> Dict[str, Any]:
        """
        Standardize project observation slot encoding.

        Normalisiert ein einzelnes Projekt-Slot zu fester Struktur.
        Matches PPO wrapper's project encoding für faire Vergleiche.

        Args:
            project_obs: Rohe Projekt-Beobachtung (oder None)
            slot_idx: Index des Slots [0, max_projects_per_agent)
            is_active: Ob Projekt aktiv ist

        Returns:
            Dict mit standardisierten, konstanten Keys
        """
        if not is_active or project_obs is None:
            # Default empty project
            return {
                "is_active": np.array([0], dtype=np.int8),
                "required_effort": np.array([0.0], dtype=np.float32),
                "current_effort": np.array([0.0], dtype=np.float32),
                "remaining_effort": np.array([0.0], dtype=np.float32),
                "progress_ratio": np.array([0.0], dtype=np.float32),
                "time_left": np.array([0.0], dtype=np.float32),
                "urgency": np.array([0.0], dtype=np.float32),
                "prestige": np.array([0.0], dtype=np.float32),
                "novelty": np.array([0.0], dtype=np.float32),
                "societal_value": np.array([0.0], dtype=np.float32),
                "num_contributors": np.array([0.0], dtype=np.float32),
                "peer_fit_mean": np.array([0.0], dtype=np.float32),
                "peer_fit_max": np.array([0.0], dtype=np.float32),
                "total_contributor_effort": np.array([0.0], dtype=np.float32),
            }

        # Active project: encode fields
        required_effort = self._safe_scalar(project_obs.get("required_effort", 0), default=0.0)
        current_effort = self._safe_scalar(project_obs.get("current_effort", 0.0), default=0.0)
        time_left = self._safe_scalar(project_obs.get("time_left", 0), default=0.0)
        prestige = self._safe_scalar(project_obs.get("prestige", 0.0), default=0.0)
        novelty = self._safe_scalar(project_obs.get("novelty", 0.0), default=0.0)
        societal_value = self._safe_scalar(
            project_obs.get("societal_value", project_obs.get("societal_value_score", 0.0)),
            default=0.0,
        )

        contributors = np.asarray(project_obs.get("contributors", np.zeros(self.max_peer_group_size, dtype=np.int8)), dtype=np.float32).ravel()
        contributor_effort = np.asarray(
            project_obs.get("contributor_effort", np.zeros(self.max_peer_group_size, dtype=np.float32)),
            dtype=np.float32,
        ).ravel()
        peer_fit = np.asarray(project_obs.get("peer_fit", np.zeros(self.max_peer_group_size, dtype=np.float32)), dtype=np.float32).ravel()

        # Pad to CB size
        if contributors.size < self.max_peer_group_size:
            contributors = np.pad(contributors, (0, self.max_peer_group_size - contributors.size))
        else:
            contributors = contributors[:self.max_peer_group_size]

        if contributor_effort.size < self.max_peer_group_size:
            contributor_effort = np.pad(contributor_effort, (0, self.max_peer_group_size - contributor_effort.size))
        else:
            contributor_effort = contributor_effort[:self.max_peer_group_size]

        if peer_fit.size < self.max_peer_group_size:
            peer_fit = np.pad(peer_fit, (0, self.max_peer_group_size - peer_fit.size))
        else:
            peer_fit = peer_fit[:self.max_peer_group_size]

        # Calculate derived fields
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

        return {
            "is_active": np.array([1], dtype=np.int8),
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

    def create_normalized_obs_template(self, obs_template: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create normalized observation template ensuring all project slots are present.

        Verwendet encode_project_slot(), um alle Slots zu standardisieren.
        """
        obs_copy = self._deep_copy_numeric(obs_template) if obs_template else {}

        running_projects = obs_copy.get("running_projects", {})
        if not isinstance(running_projects, dict):
            running_projects = {}

        normalized_running = {}
        for i in range(self.max_projects_per_agent):
            project_key = f"project_{i}"
            raw_project = running_projects.get(project_key)
            normalized_running[project_key] = self.encode_project_slot(
                raw_project,
                i,
                is_active=bool(isinstance(raw_project, dict) and len(raw_project) > 0),
            )

        obs_copy["running_projects"] = normalized_running
        return obs_copy

    def _flatten_any_like_template(self, x: Any, tmpl: Any) -> np.ndarray:
        """
        Flatten x using tmpl structure and ordering.

        Rekursiv: für dicts wird sorted() der Keys verwendet für konsistente Ordnung.
        """
        if isinstance(tmpl, dict):
            parts: List[np.ndarray] = []
            if not isinstance(x, dict):
                x = {}
            for k in sorted(tmpl.keys()):
                parts.append(self._flatten_any_like_template(x.get(k, tmpl[k]), tmpl[k]))
            # Ensure concatenation preserves float32 dtype
            return np.concatenate(parts, dtype=np.float32) if parts else np.zeros((0,), dtype=np.float32)

        arr = np.asarray(x)
        if arr.dtype == object:
            raise TypeError(f"Non-numeric object in observation: {type(x)}")
        return arr.astype(np.float32, copy=False).ravel()

    def _flatten_mask_like_template(self, mask: Any, tmpl_mask: Any) -> np.ndarray:
        """
        Flatten action_mask in template order.

        Konvertiert Mask-Werte >0 zu float32 1.0 (valid) für Concatenation.
        """
        if isinstance(tmpl_mask, dict):
            parts: List[np.ndarray] = []
            if not isinstance(mask, dict):
                mask = {}
            for k in sorted(tmpl_mask.keys()):
                v = np.asarray(mask.get(k, tmpl_mask[k]))
                v01 = (v > 0).astype(np.float32).ravel()
                parts.append(v01)
            # Ensure concatenation preserves float32 dtype
            return np.concatenate(parts, dtype=np.float32) if parts else np.zeros((0,), dtype=np.float32)

        v = np.asarray(mask)
        return (v > 0).astype(np.float32).ravel()

    def zeros_from_space(self, space):
        """
        Recursively create zero-filled structure matching a gymnasium Space.

        Wird verwendet zum Erstellen von Beobachtungsvorlagen.
        """
        if isinstance(space, gym.spaces.Dict):
            return {k: self.zeros_from_space(s) for k, s in space.spaces.items()}
        elif isinstance(space, gym.spaces.Box):
            return np.zeros(space.shape, dtype=space.dtype)
        elif isinstance(space, gym.spaces.Discrete):
            return 0
        elif isinstance(space, gym.spaces.MultiDiscrete):
            return np.zeros(len(space.nvec), dtype=np.int64)
        elif isinstance(space, gym.spaces.Tuple):
            return tuple(self.zeros_from_space(s) for s in space.spaces)
        return None

    def _safe_scalar(self, value: Any, default: float = 0.0, dtype=float) -> float:
        """Best-effort extraction of a scalar from numpy/scalar/list-like input."""
        try:
            arr = np.asarray(value)
            if arr.size == 0:
                return dtype(default)
            return dtype(arr.reshape(-1)[0])
        except Exception:
            return dtype(default)

    def _deep_copy_numeric(self, x: Any) -> Any:
        """Deep-copy dict/arrays/scalars into numpy containers."""
        if x is None:
            return None
        if isinstance(x, dict):
            return {k: self._deep_copy_numeric(v) for k, v in x.items()}
        arr = np.asarray(x)
        if arr.dtype == object:
            raise TypeError(f"Non-numeric object encountered while copying: {type(x)}")
        return arr.copy()

