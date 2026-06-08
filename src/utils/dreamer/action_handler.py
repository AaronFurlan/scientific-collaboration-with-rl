"""Action space definition and decoding for DreamerV3."""

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from typing import Any, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class DreamerActionHandler:
    """Defines and decodes action spaces for DreamerV3 (Box/Discrete)."""

    def __init__(
        self,
        n_projects_per_step: int,
        max_projects_per_agent: int,
        max_peer_group_size: int,
        action_space_type: str = "box"
    ):
        """
        Args:
            n_projects_per_step: Max projects to start per step
            max_projects_per_agent: Max active projects per agent
            max_peer_group_size: Peer group size for collaboration
            action_space_type: "box" (default), "discrete", or "multidiscrete"
        """
        self._CP = int(n_projects_per_step + 1)
        self._PE = int(max_projects_per_agent + 1)
        self._CB = int(max_peer_group_size)
        self.action_space_type = action_space_type.lower()
        self.action_space = self._create_action_space()

        logger.info(
            f"DreamerActionHandler initialized: CP={self._CP}, PE={self._PE}, "
            f"CB={self._CB}, action_space_type={self.action_space_type}, "
            f"action_space={self.action_space}"
        )

    def _create_action_space(self) -> gym.Space:
        """Factory method to create appropriate action space."""
        if self.action_space_type == "box":
            action_dim = 2 + self._CB
            return Box(low=0.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        elif self.action_space_type == "discrete":
            total_discrete_size = self._CP * self._PE * (2 ** self._CB)
            logger.warning(
                f"Creating Discrete action space of size {total_discrete_size}. "
                f"This is likely too large! Consider using 'box' instead."
            )
            return Discrete(total_discrete_size)

        elif self.action_space_type == "multidiscrete":
            logger.warning(
                f"Creating MultiDiscrete action space. "
                f"This is NOT compatible with DreamerV3's ActorNetwork!"
            )
            return MultiDiscrete([self._CP, self._PE, 2 ** self._CB])

        else:
            raise ValueError(
                f"Unsupported action_space_type: {self.action_space_type}. "
                f"Must be 'box', 'discrete', or 'multidiscrete'."
            )

    @property
    def CP(self) -> int:
        """Number of discrete choose_project options (including no-op)."""
        return self._CP

    @property
    def PE(self) -> int:
        """Number of discrete put_effort options (including no-op)."""
        return self._PE

    @property
    def CB(self) -> int:
        """Size of collaboration bit vector."""
        return self._CB

    def decode_action(self, action: np.ndarray) -> Dict[str, Any]:
        """
        Decode the action from DreamerV3 into the environment's format.

        Handles different action space types gracefully.

        Args:
            action: Output from DreamerV3 policy (shape varies by action_space_type)

        Returns:
            {
                "choose_project": int in [0, CP-1],
                "put_effort": int in [0, PE-1],
                "collaborate_with": np.ndarray of shape (CB,) with dtype int8
            }
        """
        if self.action_space_type == "box":
            return self._decode_box(action)
        elif self.action_space_type == "discrete":
            return self._decode_discrete(action)
        elif self.action_space_type == "multidiscrete":
            return self._decode_multidiscrete(action)
        else:
            raise NotImplementedError(f"Decoding for {self.action_space_type} not implemented.")

    def _decode_box(self, action: np.ndarray) -> Dict[str, Any]:
        """
        Decode Box action: continuous [0, 1] values to discrete/binary.

        action: np.ndarray of shape (2+CB,) with dtype float32, values in [0, 1]
        - action[0]: choose_project_continuous
        - action[1]: put_effort_continuous
        - action[2:2+CB]: collaborate_with_continuous
        """
        action = np.asarray(action, dtype=np.float32).flatten()

        if action.size < 2 + self._CB:
            # Invalid action, return safe default
            logger.warning(f"Box action too small: {action.size} < {2 + self._CB}")
            return {
                "choose_project": 0,
                "put_effort": 0,
                "collaborate_with": np.zeros(self._CB, dtype=np.int8),
            }

        # 1. choose_project: map [0, 1] -> [0, CP-1]
        # Use floor(value * CP) to get uniform bins, clipped to valid range
        choose_project_cont = float(np.clip(action[0], 0.0, 0.9999))  # Avoid edge case at 1.0
        choose_project = int(np.floor(choose_project_cont * self._CP))
        choose_project = int(np.clip(choose_project, 0, self._CP - 1))

        # 2. put_effort: map [0, 1] -> [0, PE-1]
        put_effort_cont = float(np.clip(action[1], 0.0, 0.9999))
        put_effort = int(np.floor(put_effort_cont * self._PE))
        put_effort = int(np.clip(put_effort, 0, self._PE - 1))

        # 3. collaborate_with: threshold at 0.5 for binary decisions
        collab_cont = np.clip(action[2:2 + self._CB], 0.0, 1.0)
        collab_bits = (collab_cont > 0.5).astype(np.int8)

        return {
            "choose_project": choose_project,
            "put_effort": put_effort,
            "collaborate_with": collab_bits,
        }

    def _decode_discrete(self, action: np.ndarray) -> Dict[str, Any]:
        """
        Decode Discrete action: unflatten single int to three discrete values.

        WARNING: This space can be extremely large, so decoding might be slow!
        """
        action_idx = int(np.asarray(action).item())

        # Unflatten: action_idx = cp_idx + pe_idx * CP + collab_idx * CP * PE
        collab_idx = action_idx // (self._CP * self._PE)
        remainder = action_idx % (self._CP * self._PE)
        pe_idx = remainder // self._CP
        cp_idx = remainder % self._CP

        # Decode collab_idx (binary representation of collaboration bits)
        collab_bits = np.zeros(self._CB, dtype=np.int8)
        for i in range(self._CB):
            if (collab_idx >> i) & 1:
                collab_bits[i] = 1

        return {
            "choose_project": cp_idx,
            "put_effort": pe_idx,
            "collaborate_with": collab_bits,
        }

    def _decode_multidiscrete(self, action: np.ndarray) -> Dict[str, Any]:
        """
        Decode MultiDiscrete action: three separate discrete values.

        action: [choose_project_idx, put_effort_idx, collaborate_bits_idx]
        """
        action = np.asarray(action).flatten()

        if action.size < 3:
            logger.warning(f"MultiDiscrete action too small: {action.size} < 3")
            return {
                "choose_project": 0,
                "put_effort": 0,
                "collaborate_with": np.zeros(self._CB, dtype=np.int8),
            }

        cp_idx = int(action[0]) % self._CP
        pe_idx = int(action[1]) % self._PE
        collab_idx = int(action[2])

        # Decode collab_idx (binary representation)
        collab_bits = np.zeros(self._CB, dtype=np.int8)
        for i in range(self._CB):
            if (collab_idx >> i) & 1:
                collab_bits[i] = 1

        return {
            "choose_project": cp_idx,
            "put_effort": pe_idx,
            "collaborate_with": collab_bits,
        }
