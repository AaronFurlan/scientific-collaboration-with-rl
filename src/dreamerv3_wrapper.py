"""
dreamerv3_wrapper.py

Single-Agent Gymnasium Wrapper for DreamerV3 in the Game-of-Science Environment.

Goals:
1. Provide a Box action space compatible with RLlib's DreamerV3
   (RLlib DreamerV3 only supports Discrete or Box, NOT MultiDiscrete)
2. Match PPO wrapper observation encoding exactly for fair comparison
3. Implement robust action mask repair (same logic as PPO)
4. Support optional top-k collaboration ablation
5. Ensure reproducible seeding and stable observation sizes

Action Space Design:
- Box(low=0, high=1, shape=(2 + CB,), dtype=float32)
- Continuous values in [0, 1] are discretized during decoding:
  - action[0]: choose_project_continuous -> discretize to [0, n_choose-1]
  - action[1]: put_effort_continuous -> discretize to [0, n_effort-1]
  - action[2:2+CB]: collaborate_with_continuous -> threshold at 0.5 for binary
- Total dimension: 2 + CB (e.g., 42 for CB=40)
- Much more efficient than 18,432 discrete actions

Note on Action Space:
The original design used MultiDiscrete, but RLlib's DreamerV3 ActorNetwork only accepts
Discrete or Box action spaces. Box with discretization provides the same semantic meaning
while being compatible with DreamerV3's continuous actor-critic architecture.

Observation Space Design:
- Flat Box vector matching PPO wrapper's flattened observation
- Uses same running_projects slot encoding with activity flags, progress, urgency, etc.
- Ensures stable size across episodes by padding/normalizing project slots

Action Mask Debugging:
This wrapper supports comprehensive action mask debugging to diagnose invalid actions,
repair logic, and training issues. Enable with --debug-action-mask flag.

Features:
- Validates raw actions from actor before repair
- Tracks which action heads are repaired (choose_project, put_effort, collaborate_with)
- Validates final action after repair (warns if still invalid)
- Logs to console and/or JSONL file
- Configurable logging frequency and initial debug window

CLI Flags:
- --debug-action-mask: Enable debugging
- --debug-action-mask-steps N: Debug first N steps (default: 50)
- --debug-action-mask-interval N: Debug every N steps after initial window (default: 100)
- --debug-action-mask-jsonl PATH: Write JSONL log to PATH

JSONL Fields:
- global_step: Total steps across all episodes
- episode_step: Steps within current episode
- agent_id: Controlled agent ID
- raw_action: Action from actor before repair
- action_mask: Environment-provided validity mask
- raw_validation: Per-head validation results for raw action
- repair_applied: Whether repair was needed
- heads_repaired: List of repaired action heads
- repaired_action: Action after repair
- final_action: Action sent to env.step()
- final_validation: Per-head validation results for final action
- reward, terminated, truncated: Episode outcome
- n_active_projects: Number of active projects for agent
- n_available_projects: Number of available projects to choose from

Usage:
    python scripts/train_dreamerv3.py \\
      --total-env-steps 5000 \\
      --num-gpus 1 \\
      --model-size XS \\
      --debug-action-mask \\
      --debug-action-mask-steps 100 \\
      --debug-action-mask-interval 250 \\
      --debug-action-mask-jsonl debug_action_mask.jsonl

Analysis:
    import pandas as pd
    df = pd.read_json('debug_action_mask.jsonl', lines=True)
    # Check invalid rates
    print(df['raw_validation'].apply(lambda x: not x['all_valid']).mean())
    # Check repair rates per head
    print(df['heads_repaired'].explode().value_counts())
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, List

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

from src.agent_policies import do_nothing_policy
from src.utils.dreamer.observation_handler import DreamerObservationHandler
from src.utils.dreamer.action_handler import DreamerActionHandler
from src.utils.dreamer.action_validator import DreamerActionValidator

logger = logging.getLogger(__name__)

# Suppress Gymnasium dtype warnings - we handle dtype conversion correctly
# These warnings appear during RLlib's initial env validation but our code is correct
warnings.filterwarnings("ignore", message=".*precision lowered by casting to float32.*")
warnings.filterwarnings("ignore", message=".*expecting numpy array dtype to be float32.*")
warnings.filterwarnings("ignore", message=".*is not within the observation space.*")

NestedObs = Dict[str, Any]
ActionDict = Dict[str, Any]


class DreamerV3SingleAgentWrapper(gym.Env):
    """
    Single-agent wrapper for DreamerV3 around PeerGroupEnvironment.

    Converts complex Dict action space to MultiDiscrete for DreamerV3 compatibility.
    Matches PPO wrapper's observation encoding for fair comparison.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        env,
        controlled_agent: str = "agent_0",
        other_policies: Optional[Dict[str, Callable[[Any], Any]]] = None,
        *,
        max_peer_group_size: Optional[int] = None,
        topk_collab: Optional[int] = None,
        topk_mode: str = "score",
        topk_seed: int = 0,
        w_rep: float = 1.0,
        w_dist: float = 1.0,
        w_same: float = 0.5,
        debug_effort: bool = False,
        use_light_policy_obs: bool = False,
        env_config: Optional[Dict[str, Any]] = None,
        debug_action_mask: bool = False,
        debug_action_mask_steps: int = 50,
        debug_action_mask_interval: int = 100,
        debug_action_mask_jsonl: Optional[str] = None,
        invalid_action_penalty: float = 0.0,
        debug_actions: bool = False,
        action_space_type: str = "box",
    ):
        """
        Args:
            env: PeerGroupEnvironment (PettingZoo ParallelEnv)
            controlled_agent: Agent ID to control (default "agent_0")
            other_policies: Dict mapping agent_id -> policy function for non-controlled agents
            max_peer_group_size: Maximum number of peers (for collaborate_with). If None, uses env.max_peer_group_size
            topk_collab: If set, enforce top-k collaboration constraint (like PPO wrapper)
            topk_mode: Scoring mode for top-k ("score" uses reputation+distance+same-group)
            topk_seed: Seed for deterministic tie-breaking in top-k
            w_rep / w_dist / w_same: Weights for collaboration scoring
            debug_effort: Enable debug logging for effort allocation
            use_light_policy_obs: If True, use lightweight observations for non-controlled agents (performance optimization)
            env_config: Optional RLlib env_config
            debug_action_mask: Enable action mask debugging
            debug_action_mask_steps: Number of initial steps to always debug
            debug_action_mask_interval: Debug every N steps after initial period
            debug_action_mask_jsonl: Path to JSONL log file for action mask debugging
            invalid_action_penalty: Reward penalty per invalid action head (e.g., 0.1 means -0.1 per invalid head)
            action_space_type: "box" or "discrete"
        """
        super().__init__()
        self.env = env
        self.controlled_agent = controlled_agent
        self.other_policies = other_policies or {}
        # Use environment's max_peer_group_size if not explicitly provided
        self.max_peer_group_size = max_peer_group_size if max_peer_group_size is not None else env.max_peer_group_size
        self.debug_effort = debug_effort
        self.debug_actions = bool(debug_actions)

        # Action mask debugging configuration
        self.debug_action_mask = bool(env_config.get("debug_action_mask", debug_action_mask))
        self.debug_actions = bool(env_config.get("debug_actions", self.debug_actions))
        self.debug_action_mask_steps = int(env_config.get("debug_action_mask_steps", debug_action_mask_steps))
        self.debug_action_mask_interval = int(debug_action_mask_interval)
        self.debug_action_mask_jsonl = debug_action_mask_jsonl
        self._debug_step_counter = 0
        self._debug_jsonl_file = None

        # Invalid action penalty
        self.invalid_action_penalty = float(invalid_action_penalty)

        # Open JSONL file if path provided
        if self.debug_action_mask and self.debug_action_mask_jsonl:
            log_path = Path(self.debug_action_mask_jsonl)
            # Append worker and vector index to filename to avoid collisions
            unique_filename = f"{log_path.stem}_worker{self.worker_index}_vec{self.vector_index}{log_path.suffix}"
            unique_path = log_path.parent / unique_filename
            self._debug_jsonl_file = open(unique_path, 'w', encoding='utf-8')
            logger.info(f"Action mask debug logging to: {unique_path}")

        # Light observation optimization (same as PPO wrapper)
        if use_light_policy_obs:
            self.env.use_light_policy_obs = True

        # Top-k collaboration config (for fair comparison with PPO)
        self.topk_collab = topk_collab
        self.topk_mode = topk_mode
        self.topk_seed = int(topk_seed)
        self.w_rep = float(w_rep)
        self.w_dist = float(w_dist)
        self.w_same = float(w_same)

        # Seeding
        env_config = env_config or {}
        # Support RLlib EnvContext which has worker_index and vector_index attributes
        self.worker_index = getattr(env_config, "worker_index", int(env_config.get("worker_index", 0)))
        self.vector_index = getattr(env_config, "vector_index", int(env_config.get("vector_index", 0)))
        self.base_seed = int(env_config.get("base_seed", 0))

        self._episode_counter = 0

        # Episode-level metrics
        self._episode_step = 0
        self._episode_env_reward_sum = 0.0
        self._episode_training_reward_sum = 0.0
        self._episode_invalid_action_penalty_sum = 0.0

        self._episode_raw_invalid_any_count = 0
        self._episode_raw_choose_project_invalid_count = 0
        self._episode_raw_put_effort_invalid_count = 0
        self._episode_raw_collaborate_with_invalid_count = 0

        self._episode_repair_any_count = 0
        self._episode_repair_choose_project_count = 0
        self._episode_repair_put_effort_count = 0
        self._episode_repair_collaborate_with_count = 0

        self._episode_choose_project_none_count = 0
        self._episode_choose_project_nonzero_count = 0
        self._episode_put_effort_none_count = 0
        self._episode_put_effort_nonzero_count = 0
        self._episode_collab_count_sum = 0
        self._episode_collab_count_max = 0

        # RL Agent specific stats
        self._episode_started_projects = 0
        self._episode_steps_until_first_reward = None
        self._episode_reward_nonzero_count = 0
        self._rl_agent_rewardless_steps = 0
        self._rl_agent_age = 0
        self._rl_agent_removed = False

        # Effort specific episode metrics
        self._episode_effort_total_count = 0
        self._episode_effort_valid_count = 0
        self._episode_effort_invalid_count = 0

        # Counters for action repair and top-k pruning
        self._repaired_actions = 0
        self._topk_calls = 0
        self._topk_pruned = 0
        self._topk_selected_count_sum = 0

        self._last_observations: Dict[str, Any] = {}
        self._last_actions: Dict[str, Any] = {}
        self._last_obs_stats: Dict[str, Any] = {}

        # Initialize Action Handler
        self._action_handler = DreamerActionHandler(
            n_projects_per_step=self.env.n_projects_per_step,
            max_projects_per_agent=self.env.max_projects_per_agent,
            max_peer_group_size=self.max_peer_group_size,
            action_space_type=action_space_type
        )
        self.action_space = self._action_handler.action_space
        self._CP = self._action_handler.CP
        self._PE = self._action_handler.PE
        self._CB = self._action_handler.CB

        # Initialize Action Validator
        self._action_validator = DreamerActionValidator(
            max_peer_group_size=self.max_peer_group_size,
            n_projects_per_step=self.env.n_projects_per_step,
            max_projects_per_agent=self.env.max_projects_per_agent,
            topk_collab=topk_collab,
            topk_mode=topk_mode,
            topk_seed=topk_seed,
            w_rep=w_rep,
            w_dist=w_dist,
            w_same=w_same,
            debug_steps=debug_action_mask_steps,
        )

        # Initialize Observation Handler
        self._obs_handler = DreamerObservationHandler(
            env=self.env,
            max_projects_per_agent=self.env.max_projects_per_agent,
            max_peer_group_size=self.max_peer_group_size
        )

        # Reference agent for building observation template
        self._ref_agent = self.controlled_agent

        # Build observation space template (same logic as PPO wrapper)
        obs_space = self.env.observation_space(self._ref_agent)
        raw_obs_template = self._obs_handler.zeros_from_space(obs_space)
        self._mask_template = {
            "choose_project": np.zeros(self._CP, dtype=np.int8),
            "collaborate_with": np.zeros(self._CB, dtype=np.int8),
            "put_effort": np.zeros(self._PE, dtype=np.int8),
        }

        # Normalize observation template to ensure all project slots are present
        self._obs_template = self._obs_handler.create_normalized_obs_template(raw_obs_template)
        self._obs_handler.set_templates(self._obs_template, self._mask_template)

        # Store templates locally as well for backwards compatibility
        self._obs_template_local = self._obs_template
        self._mask_template_local = self._mask_template

        # Create sample flattened vector
        nested_template = {
            "observation": self._obs_template,
            "action_mask": self._mask_template,
            "_is_template": True
        }
        sample_vec = self._obs_handler.flatten_to_vector(nested_template)

        # Set observation space (use float32 arrays to avoid dtype warnings)
        obs_dim = int(sample_vec.size)
        self.observation_space = Box(
            low=np.full(obs_dim, -np.inf, dtype=np.float32),
            high=np.full(obs_dim, np.inf, dtype=np.float32),
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.expected_obs_size = int(sample_vec.size)
        logger.info(
            "DreamerV3 wrapper initialized: obs_size=%d, action_space=%s, CP=%d, PE=%d, CB=%d",
            self.expected_obs_size,
            str(self.action_space),
            self._CP,
            self._PE,
            self._CB
        )

    # -------------------------------------------------------------------------
    # Peer collaboration intents
    # -------------------------------------------------------------------------


    def _extract_peer_collaboration_intents(self, rl_agent_idx: int) -> List[int]:
        """
        Extract collaboration intents from peer agents towards the RL agent.
        Only includes agents in the same peer group as the RL agent.

        Returns a list where each element indicates if that peer wants to
        collaborate with the RL agent (1=yes, 0=no).

        Args:
            rl_agent_idx: The index of the RL agent in the environment

        Returns:
            List of binary values indicating peer collaboration intents
        """
        peer_intents = []

        if not self._last_actions or rl_agent_idx is None:
            return []

        # Get the peer group for the RL agent
        peer_group_idx = self.env.agent_peer_idx[rl_agent_idx]
        peer_group = self.env.peer_groups[peer_group_idx]

        # Find RL agent's position in peer_group
        rl_peer_idx = None
        for idx, agent_idx in enumerate(peer_group):
            if agent_idx == rl_agent_idx:
                rl_peer_idx = idx
                break

        if rl_peer_idx is None:
            return []

        # Iterate through peer group agents (excluding RL agent)
        for agent_idx in peer_group:
            # Skip the RL agent itself
            if agent_idx == rl_agent_idx:
                continue

            # Find agent_id from agent_idx
            agent_id = None
            for aid, idx in self.env.agent_to_id.items():
                if idx == agent_idx:
                    agent_id = aid
                    break

            if agent_id is None:
                peer_intents.append(0)
                continue

            # Get the action for this peer
            action = self._last_actions.get(agent_id, {})
            collab_with = action.get("collaborate_with", np.zeros(self._CB, dtype=np.int8))

            # Check if this peer wants to collaborate with the RL agent
            # The peer uses rl_peer_idx to indicate collaboration with RL agent
            if rl_peer_idx < len(collab_with):
                peer_intents.append(int(collab_with[rl_peer_idx]))
            else:
                peer_intents.append(0)

        return peer_intents

    def _debug_action_mask_event(
        self,
        *,
        global_step: int,
        episode_step: int,
        agent_id: str,
        raw_action: ActionDict,
        action_mask: Dict[str, Any],
        raw_validation: Dict[str, bool],
        repaired_action: ActionDict,
        repair_applied: bool,
        heads_repaired: List[str],
        final_action: ActionDict,
        final_validation: Dict[str, bool],
        reward: float,
        terminated: bool,
        truncated: bool,
        n_active_projects: int = 0,
        n_available_projects: int = 0,
        peer_collab_with_rl: List[int] = None,
    ):
        """
        Log action mask debug event to console and/or JSONL file.

        This is called when:
        - Step is within first debug_action_mask_steps
        - Step is at debug_action_mask_interval intervals
        - Raw action was invalid or repair was applied
        """
        # Determine if we should log this event
        should_log = False
        if self._debug_step_counter < self.debug_action_mask_steps:
            should_log = True
        elif self._debug_step_counter % self.debug_action_mask_interval == 0:
            should_log = True
        elif not raw_validation["all_valid"] or repair_applied:
            should_log = True

        if not should_log:
            return

        # Build event dict
        # Helper to convert dicts with numpy bools to native Python types
        def to_json_serializable(obj):
            """Recursively convert numpy types to native Python types."""
            if isinstance(obj, dict):
                return {k: to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [to_json_serializable(item) for item in obj]
            elif isinstance(obj, (np.bool_, np.generic)):
                return obj.item()
            else:
                return obj

        event = {
            "global_step": global_step,
            "episode_step": episode_step,
            "agent_id": agent_id,
            "raw_action": {
                "choose_project": int(raw_action.get("choose_project", 0)),
                "put_effort": int(raw_action.get("put_effort", 0)),
                "collaborate_with": raw_action.get("collaborate_with", np.zeros(self._CB, dtype=np.int8)).tolist(),
                "n_collaborators": int(np.sum(raw_action.get("collaborate_with", np.zeros(self._CB, dtype=np.int8)))),
            },
            "action_mask": {
                "choose_project": action_mask.get("choose_project", []).tolist() if hasattr(action_mask.get("choose_project", []), "tolist") else list(action_mask.get("choose_project", [])),
                "put_effort": action_mask.get("put_effort", []).tolist() if hasattr(action_mask.get("put_effort", []), "tolist") else list(action_mask.get("put_effort", [])),
                "collaborate_with": action_mask.get("collaborate_with", []).tolist() if hasattr(action_mask.get("collaborate_with", []), "tolist") else list(action_mask.get("collaborate_with", [])),
            },
            "raw_validation": to_json_serializable(raw_validation),
            "repair_applied": bool(repair_applied),
            "heads_repaired": heads_repaired,
            "repaired_action": {
                "choose_project": int(repaired_action.get("choose_project", 0)),
                "put_effort": int(repaired_action.get("put_effort", 0)),
                "collaborate_with": repaired_action.get("collaborate_with", np.zeros(self._CB, dtype=np.int8)).tolist(),
                "n_collaborators": int(np.sum(repaired_action.get("collaborate_with", np.zeros(self._CB, dtype=np.int8)))),
            },
            "final_action": {
                "choose_project": int(final_action.get("choose_project", 0)),
                "put_effort": int(final_action.get("put_effort", 0)),
                "collaborate_with": final_action.get("collaborate_with", np.zeros(self._CB, dtype=np.int8)).tolist(),
                "n_collaborators": int(np.sum(final_action.get("collaborate_with", np.zeros(self._CB, dtype=np.int8)))),
            },
            "final_validation": to_json_serializable(final_validation),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "n_active_projects": int(n_active_projects),
            "n_available_projects": int(n_available_projects),
            "peer_collab_with_rl": peer_collab_with_rl if peer_collab_with_rl is not None else [],
        }

        # Console logging
        if not raw_validation["all_valid"] or repair_applied:
            # Always print invalid/repaired actions
            print(f"[ActionMask] Step {global_step} (ep {episode_step}): "
                  f"raw_valid={raw_validation['all_valid']} "
                  f"repaired={repair_applied} "
                  f"heads={heads_repaired} "
                  f"cp={raw_action.get('choose_project')}→{final_action.get('choose_project')} "
                  f"pe={raw_action.get('put_effort')}→{final_action.get('put_effort')} "
                  f"final_valid={final_validation['all_valid']}")
        elif self._debug_step_counter < self.debug_action_mask_steps or \
             self._debug_step_counter % self.debug_action_mask_interval == 0:
            # Print summary for interval logs
            print(f"[ActionMask] Step {global_step} (ep {episode_step}): "
                  f"valid={raw_validation['all_valid']} "
                  f"cp={final_action.get('choose_project')} "
                  f"pe={final_action.get('put_effort')} "
                  f"n_collab={event['final_action']['n_collaborators']}")

        # JSONL logging
        if self._debug_jsonl_file:
            try:
                json.dump(event, self._debug_jsonl_file)
                self._debug_jsonl_file.write('\n')
                self._debug_jsonl_file.flush()
            except Exception as e:
                logger.warning(f"Failed to write JSONL debug event: {e}")

    # -------------------------------------------------------------------------
    # Action decoding
    # -------------------------------------------------------------------------

    def _decode_action(self, action: np.ndarray, agent_id: Optional[str] = None) -> ActionDict:
        """
        Decode continuous Box action into environment action dict.

        DreamerV3 outputs continuous values in [0, 1]. We discretize them:

        Input: np.ndarray of shape (2 + CB,) with continuous values in [0, 1]:
          - action[0]: choose_project_continuous -> discretize to [0, CP-1]
          - action[1]: put_effort_continuous -> discretize to [0, PE-1]
          - action[2:2+CB]: collaborate_with_continuous -> threshold at 0.5 for binary

        Output: {"choose_project": int, "put_effort": int, "collaborate_with": np.ndarray}
        """
        action = np.asarray(action, dtype=np.float32).flatten()

        if action.size < 2 + self._CB:
            # Invalid action, return safe default
            return {
                "choose_project": 0,
                "put_effort": 0,
                "collaborate_with": np.zeros(self._CB, dtype=np.int8),
            }

        # Discretize continuous values with proper rounding
        # This ensures the environment receives clean integer values

        # choose_project: map [0, 1] -> [0, CP-1]
        # Use floor(value * CP) to get uniform bins
        choose_project_cont = float(np.clip(action[0], 0.0, 0.9999))  # Avoid edge case at 1.0
        choose_project = int(np.floor(choose_project_cont * self._CP))
        choose_project = int(np.clip(choose_project, 0, self._CP - 1))

        # put_effort: map [0, 1] -> [0, PE-1]
        put_effort_cont = float(np.clip(action[1], 0.0, 0.9999))
        put_effort = int(np.floor(put_effort_cont * self._PE))
        put_effort = int(np.clip(put_effort, 0, self._PE - 1))

        # collaborate_with: threshold at 0.5 for binary decisions
        collab_cont = np.clip(action[2:2 + self._CB], 0.0, 1.0)
        collab_bits = (collab_cont > 0.5).astype(np.int8)

        # Debug logging for continuous->discrete conversion
        if self._debug_step_counter < self.debug_action_mask_steps:
            n_collab_selected = int(np.sum(collab_bits))
            logger.debug(
                f"[Decode] cont=[{choose_project_cont:.3f}, {put_effort_cont:.3f}, collab_sum={np.sum(collab_cont):.1f}] "
                f"→ discrete=[cp={choose_project}, pe={put_effort}, n_collab={n_collab_selected}]"
            )

        # Ensure all outputs are proper integers, not numpy types that might cause issues
        assert isinstance(choose_project, int), f"choose_project must be int, got {type(choose_project)}"
        assert isinstance(put_effort, int), f"put_effort must be int, got {type(put_effort)}"
        assert collab_bits.dtype == np.int8, f"collab_bits must be int8, got {collab_bits.dtype}"

        return {
            "choose_project": choose_project,
            "put_effort": put_effort,
            "collaborate_with": collab_bits,
        }

    def _apply_action_mask(
        self,
        decoded: ActionDict,
        nested_obs: NestedObs,
        agent_id: Optional[str] = None,
        debug_step: int = 0,
    ) -> tuple[ActionDict, bool, List[str]]:
        """
        Repair invalid actions using the action validator.

        Delegates to DreamerActionValidator.repair_action() for consistent logic.
        """
        return self._action_validator.repair_action(
            decoded,
            nested_obs,
            agent_id=agent_id,
            debug_step=debug_step
        )

    # -------------------------------------------------------------------------
    # Top-k collaboration helpers (same as PPO wrapper)
    # -------------------------------------------------------------------------

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
            return np.full((length,), float(default_value), dtype=np.float32)

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

    def _compute_peer_scores(self, nested_obs: NestedObs, agent_id: Optional[str], k: int, seed: int) -> np.ndarray:
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

    # -------------------------------------------------------------------------
    # DEPRECATED: These methods are now in observation_handler
    # Keeping as thin wrappers for backwards compatibility during transition
    # -------------------------------------------------------------------------

    def _safe_scalar(self, value: Any, default: float = 0.0, dtype=float):
        """Deprecated: Use _obs_handler._safe_scalar instead"""
        return self._obs_handler._safe_scalar(value, default, dtype)

    def _encode_project_slot(self, project_obs: Optional[Dict[str, Any]], slot_idx: int, *, is_active: bool) -> Dict[str, Any]:
        """Deprecated: Use _obs_handler.encode_project_slot instead"""
        return self._obs_handler.encode_project_slot(project_obs, slot_idx, is_active=is_active)

    def _create_normalized_obs_template(self, obs_template: Dict[str, Any]) -> Dict[str, Any]:
        """Deprecated: Use _obs_handler.create_normalized_obs_template instead"""
        return self._obs_handler.create_normalized_obs_template(obs_template)

    def _flatten_to_vector(self, nested_obs: Any) -> np.ndarray:
        """Deprecated: Use _obs_handler.flatten_to_vector instead"""
        return self._obs_handler.flatten_to_vector(nested_obs)

    def _normalize_observation(self, obs_part: Dict[str, Any]) -> Dict[str, Any]:
        """Deprecated: Use _obs_handler.normalize_observation instead"""
        return self._obs_handler.normalize_observation(obs_part)

    def _flatten_any_like_template(self, x: Any, tmpl: Any) -> np.ndarray:
        """Deprecated: Use _obs_handler._flatten_any_like_template instead"""
        return self._obs_handler._flatten_any_like_template(x, tmpl)

    def _flatten_mask_like_template(self, mask: Any, tmpl_mask: Any) -> np.ndarray:
        """Deprecated: Use _obs_handler._flatten_mask_like_template instead"""
        return self._obs_handler._flatten_mask_like_template(mask, tmpl_mask)

    def _zeros_from_space(self, space):
        """Deprecated: Use _obs_handler.zeros_from_space instead"""
        return self._obs_handler.zeros_from_space(space)

    def _deep_copy_numeric(self, x: Any) -> Any:
        """Deprecated: Use _obs_handler._deep_copy_numeric instead"""
        return self._obs_handler._deep_copy_numeric(x)

    # -------------------------------------------------------------------------
    # Gymnasium API
    # -------------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """Reset environment and return initial observation."""
        self._episode_counter += 1
        # Reset episode-level metrics
        self._episode_step = 0
        self._episode_env_reward_sum = 0.0
        self._episode_training_reward_sum = 0.0
        self._episode_invalid_action_penalty_sum = 0.0

        self._episode_raw_invalid_any_count = 0
        self._episode_raw_choose_project_invalid_count = 0
        self._episode_raw_put_effort_invalid_count = 0
        self._episode_raw_collaborate_with_invalid_count = 0

        self._episode_repair_any_count = 0
        self._episode_repair_choose_project_count = 0
        self._episode_repair_put_effort_count = 0
        self._episode_repair_collaborate_with_count = 0

        self._episode_choose_project_none_count = 0
        self._episode_choose_project_nonzero_count = 0
        self._episode_put_effort_none_count = 0
        self._episode_put_effort_nonzero_count = 0
        self._episode_collab_count_sum = 0
        self._episode_collab_count_max = 0

        # RL Agent specific stats
        self._episode_started_projects = 0
        self._episode_steps_until_first_reward = None
        self._episode_reward_nonzero_count = 0
        self._rl_agent_rewardless_steps = 0
        self._rl_agent_age = 0
        self._rl_agent_removed = False

        self._episode_effort_total_count = 0
        self._episode_effort_valid_count = 0
        self._episode_effort_invalid_count = 0

        # Generate deterministic seed for training consistency
        ss = np.random.SeedSequence([
            self.base_seed,
            self.worker_index,
            self.vector_index,
            self._episode_counter,
        ])
        episode_seed = int(ss.generate_state(1, dtype=np.uint32)[0])

        observations, infos = self.env.reset(seed=episode_seed, options=options)

        # If light observations are enabled, we need to rebuild the FULL observation
        # for our controlled agent (light obs are only for non-controlled agents)
        if getattr(self.env, "use_light_policy_obs", False):
            if self.controlled_agent in observations:
                # Re-build FULL observation for controlled agent
                full_obs = self.env._get_observation(self.controlled_agent)
                observations[self.controlled_agent]["observation"] = full_obs
                # Update env's internal storage
                self.env.observations[self.controlled_agent] = full_obs

        self._last_observations = observations

        if self.controlled_agent not in observations:
            raise RuntimeError(f"Controlled agent {self.controlled_agent} not in env observations")

        nested = observations[self.controlled_agent]
        obs_vec = self._obs_handler.flatten_to_vector(nested)
        obs_vec = self._ensure_obs_vector_ok(obs_vec, where="reset")

        info = infos.get(self.controlled_agent, {})
        if info is None:
            info = {}

        info["episode_seed"] = episode_seed
        info["episode_counter"] = self._episode_counter

        logger.debug(
            "DreamerV3 wrapper reset: episode=%d, seed=%d, obs_size=%d",
            self._episode_counter,
            episode_seed,
            obs_vec.size
        )

        return obs_vec, info

    def step(self, action):
        """Execute one environment step."""
        if self.controlled_agent is None:
            raise RuntimeError("Wrapper not reset() before step().")

        # Increment step counters
        self._debug_step_counter += 1
        self._episode_step += 1

        # Extract raw continuous actions for logging
        raw_cont_choose_project = float(action[0])
        raw_cont_put_effort = float(action[1])
        raw_cont_collab = action[2:].astype(float)

        # Decode controlled agent's action
        decoded_original = self._action_handler.decode_action(action)
        nested_obs = self._last_observations[self.controlled_agent]
        action_mask = nested_obs.get("action_mask", {})

        # Debug actions
        if self.debug_actions:
            print(f"\n--- DEBUG ACTIONS (Step {self._episode_step}) ---")
            print(f"Agent: {self.controlled_agent}")
            
            obs = nested_obs.get("observation", {})
            mask = nested_obs.get("action_mask", {})

            # 1. Action: Choose Project
            raw_choose = decoded_original['choose_project']
            print(f"\n[ACTION: choose_project] (RL Agent raw choice: {raw_choose})")
            print("  Observations (project_opportunities):")
            proj_opps = obs.get("project_opportunities", {})
            if not proj_opps:
                print("    - None -")
            else:
                for p_id, p_obs in proj_opps.items():
                    print(f"    {p_id}: req_effort={p_obs['required_effort'][0]}, prestige={p_obs['prestige'][0]:.2f}, novelty={p_obs['novelty'][0]:.2f}")
            print(f"  Mask (can_choose): {mask.get('can_choose', [])}")

            # 2. Action: Put Effort
            raw_put = decoded_original['put_effort']
            print(f"\n[ACTION: put_effort] (RL Agent raw choice: {raw_put})")
            print("  Observations (running_projects):")
            running_projs = obs.get("running_projects", {})
            if not running_projs:
                print("    - None -")
            else:
                for p_id, p_obs in running_projs.items():
                    print(f"    {p_id}: req_effort={p_obs['required_effort'][0]}, current_effort={p_obs['current_effort'][0]}, time_left={p_obs['time_left'][0]}")
            print(f"  Mask (can_put_effort): {mask.get('can_put_effort', [])}")

            # 3. Action: Collaborate With
            collab_before = decoded_original['collaborate_with']
            collab_indices = np.where(collab_before == 1)[0]
            print(f"\n[ACTION: collaborate_with] (RL Agent raw choice: {collab_indices.tolist()})")
            print("  Observations (peer_group):")
            peer_group = obs.get("peer_group", [])
            reputations = obs.get("peer_reputation", [])
            active_peers = []
            for i, active in enumerate(peer_group):
                if active:
                    rep = reputations[i] if i < len(reputations) else 0.0
                    active_peers.append(f"idx {i} (rep={rep:.2f})")
            
            if not active_peers:
                 print("    - No active peers -")
            else:
                 # Print in chunks to avoid too many lines
                 for i in range(0, len(active_peers), 5):
                     print("    " + ", ".join(active_peers[i:i+5]))
            print(f"  Mask (can_collaborate): {mask.get('can_collaborate', [])}")
            
            # General State
            print(f"\n[STATE] Age: {obs.get('age', [0])[0]} | Acc. Reward: {obs.get('accumulated_rewards', [0.0])[0]:.2f}")

        # Store original action for tracking
        raw_action = {
            "choose_project": int(decoded_original["choose_project"]),
            "put_effort": int(decoded_original["put_effort"]),
            "collaborate_with": decoded_original["collaborate_with"].copy() if hasattr(decoded_original["collaborate_with"], "copy") else decoded_original["collaborate_with"]
        }

        # Always validate raw action (bugfix: always needed for metrics/penalty)
        needs_action_validation = (
            self.debug_action_mask
            or self.invalid_action_penalty > 0.0
            or True  # always want validity metrics
        )

        raw_validation = self._action_validator.validate_action_against_mask(decoded_original, action_mask)

        # Apply action mask (may repair invalid actions)
        decoded, repair_applied, heads_repaired = self._apply_action_mask(
            decoded_original,
            nested_obs,
            agent_id=self.controlled_agent,
            debug_step=self._debug_step_counter
        )

        if self.debug_actions:
            print(f"\n[SUMMARY] Final Action (after mask):")
            print(f"  choose_project: {decoded['choose_project']}")
            print(f"  put_effort: {decoded['put_effort']}")
            collab_final_indices = np.where(decoded['collaborate_with'] == 1)[0]
            print(f"  collaborate_with: {collab_final_indices.tolist()}")
            print("-" * 35)


        # Track if action was repaired (for effort metrics)
        action_was_repaired = repair_applied

        # Build actions dict for all agents
        actions_dict = {self.controlled_agent: decoded}

        # Execute policies for non-controlled agents
        for agent_id in self.env.agents:
            if agent_id == self.controlled_agent:
                continue

            if agent_id not in self._last_observations:
                continue

            agent_obs = self._last_observations[agent_id]

            # Robustness check for heuristic policies
            if not isinstance(agent_obs, dict) or agent_obs.get("observation") is None:
                actions_dict[agent_id] = {
                    "choose_project": 0,
                    "put_effort": 0,
                    "collaborate_with": np.zeros(self._CB, dtype=np.int8)
                }
                continue

            if agent_id in self.other_policies:
                try:
                    policy = self.other_policies[agent_id]
                    action_out = policy(agent_obs)
                    if action_out is not None:
                        actions_dict[agent_id] = action_out
                    else:
                        actions_dict[agent_id] = {
                            "choose_project": 0,
                            "put_effort": 0,
                            "collaborate_with": np.zeros(self._CB, dtype=np.int8)
                        }
                except Exception as e:
                    logger.error(f"Policy for agent {agent_id} failed: {e}")
                    actions_dict[agent_id] = {
                        "choose_project": 0,
                        "put_effort": 0,
                        "collaborate_with": np.zeros(self._CB, dtype=np.int8)
                    }
            else:
                # No policy: do nothing
                actions_dict[agent_id] = {
                    "choose_project": 0,
                    "put_effort": 0,
                    "collaborate_with": np.zeros(self._CB, dtype=np.int8)
                }

        # Validate final action before sending to environment
        final_validation = self._action_validator.validate_action_against_mask(decoded, action_mask)

        # Update episode-level action validity counters
        if not raw_validation["all_valid"]:
            self._episode_raw_invalid_any_count += 1
        if not raw_validation["choose_project_valid"]:
            self._episode_raw_choose_project_invalid_count += 1
        if not raw_validation["put_effort_valid"]:
            self._episode_raw_put_effort_invalid_count += 1
        if not raw_validation["collaborate_with_valid"]:
            self._episode_raw_collaborate_with_invalid_count += 1

        if repair_applied:
            self._episode_repair_any_count += 1
        if "choose_project" in heads_repaired:
            self._episode_repair_choose_project_count += 1
        if "put_effort" in heads_repaired:
            self._episode_repair_put_effort_count += 1
        if "collaborate_with" in heads_repaired:
            self._episode_repair_collaborate_with_count += 1

        # Action distribution counters
        if decoded["choose_project"] == 0:
            self._episode_choose_project_none_count += 1
        else:
            self._episode_choose_project_nonzero_count += 1
        
        if decoded["put_effort"] == 0:
            self._episode_put_effort_none_count += 1
        else:
            self._episode_put_effort_nonzero_count += 1
        
        collab_count = int(np.sum(decoded["collaborate_with"]))
        self._episode_collab_count_sum += collab_count
        self._episode_collab_count_max = max(self._episode_collab_count_max, collab_count)

        if self.debug_action_mask and not final_validation.get("all_valid", True):
            # WARNING: Final action is still invalid after repair!
            warning_msg = (f"WARNING: Final action INVALID after repair! "
                          f"cp_valid={final_validation.get('choose_project_valid')} "
                          f"pe_valid={final_validation.get('put_effort_valid')} "
                          f"c_valid={final_validation.get('collaborate_with_valid')} "
                          f"step={self._debug_step_counter}")
            print(f"[ActionMask] {warning_msg}")
            logger.warning(warning_msg)

        # Store actions for debugging
        self._last_actions = actions_dict

        # Track started projects
        if decoded["choose_project"] > 0:
            # A project is started if choose_project is > 0 and it's valid
            # (Note: environmental validation happens in env.step, but here we track intent)
            self._episode_started_projects += 1

        # Environment step
        observations, rewards, terminations, truncations, infos = self.env.step(actions_dict)

        # If light observations are enabled, we need to rebuild the FULL observation
        # for our controlled agent (light obs are only for non-controlled agents)
        if getattr(self.env, "use_light_policy_obs", False):
            if self.controlled_agent in observations:
                # Re-build FULL observation for controlled agent
                full_obs = self.env._get_observation(self.controlled_agent)
                observations[self.controlled_agent]["observation"] = full_obs
                # Update env's internal storage
                self.env.observations[self.controlled_agent] = full_obs

        self._last_observations = observations

        # Extract controlled agent's outcome
        if self.controlled_agent in observations:
            nested = observations[self.controlled_agent]
            obs_vec = self._obs_handler.flatten_to_vector(nested)
            obs_vec = self._ensure_obs_vector_ok(obs_vec, where="step")
            env_reward = float(rewards.get(self.controlled_agent, 0.0))
            terminated = bool(terminations.get(self.controlled_agent, False))
            truncated = bool(truncations.get(self.controlled_agent, False))
            info = infos.get(self.controlled_agent, {})
        else:
            # Agent is no longer active (died/terminated)
            obs_vec = np.zeros(self.expected_obs_size, dtype=np.float32)
            env_reward = float(rewards.get(self.controlled_agent, 0.0))
            terminated = True
            truncated = False
            info = {}

        # Update reward tracking
        if env_reward > 0:
            self._episode_reward_nonzero_count += 1
            if self._episode_steps_until_first_reward is None:
                self._episode_steps_until_first_reward = self._episode_step

        # Update agent status from env internal state
        try:
            agent_idx = self.env.agent_to_id.get(self.controlled_agent)
            if agent_idx is not None:
                self._rl_agent_rewardless_steps = self.env.rewardless_steps[agent_idx]
                self._rl_agent_age = self.env.agent_steps[agent_idx]
                if self.env.active_agents[agent_idx] == 0:
                    self._rl_agent_removed = True
        except:
            pass

        # Apply invalid action penalty
        invalid_action_penalty_val = 0.0
        n_invalid_heads = 0
        if self.invalid_action_penalty > 0.0:
            if not raw_validation.get("choose_project_valid", True):
                n_invalid_heads += 1
            if not raw_validation.get("put_effort_valid", True):
                n_invalid_heads += 1
            if not raw_validation.get("collaborate_with_valid", True):
                n_invalid_heads += 1

            if n_invalid_heads > 0:
                invalid_action_penalty_val = -self.invalid_action_penalty * n_invalid_heads

        training_reward = env_reward + invalid_action_penalty_val
        
        # Update episode-level reward sums
        self._episode_env_reward_sum += env_reward
        self._episode_training_reward_sum += training_reward
        self._episode_invalid_action_penalty_sum += invalid_action_penalty_val

        info["reward_components"] = {
            "env_reward": float(env_reward),
            "invalid_action_penalty": float(invalid_action_penalty_val),
            "training_reward": float(training_reward),
        }
        info["episode_reward_components"] = {
            "env_reward_sum": float(self._episode_env_reward_sum),
            "invalid_action_penalty_sum": float(self._episode_invalid_action_penalty_sum),
            "training_reward_sum": float(self._episode_training_reward_sum),
        }

        # Action validity metrics
        info["action_validity"] = {
            "raw_choose_project_invalid": int(not raw_validation["choose_project_valid"]),
            "raw_put_effort_invalid": int(not raw_validation["put_effort_valid"]),
            "raw_collaborate_with_invalid": int(not raw_validation["collaborate_with_valid"]),
            "raw_any_invalid": int(not raw_validation["all_valid"]),

            "repaired_choose_project": int("choose_project" in heads_repaired),
            "repaired_put_effort": int("put_effort" in heads_repaired),
            "repaired_collaborate_with": int("collaborate_with" in heads_repaired),
            "repaired_any": int(repair_applied),

            "final_all_valid": int(final_validation["all_valid"]),
        }

        # Episode-level rates
        ep_step = max(1, self._episode_step)
        info["action_validity_episode"] = {
            "raw_invalid_rate": self._episode_raw_invalid_any_count / ep_step,
            "raw_choose_project_invalid_rate": self._episode_raw_choose_project_invalid_count / ep_step,
            "raw_put_effort_invalid_rate": self._episode_raw_put_effort_invalid_count / ep_step,
            "raw_collaborate_with_invalid_rate": self._episode_raw_collaborate_with_invalid_count / ep_step,
            "repair_rate": self._episode_repair_any_count / ep_step,
            "repair_choose_project_rate": self._episode_repair_choose_project_count / ep_step,
            "repair_put_effort_rate": self._episode_repair_put_effort_count / ep_step,
            "repair_collaborate_with_rate": self._episode_repair_collaborate_with_count / ep_step,
        }

        # Action distribution metrics
        info["action_distribution"] = {
            "choose_project": int(decoded["choose_project"]),
            "put_effort": int(decoded["put_effort"]),
            "choose_project_is_none": int(decoded["choose_project"] == 0),
            "choose_project_is_nonzero": int(decoded["choose_project"] > 0),
            "put_effort_is_none": int(decoded["put_effort"] == 0),
            "put_effort_is_nonzero": int(decoded["put_effort"] > 0),
            "collab_count": int(np.sum(decoded["collaborate_with"])),
            "raw_cont_choose_project": float(raw_cont_choose_project),
            "raw_cont_put_effort": float(raw_cont_put_effort),
            "raw_cont_collab_mean": float(np.mean(raw_cont_collab)),
            "raw_cont_collab_max": float(np.max(raw_cont_collab)),
            "raw_cont_collab_min": float(np.min(raw_cont_collab)),
        }

        info["action_distribution_episode"] = {
            "choose_project_none_rate": self._episode_choose_project_none_count / ep_step,
            "choose_project_nonzero_rate": self._episode_choose_project_nonzero_count / ep_step,
            "put_effort_none_rate": self._episode_put_effort_none_count / ep_step,
            "put_effort_nonzero_rate": self._episode_put_effort_nonzero_count / ep_step,
            "collab_count_mean": self._episode_collab_count_sum / ep_step,
            "collab_count_max": float(self._episode_collab_count_max),
        }

        # Inject paper stats if available (for PapersMetricsCallback)
        try:
            agent_idx = self.env.agent_to_id.get(self.controlled_agent)
            if agent_idx is not None:
                # Count active projects for the controlled agent
                n_active_projects_agent = sum(
                    1 for pid in self.env.agent_active_projects[agent_idx]
                    if pid is not None
                )

                # Individual agent paper stats
                completed = int(self.env.agent_completed_projects[agent_idx])
                accepted_pids = self.env.agent_successful_projects[agent_idx]
                accepted = len(accepted_pids)
                rejected = completed - accepted

                info["rl_agent_projects"] = {
                    "active_projects": int(n_active_projects_agent),
                    "completed_projects": int(completed),
                    "accepted_projects": int(accepted),
                    "rejected_projects": int(rejected),
                }

                # Global paper stats
                n_published_projects = sum(len(pids) for pids in self.env.agent_successful_projects)
                n_rejected_projects = sum(self.env.agent_completed_projects) - n_published_projects

                # Count active projects (projects in progress)
                n_active_projects = sum(
                    1 for project in self.env.projects.values()
                    if project is not None and not project.finished
                )

                info["global_projects"] = {
                    "active_projects": int(n_active_projects),
                    "published_projects": int(n_published_projects),
                    "rejected_projects": int(n_rejected_projects),
                }

                # RL Agent detailed stats
                info["rl_agent_stats"] = {
                    "episode_return": float(self._episode_env_reward_sum),
                    "accumulated_reward": float(self._episode_env_reward_sum), # User asked for both
                    "started_projects": int(self._episode_started_projects),
                    "steps_until_first_reward": float(self._episode_steps_until_first_reward) if self._episode_steps_until_first_reward is not None else 0.0,
                    "reward_nonzero_frac": float(self._episode_reward_nonzero_count / max(1, self._episode_step)),
                    "age_at_done": int(self._rl_agent_age),
                    "rewardless_steps": int(self._rl_agent_rewardless_steps),
                    "agent_removed": int(self._rl_agent_removed),
                    "termination_reason": 1 if self._rl_agent_removed else (2 if truncated else 0),
                }

                # Count due projects (active projects near deadline - within 10 steps)
                n_due_projects = sum(
                    1 for project in self.env.projects.values()
                    if project is not None and not project.finished and
                       project.get_time_remaining(self.env.timestep) <= 10
                )

                info["paper_stats"] = {
                    "n_active_projects": n_active_projects,
                    "n_due_projects": n_due_projects,
                    "n_published_projects": n_published_projects,
                    "n_rejected_projects": n_rejected_projects,
                }

                # Add detailed effort metrics (for controlled agent)
                # Count active projects for the controlled agent
                n_active_projects_agent = sum(
                    1 for pid in self.env.agent_active_projects[agent_idx]
                    if pid is not None
                )

                # Calculate effort applied this step
                effort_applied = 0.0
                effort_action_invalid = 0
                if decoded["put_effort"] > 0:
                    # Check if effort action was valid
                    if decoded["put_effort"] <= n_active_projects_agent:
                        # Valid effort allocation
                        selected_project_idx = decoded["put_effort"] - 1
                        effort_project_id = self.env.agent_active_projects[agent_idx][selected_project_idx]
                        if effort_project_id is not None and effort_project_id in self.env.projects:
                            project = self.env.projects[effort_project_id]
                            if project is not None:
                                # Get the effort amount (peer_fit for this agent)
                                try:
                                    contributors_idx = list(project.contributors).index(agent_idx)
                                    effort_applied = float(project.peer_fit[contributors_idx])
                                except (ValueError, IndexError):
                                    effort_applied = 0.0
                    else:
                        # Invalid effort action (repaired by mask)
                        effort_action_invalid = 1

                # Check if choose_project was effective (started a new project)
                choose_effective = 0
                if decoded["choose_project"] > 0 and n_active_projects_agent < self.env.max_projects_per_agent:
                    choose_effective = 1  # Agent had capacity and chose a project

                # Track action repair
                if action_was_repaired:
                    effort_action_invalid = 1

                info["debug_effort"] = {
                    "effort_applied_this_step": effort_applied,
                    "effort_action_invalid": effort_action_invalid,
                    "choose_project_effective": choose_effective,
                    "n_active_projects_agent": n_active_projects_agent,
                }

                # Add effort analysis metrics (prestige and deadline analysis)
                if decoded["put_effort"] > 0 and n_active_projects_agent > 0:
                    # Get agent's active projects
                    active_pids = [pid for pid in self.env.agent_active_projects[agent_idx] if pid is not None]
                    if active_pids:
                        # Get prestige and remaining time for each active project
                        active_projects_data = []
                        for pid in active_pids:
                            if pid in self.env.projects:
                                proj = self.env.projects[pid]
                                if proj is not None:
                                    active_projects_data.append({
                                        'prestige': proj.prestige,
                                        'remaining_time': proj.get_time_remaining(self.env.timestep)
                                    })

                        if active_projects_data and decoded["put_effort"] <= len(active_projects_data):
                            # Get chosen project
                            chosen_idx = decoded["put_effort"] - 1
                            chosen_data = active_projects_data[chosen_idx]

                            # Get max/min values across all active projects
                            max_prestige = max(p['prestige'] for p in active_projects_data)
                            min_remaining_time = min(p['remaining_time'] for p in active_projects_data)

                            info["effort_analysis"] = {
                                "effort_idx": decoded["put_effort"],
                                "chosen_prestige": float(chosen_data['prestige']),
                                "max_prestige": float(max_prestige),
                                "chose_max_prestige": int(chosen_data['prestige'] >= max_prestige),
                                "chosen_remaining_time": float(chosen_data['remaining_time']),
                                "min_remaining_time": float(min_remaining_time),
                                "chose_most_urgent": int(chosen_data['remaining_time'] <= min_remaining_time),
                            }

                # Add effort total/valid/invalid counts
                if not hasattr(self, '_effort_total_count'):
                    self._effort_total_count = 0
                    self._effort_invalid_count = 0
                    self._effort_valid_count = 0

                if decoded["put_effort"] > 0:
                    self._effort_total_count += 1
                    if effort_action_invalid:
                        self._effort_invalid_count += 1
                    else:
                        self._effort_valid_count += 1

                info["effort_total_count"] = self._effort_total_count
                info["effort_invalid_count"] = self._effort_invalid_count
                info["effort_valid_count"] = self._effort_valid_count

                # # DEBUG: Print first few times to verify data
                # if not hasattr(self, '_debug_paper_stats_count'):
                #     self._debug_paper_stats_count = 0
                # if self._debug_paper_stats_count < 3:
                #     print(f"[DEBUG step] paper_stats: total={n_projects_total}, active={n_active_projects}, "
                #           f"published={n_published_projects}, rejected={n_rejected_projects}")
                #     print(f"[DEBUG step] debug_effort: n_active_projects_agent={n_active_projects_agent}")
                #     self._debug_paper_stats_count += 1

        except Exception as e:
            print(f"[DEBUG step] Exception setting paper_stats: {e}")
            import traceback
            traceback.print_exc()

        # Debug action mask logging
        if self.debug_action_mask:
            try:
                # Get active projects count
                agent_idx = self.env.agent_to_id.get(self.controlled_agent)
                n_active_projects = 0
                n_available_projects = 0
                if agent_idx is not None:
                    n_active_projects = sum(
                        1 for pid in self.env.agent_active_projects[agent_idx]
                        if pid is not None
                    )
                    # n_available_projects is choose_project mask size - 1 (excluding "no project" option)
                    cp_mask = np.asarray(action_mask.get("choose_project", []))
                    n_available_projects = int(np.sum(cp_mask > 0)) - 1 if cp_mask.size > 0 else 0

                # Calculate episode step (approximate, since we don't track resets in step counter)
                # For now, use global step counter as episode step
                episode_step = self._debug_step_counter

                # Extract peer collaboration intents towards the RL agent
                peer_collaboration_with_rl_agent = self._extract_peer_collaboration_intents(agent_idx)

                self._debug_action_mask_event(
                    global_step=self._debug_step_counter,
                    episode_step=episode_step,
                    agent_id=self.controlled_agent,
                    raw_action=raw_action,
                    action_mask=action_mask,
                    raw_validation=raw_validation,
                    repaired_action=decoded,
                    repair_applied=repair_applied,
                    heads_repaired=heads_repaired,
                    final_action=decoded,
                    final_validation=final_validation,
                    reward=float(training_reward),
                    terminated=terminated,
                    truncated=truncated,
                    n_active_projects=n_active_projects,
                    n_available_projects=n_available_projects,
                    peer_collab_with_rl=peer_collaboration_with_rl_agent,
                )
            except Exception as e:
                logger.warning(f"Failed to log action mask debug event: {e}")

        # Observation statistics
        info["observation_stats"] = self._last_obs_stats.copy()

        return obs_vec, training_reward, terminated, truncated, info

    def render(self):
        """Render environment."""
        return self.env.render()

    def close(self):
        """Close environment and log statistics."""
        if self._repaired_actions > 0:
            logger.info("DreamerV3 wrapper: repaired %d invalid actions during episode", self._repaired_actions)

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

        # Close JSONL debug file if open
        if self._debug_jsonl_file is not None:
            try:
                self._debug_jsonl_file.close()
                logger.info(f"Closed action mask debug JSONL file: {self.debug_action_mask_jsonl}")
            except Exception as e:
                logger.warning(f"Error closing JSONL debug file: {e}")

        return self.env.close()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _ensure_obs_vector_ok(self, obs_vec: Any, *, where: str) -> np.ndarray:
        """Ensure observation is a 1D float32 vector matching observation_space length."""
        # Convert to numpy array first
        if not isinstance(obs_vec, np.ndarray):
            obs_vec = np.asarray(obs_vec, dtype=np.float32)

        # Ensure it's 1D
        if obs_vec.ndim != 1:
            obs_vec = obs_vec.ravel()

        # Calculate statistics BEFORE any transformations
        nan_count = int(np.isnan(obs_vec).sum())
        inf_count = int(np.isinf(obs_vec).sum())
        clean_obs = obs_vec[np.isfinite(obs_vec)]
        if clean_obs.size > 0:
            abs_max = float(np.max(np.abs(clean_obs)))
            mean_val = float(np.mean(clean_obs))
            std_val = float(np.std(clean_obs))
        else:
            abs_max = 0.0
            mean_val = 0.0
            std_val = 0.0

        self._last_obs_stats = {
            "abs_max": abs_max,
            "mean": mean_val,
            "std": std_val,
            "nan_count": nan_count,
            "inf_count": inf_count,
        }

        # Convert to float32 BEFORE checking for NaN/Inf (avoid dtype mismatch warnings)
        obs_vec = obs_vec.astype(np.float32, copy=False)

        # Check for NaN/Inf after dtype conversion
        if not np.all(np.isfinite(obs_vec)):
            # logger.warning("NaN/Inf detected in observation at %s, replacing with zeros", where)
            obs_vec = np.nan_to_num(obs_vec, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

        expected_len = int(self.observation_space.shape[0])

        if obs_vec.size != expected_len:
            if obs_vec.size < expected_len:
                pad = np.zeros((expected_len - obs_vec.size,), dtype=np.float32)
                obs_vec = np.concatenate([obs_vec, pad])
            else:
                obs_vec = obs_vec[:expected_len]

        # Final assertion: ensure dtype is exactly float32
        assert obs_vec.dtype == np.float32, f"obs_vec dtype is {obs_vec.dtype}, expected float32"

        return obs_vec
