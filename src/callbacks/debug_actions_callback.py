from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks


# ---------------------------------------------------------------------------
# Standalone decoder – mirrors RLLibSingleAgentWrapper._decode_action
# but needs no env reference (works in any Ray worker process).
# ---------------------------------------------------------------------------

class BoxActionDecoder:
    """Decode a flat Box action (numpy array) into a human-readable dict."""

    def __init__(self, n_projects_per_step: int, max_projects_per_agent: int,
                 max_peer_group_size: int):
        self.CP = int(n_projects_per_step) + 1
        self.PE = int(max_projects_per_agent) + 1
        self.CB = int(max_peer_group_size)

    def decode(self, action: Any) -> Dict[str, Any]:
        # Handle the case where action might be a list or tuple from RLlib
        if isinstance(action, (list, tuple)):
            action = np.asarray(action)

        if isinstance(action, np.ndarray):
            # If it's a batch or has extra dims, take the first one if it matches the expected length
            if action.ndim > 1:
                action = action.flatten()
            
            if len(action) == (self.CB + 2):
                choose_project = int(np.round(action[0]))
                put_effort = int(np.round(action[1]))
                collab_bits = np.round(action[2:]).astype(np.int8)
                return {
                    "choose_project": choose_project,
                    "put_effort": put_effort,
                    "collaborate_with": collab_bits.tolist(),
                    "n_collaborators": int(np.sum(collab_bits)),
                }
            else:
                return {"error": f"length mismatch: got {len(action)} expected {self.CB + 2}", "val": str(action)}
        
        # Fallback for integer actions (if any)
        if isinstance(action, (int, np.integer)):
            a = int(action)
            collab_base = 1 << self.CB
            collab_code = a % collab_base
            a //= collab_base
            put_effort = a % self.PE
            a //= self.PE
            choose_project = a % self.CP
            return {
                "choose_project": choose_project,
                "put_effort": put_effort,
                "n_collaborators": bin(collab_code).count('1'),
            }

        return {"error": f"unsupported action type: {type(action)}", "val": str(action)}


# ---------------------------------------------------------------------------
# Callback factory
# ---------------------------------------------------------------------------

def make_action_info_callback(
    controlled_agent_id: str,
    info_interval: int,
    *,
    n_projects_per_step: int = 1,
    max_projects_per_agent: int = 6,
    max_peer_group_size: int = 8,
):
    """Factory returning a callback class that logs *decoded* actions.

    The encoding parameters must match those used by ``RLLibSingleAgentWrapper``.
    They are baked into the callback at construction time so that decoding works
    in every Ray worker process without needing access to the wrapper instance.

    Compatible with **RLlib v2 new-stack** (``samples`` = list of Episode
    objects) and the old stack (``SampleBatch``).
    """

    # Build the decoder once; it will be serialised into every worker.
    _decoder = BoxActionDecoder(
        n_projects_per_step=n_projects_per_step,
        max_projects_per_agent=max_projects_per_agent,
        max_peer_group_size=max_peer_group_size,
    )

    class ActionInfoCallback(DefaultCallbacks):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.controlled_agent_id = controlled_agent_id
            self.info_interval = max(1, int(info_interval))
            self._global_t = 0
            self._printed_header = False
            self.decoder = _decoder

        def _format_decoded(self, action: Any) -> str:
            """Return ``action=<val> decoded={...}`` string."""
            try:
                decoded = self.decoder.decode(action)
            except Exception as e:
                return f"action={action} decoded=<error: {e}>"
            return f"action={action} decoded={decoded}"

        # ------------------------------------------------------------------
        # on_episode_step – per env.step(); primary per-step logging hook
        # ------------------------------------------------------------------
        def on_episode_step(
            self,
            *,
            episode,
            env_runner=None,
            metrics_logger=None,
            env=None,
            env_index=None,
            rl_module=None,
            # legacy kwargs
            worker=None,
            base_env=None,
            policies=None,
            **kwargs,
        ):
            self._global_t += 1
            if not self._printed_header:
                print(
                    f"[ACTION] on_episode_step enabled: interval={self.info_interval}"
                )
                self._printed_header = True
            
            # NOTE: Per-step logging of individual actions is now handled directly 
            # within RLLibSingleAgentWrapper.step() to ensure the action is 
            # captured at the correct time in RLlib 2.x.
            return

        # ------------------------------------------------------------------
        # on_episode_end – log paper acceptance stats for controlled agent
        # ------------------------------------------------------------------
        def on_episode_end(
            self,
            *,
            episode,
            env_runner=None,
            metrics_logger=None,
            env=None,
            env_index=None,
            rl_module=None,
            # legacy kwargs
            worker=None,
            base_env=None,
            policies=None,
            **kwargs,
        ):
            info = None
            # New-stack: episode.get_infos(-1)
            if hasattr(episode, "get_infos"):
                try:
                    info = episode.get_infos(-1)
                except Exception:
                    pass
            # Old-stack: episode.last_info_for()
            if info is None and hasattr(episode, "last_info_for"):
                try:
                    info = episode.last_info_for(self.controlled_agent_id)
                except Exception:
                    pass

            if not isinstance(info, dict):
                return

            accepted = info.get("papers_accepted")
            rejected = info.get("papers_rejected")
            completed = info.get("papers_completed")

            if accepted is not None:
                print(
                    f"[PAPERS] {self.controlled_agent_id}: "
                    f"accepted={accepted} rejected={rejected} "
                    f"completed={completed}"
                )

                # Log as custom metrics (visible in RLlib results & wandb)
                if metrics_logger is not None:
                    try:
                        metrics_logger.log_value("agent0_papers_accepted", accepted)
                        metrics_logger.log_value("agent0_papers_rejected", rejected)
                        metrics_logger.log_value("agent0_papers_completed", completed)
                    except Exception:
                        pass
                # Old-stack custom_metrics fallback
                if hasattr(episode, "custom_metrics"):
                    try:
                        episode.custom_metrics["agent0_papers_accepted"] = accepted
                        episode.custom_metrics["agent0_papers_rejected"] = rejected
                        episode.custom_metrics["agent0_papers_completed"] = completed
                    except Exception:
                        pass

        # ------------------------------------------------------------------
        # on_sample_end – batch-level summary
        # ------------------------------------------------------------------
        def on_sample_end(
            self,
            *,
            env_runner=None,
            metrics_logger=None,
            samples=None,
            # legacy kwargs
            worker=None,
            **kwargs,
        ):
            if samples is None:
                return

            actions_list: list = []

            # ---- New-stack path: samples is List[Episode] ----
            if isinstance(samples, list):
                for ep in samples:
                    if hasattr(ep, "get_actions"):
                        try:
                            acts = ep.get_actions(slice(None))
                            if acts is not None:
                                if isinstance(acts, np.ndarray):
                                    actions_list.extend(acts.tolist())
                                elif isinstance(acts, (list, tuple)):
                                    actions_list.extend(acts)
                                else:
                                    actions_list.append(acts)
                        except Exception:
                            pass

            # ---- Old-stack path: SampleBatch / MultiAgentBatch ----
            else:
                sb = samples
                if hasattr(samples, "policy_batches"):
                    try:
                        sb = samples.policy_batches.get("default_policy")
                        if sb is None and len(samples.policy_batches) > 0:
                            sb = next(iter(samples.policy_batches.values()))
                    except Exception:
                        return
                if sb is None:
                    return
                try:
                    actions = sb["actions"]
                    actions_list = list(actions)
                except Exception:
                    return

            if not actions_list:
                return

            # Compact summary with decoded top actions
            total = len(actions_list)
            # Use string representation for unique as Box actions are numpy arrays
            str_actions = [str(a) for a in actions_list]
            unique_str, indices, counts = np.unique(str_actions, return_index=True, return_counts=True)
            top_k = min(5, len(unique_str))
            sorted_idx = np.argsort(-counts)[:top_k]

            top_entries = []
            for i in sorted_idx:
                action = actions_list[indices[i]]
                cnt = int(counts[i])
                try:
                    dec = self.decoder.decode(action)
                    top_entries.append(
                        f"cnt={cnt} "
                        f"proj={dec['choose_project']} "
                        f"effort={dec['put_effort']} "
                        f"collab={dec['n_collaborators']}"
                    )
                except Exception:
                    top_entries.append(f"val={action} cnt={cnt}")

            summary = " | ".join(top_entries)
            print(
                f"[ACTION] on_sample_end: {total} actions sampled | top: {summary}"
            )

    return ActionInfoCallback
