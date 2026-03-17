from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks


# ---------------------------------------------------------------------------
# Standalone decoder – mirrors RLLibSingleAgentWrapper._decode_action
# but needs no env reference (works in any Ray worker process).
# ---------------------------------------------------------------------------

class MacroActionDecoder:
    """Decode a discrete macro-action id into a human-readable dict.

    Encoding layout (least-significant first):
        action_id = choose_project * (PE * COLLAB_BASE) + put_effort * COLLAB_BASE + collab_code
    where COLLAB_BASE = 2 ** CB.
    """

    def __init__(self, n_projects_per_step: int, max_projects_per_agent: int,
                 max_peer_group_size: int):
        self.CP = int(n_projects_per_step) + 1      # choose_project choices
        self.PE = int(max_projects_per_agent) + 1  # put_effort choices
        self.CB = int(max_peer_group_size)         # collaboration bits
        self.COLLAB_BASE = 1 << self.CB
        self.ACTION_N = self.CP * self.PE * self.COLLAB_BASE

    def decode(self, action_id: int) -> Dict[str, Any]:
        a = int(action_id)
        if a < 0 or a >= self.ACTION_N:
            return {"action_id": a, "error": f"out of range [0, {self.ACTION_N})"}

        collab_code = a % self.COLLAB_BASE
        a //= self.COLLAB_BASE
        put_effort = a % self.PE
        a //= self.PE
        choose_project = a % self.CP

        collab_bits = [(collab_code >> i) & 1 for i in range(self.CB)]
        n_collab = sum(collab_bits)

        return {
            "action_id": int(action_id),
            "choose_project": choose_project,
            "put_effort": put_effort,
            "collaborate_with": collab_bits,
            "n_collaborators": n_collab,
        }


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
    _decoder = MacroActionDecoder(
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

        def _format_decoded(self, action_id: Any) -> str:
            """Return ``action_id=<id> decoded={...}`` string."""
            try:
                aid = int(action_id)
            except Exception:
                return f"action_id={action_id} decoded=<unavailable>"
            try:
                decoded = self.decoder.decode(aid)
            except Exception:
                return f"action_id={aid} decoded=<unavailable>"
            return f"action_id={aid} decoded={decoded}"

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
            if self._global_t % self.info_interval != 0:
                return

            if not self._printed_header:
                print(
                    f"[ACTION] on_episode_step enabled: "
                    f"interval={self.info_interval} "
                    f"action_space_n={self.decoder.ACTION_N}"
                )
                self._printed_header = True

            action = None

            # --- New-stack Episode API (SingleAgentEpisode) ---
            if hasattr(episode, "get_actions"):
                try:
                    action = episode.get_actions(-1)
                except Exception:
                    pass

            # --- Old-stack Episode API ---
            if action is None:
                try:
                    if hasattr(episode, "get_last_action"):
                        action = episode.get_last_action()
                except Exception:
                    pass

            if action is None:
                try:
                    if hasattr(episode, "last_action_for"):
                        action = episode.last_action_for(self.controlled_agent_id)
                except Exception:
                    pass

            print(f"[ACTION] t={self._global_t} {self._format_decoded(action)}")

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
            unique, counts = np.unique(actions_list, return_counts=True)
            top_k = min(5, len(unique))
            sorted_idx = np.argsort(-counts)[:top_k]

            top_entries = []
            for i in sorted_idx:
                aid = int(unique[i])
                cnt = int(counts[i])
                try:
                    dec = self.decoder.decode(aid)
                    top_entries.append(
                        f"id={aid} cnt={cnt} "
                        f"proj={dec['choose_project']} "
                        f"effort={dec['put_effort']} "
                        f"collab={dec['n_collaborators']}"
                    )
                except Exception:
                    top_entries.append(f"id={aid} cnt={cnt}")

            summary = " | ".join(top_entries)
            print(
                f"[ACTION] on_sample_end: {total} actions sampled | top: {summary}"
            )

    return ActionInfoCallback
