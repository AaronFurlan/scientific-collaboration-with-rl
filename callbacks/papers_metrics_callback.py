"""
papers_metrics_callback.py

RLlib new-API-stack compatible callback that aggregates paper & effort
diagnostics from the env info dict into custom_metrics / metrics_logger.

These metrics are visible in result["custom_metrics"] and therefore in W&B.

Designed to work in multi-worker setups without print-spam:
- Accumulates per-step counters in episode.user_data.
- Writes aggregated values on episode_end.
- No prints (unless PAPERS_CALLBACK_DEBUG env var is set).

Compatible with:
  - RLlib new stack (SingleAgentEpisode, metrics_logger)
  - RLlib old stack (episode.custom_metrics, episode.user_data)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks

# Debug flag: set PAPERS_CALLBACK_DEBUG=1 env var to enable per-episode prints
_DEBUG = os.environ.get("PAPERS_CALLBACK_DEBUG", "0") == "1"


class PapersMetricsCallback(DefaultCallbacks):
    """Aggregate paper_stats and debug_effort from env info into custom metrics.

    Uses an internal dict keyed by episode.id_ because SingleAgentEpisode
    (new API stack) uses __slots__ and has no user_data attribute.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # episode.id_ -> accumulator dict.  Cleaned up in on_episode_end.
        self._ep_data: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Episode start: initialise accumulators
    # ------------------------------------------------------------------
    def on_episode_start(
        self,
        *,
        episode,
        env_runner=None,
        metrics_logger=None,
        env=None,
        env_index=None,
        rl_module=None,
        worker=None,
        base_env=None,
        policies=None,
        **kwargs,
    ):
        eid = self._episode_id(episode)
        self._ep_data[eid] = {
            # Paper stats accumulators (per-step lists for mean/last)
            "_ps_total": [],
            "_ps_active": [],
            "_ps_due": [],
            "_ps_published": [],
            "_ps_rejected": [],
            # Effort diagnostics accumulators
            "_eff_applied": [],
            "_eff_invalid": [],
            "_choose_effective": [],
            "_n_active_agent": [],
            # Horizon / truncation analysis (captured from last info)
            "_env_metrics": None,
        }

    # ------------------------------------------------------------------
    # Episode step: read info and accumulate
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
        worker=None,
        base_env=None,
        policies=None,
        **kwargs,
    ):
        info = self._get_last_info(episode)
        if not isinstance(info, dict):
            return

        eid = self._episode_id(episode)
        ud = self._ep_data.get(eid)
        if ud is None:
            # on_episode_start was not called (should not happen, but be safe)
            return

        # Paper stats
        ps = info.get("paper_stats")
        if isinstance(ps, dict):
            ud["_ps_total"].append(ps.get("n_projects_total", 0))
            ud["_ps_active"].append(ps.get("n_active_projects", 0))
            ud["_ps_due"].append(ps.get("n_due_projects", 0))
            ud["_ps_published"].append(ps.get("n_published_projects", 0))
            ud["_ps_rejected"].append(ps.get("n_rejected_projects", 0))

        # Effort diagnostics
        de = info.get("debug_effort")
        if isinstance(de, dict):
            ud["_eff_applied"].append(
                float(de.get("effort_applied_this_step", 0.0))
            )
            ud["_eff_invalid"].append(
                int(de.get("effort_action_invalid", 0))
            )
            ud["_choose_effective"].append(
                int(de.get("choose_project_effective", 0))
            )
            ud["_n_active_agent"].append(
                int(de.get("n_active_projects_agent", 0))
            )

        # Horizon / truncation analysis — env_metrics is only present in the
        # terminal step's info (injected by the wrapper).  Capture it whenever
        # it appears so it is available in on_episode_end via the accumulator.
        em = info.get("env_metrics")
        if isinstance(em, dict):
            ud["_env_metrics"] = em

    # ------------------------------------------------------------------
    # Episode end: compute aggregates and write custom_metrics
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
        worker=None,
        base_env=None,
        policies=None,
        **kwargs,
    ):
        eid = self._episode_id(episode)
        ud = self._ep_data.pop(eid, None)
        if ud is None:
            return

        # Paper stats: last observed value (snapshot at end of episode)
        ps_total = ud.get("_ps_total", [])
        ps_active = ud.get("_ps_active", [])
        ps_published = ud.get("_ps_published", [])
        ps_rejected = ud.get("_ps_rejected", [])

        papers_total_last = ps_total[-1] if ps_total else 0
        papers_active_mean = float(np.mean(ps_active)) if ps_active else 0.0
        papers_published_last = ps_published[-1] if ps_published else 0
        papers_rejected_last = ps_rejected[-1] if ps_rejected else 0

        # Effort diagnostics
        eff_applied = ud.get("_eff_applied", [])
        eff_invalid = ud.get("_eff_invalid", [])
        choose_eff = ud.get("_choose_effective", [])
        n_active_agent = ud.get("_n_active_agent", [])

        effort_applied_sum = float(np.sum(eff_applied)) if eff_applied else 0.0
        effort_invalid_frac = float(np.mean(eff_invalid)) if eff_invalid else 0.0
        choose_effective_frac = float(np.mean(choose_eff)) if choose_eff else 0.0
        active_projects_agent_mean = float(np.mean(n_active_agent)) if n_active_agent else 0.0

        metrics = {
            "papers_total": papers_total_last,
            "papers_active_mean": papers_active_mean,
            "papers_published_count": papers_published_last,
            "papers_rejected_count": papers_rejected_last,
            "agent0_effort_applied_sum": effort_applied_sum,
            "agent0_effort_invalid_frac": effort_invalid_frac,
            "agent0_choose_effective_frac": choose_effective_frac,
            "agent0_active_projects_mean": active_projects_agent_mean,
        }

        # ------------------------------------------------------------------
        # Episode-level env_metrics (horizon / truncation analysis)
        # Primary source: accumulated during on_episode_step.
        # Fallback: read from the last info dict of the episode object.
        # ------------------------------------------------------------------
        em = ud.get("_env_metrics")
        if em is None:
            last_info = self._get_last_info(episode)
            if isinstance(last_info, dict):
                em = last_info.get("env_metrics")
        if isinstance(em, dict):
            for k, v in em.items():
                metrics[f"horizon_{k}"] = float(v)

        # Write to custom_metrics (old stack) and metrics_logger (new stack)
        self._write_metrics(episode, metrics_logger, metrics)

        if _DEBUG:
            print(
                f"[PAPERS] ep_end: total={papers_total_last} "
                f"active_mean={papers_active_mean:.1f} "
                f"published={papers_published_last} "
                f"rejected={papers_rejected_last} "
                f"effort_sum={effort_applied_sum:.2f} "
                f"invalid_frac={effort_invalid_frac:.3f} "
                f"choose_eff={choose_effective_frac:.3f} "
                f"active_agent_mean={active_projects_agent_mean:.1f}"
            )
            if isinstance(em, dict):
                print(
                    f"[PAPERS] env_metrics: started={em.get('projects_started_total', '?')} "
                    f"due_rate={em.get('due_within_episode_rate', '?'):.3f} "
                    f"open_end={em.get('projects_open_end', '?')} "
                    f"clipped_rate={em.get('clipped_rate', '?'):.3f}"
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _episode_id(episode) -> str:
        """Return a stable episode identifier (works on new & old stack)."""
        # New stack: SingleAgentEpisode has .id_ (str/uuid)
        if hasattr(episode, "id_"):
            return str(episode.id_)
        # Old stack: EpisodeV2 has .episode_id
        if hasattr(episode, "episode_id"):
            return str(episode.episode_id)
        # Last resort: object id
        return str(id(episode))

    @staticmethod
    def _get_last_info(episode) -> Optional[dict]:
        """Extract the last info dict from episode (new & old stack)."""
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
                info = episode.last_info_for()
            except Exception:
                pass
        # Old-stack: episode._last_infos
        if info is None and hasattr(episode, "_last_infos"):
            try:
                infos = episode._last_infos
                if isinstance(infos, dict) and infos:
                    info = next(iter(infos.values()))
            except Exception:
                pass
        return info

    @staticmethod
    def _write_metrics(episode, metrics_logger, metrics: Dict[str, Any]):
        """Write metrics to custom_metrics and/or metrics_logger."""
        # New-stack metrics_logger
        if metrics_logger is not None:
            for k, v in metrics.items():
                try:
                    metrics_logger.log_value(k, v)
                except Exception:
                    pass

        # Old-stack custom_metrics
        if hasattr(episode, "custom_metrics"):
            try:
                for k, v in metrics.items():
                    episode.custom_metrics[k] = v
            except Exception:
                pass


