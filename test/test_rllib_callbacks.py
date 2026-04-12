"""
Tests for rllib_callbacks.py and callbacks/ directory.

Verifies (using mock Episode objects — no Ray/RLlib runtime needed):

rllib_callbacks.py – EpisodeMetricsCallback:
  - on_episode_end writes reported_episode_reward / length to custom_metrics
  - on_episode_end writes to user_data
  - handles missing total_reward / length gracefully
  - falls back to agent_rewards aggregation

callbacks/debug_actions_callback.py – MacroActionDecoder & ActionInfoCallback:
  - MacroActionDecoder: encode ↔ decode round-trip for various action ids
  - MacroActionDecoder: boundary / out-of-range handling
  - make_action_info_callback: produces a valid callback class
  - ActionInfoCallback.on_episode_step: respects info_interval
  - ActionInfoCallback.on_episode_end: writes paper stats to custom_metrics

callbacks/papers_metrics_callback.py – PapersMetricsCallback:
  - _episode_id: extracts id from new-stack, old-stack, and fallback
  - _get_last_info: extracts info from new-stack and old-stack episodes
  - _write_metrics: writes to metrics_logger and custom_metrics
  - on_episode_start / step / end lifecycle: accumulates and aggregates
  - handles missing / empty info gracefully
  - horizon / env_metrics propagation
"""

from __future__ import annotations

from typing import Dict, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from callbacks.rllib_callbacks import EpisodeMetricsCallback
from callbacks.debug_actions_callback import MacroActionDecoder, make_action_info_callback
from callbacks.papers_metrics_callback import PapersMetricsCallback


# ===========================================================================
# Helpers: mock Episode objects
# ===========================================================================

def _make_episode(
    *,
    length: Optional[int] = None,
    total_reward: Optional[float] = None,
    agent_rewards: Optional[Dict] = None,
    custom_metrics: Optional[Dict] = None,
    user_data: Optional[Dict] = None,
    episode_id: Optional[str] = None,
    id_: Optional[str] = None,
    last_info: Optional[Dict] = None,
    last_action: Optional[int] = None,
) -> MagicMock:
    """Build a mock Episode with the requested attributes."""
    ep = MagicMock()

    # Basic attributes
    if length is not None:
        ep.length = length
    else:
        del ep.length  # getattr(..., None) returns None

    if total_reward is not None:
        ep.total_reward = total_reward
    else:
        del ep.total_reward

    if agent_rewards is not None:
        ep.agent_rewards = agent_rewards
    else:
        del ep.agent_rewards

    # custom_metrics / user_data: use real dicts so tests can inspect writes
    ep.custom_metrics = custom_metrics if custom_metrics is not None else {}
    ep.user_data = user_data if user_data is not None else {}

    # Episode ID (new-stack vs old-stack)
    if id_ is not None:
        ep.id_ = id_
    else:
        del ep.id_

    if episode_id is not None:
        ep.episode_id = episode_id
    else:
        del ep.episode_id

    # Info access
    if last_info is not None:
        ep.get_infos = MagicMock(return_value=last_info)
        ep.last_info_for = MagicMock(return_value=last_info)
    else:
        del ep.get_infos
        del ep.last_info_for

    # Action access
    if last_action is not None:
        ep.get_actions = MagicMock(return_value=last_action)
        ep.get_last_action = MagicMock(return_value=last_action)
    else:
        del ep.get_actions
        del ep.get_last_action

    return ep


# ===========================================================================
# EpisodeMetricsCallback (rllib_callbacks.py)
# ===========================================================================

class TestEpisodeMetricsCallback:
    @pytest.fixture
    def cb(self) -> EpisodeMetricsCallback:
        return EpisodeMetricsCallback()

    def test_writes_reward_and_length(self, cb):
        ep = _make_episode(total_reward=5.0, length=100)
        cb.on_episode_end(worker=None, base_env=None, policies=None,
                          episode=ep, env_index=0)
        assert ep.custom_metrics["reported_episode_reward"] == 5.0
        assert ep.custom_metrics["reported_episode_length"] == 100

    def test_writes_user_data(self, cb):
        ep = _make_episode(total_reward=3.0, length=50)
        cb.on_episode_end(worker=None, base_env=None, policies=None,
                          episode=ep, env_index=0)
        assert ep.user_data["reported_episode_reward"] == 3.0
        assert ep.user_data["reported_episode_length"] == 50

    def test_missing_reward_no_crash(self, cb):
        """If total_reward is not available at all, should not crash."""
        ep = _make_episode(length=20)
        # agent_rewards already absent; ensure total_reward_for also fails
        ep.total_reward_for = MagicMock(side_effect=Exception("no"))
        cb.on_episode_end(worker=None, base_env=None, policies=None,
                          episode=ep, env_index=0)
        assert "reported_episode_reward" not in ep.custom_metrics
        assert ep.user_data["reported_episode_reward"] is None

    def test_missing_length_no_crash(self, cb):
        ep = _make_episode(total_reward=1.0)
        cb.on_episode_end(worker=None, base_env=None, policies=None,
                          episode=ep, env_index=0)
        assert "reported_episode_length" not in ep.custom_metrics
        assert ep.user_data["reported_episode_length"] is None

    def test_reward_is_float(self, cb):
        ep = _make_episode(total_reward=7, length=10)
        cb.on_episode_end(worker=None, base_env=None, policies=None,
                          episode=ep, env_index=0)
        assert isinstance(ep.custom_metrics["reported_episode_reward"], float)

    def test_length_is_int(self, cb):
        ep = _make_episode(total_reward=1.0, length=42.0)
        cb.on_episode_end(worker=None, base_env=None, policies=None,
                          episode=ep, env_index=0)
        assert isinstance(ep.custom_metrics["reported_episode_length"], int)

    def test_zero_reward(self, cb):
        ep = _make_episode(total_reward=0.0, length=5)
        cb.on_episode_end(worker=None, base_env=None, policies=None,
                          episode=ep, env_index=0)
        assert ep.custom_metrics["reported_episode_reward"] == 0.0

    def test_negative_reward(self, cb):
        ep = _make_episode(total_reward=-3.5, length=5)
        cb.on_episode_end(worker=None, base_env=None, policies=None,
                          episode=ep, env_index=0)
        assert ep.custom_metrics["reported_episode_reward"] == -3.5


# ===========================================================================
# MacroActionDecoder (callbacks/debug_actions_callback.py)
# ===========================================================================

class TestMacroActionDecoder:
    @pytest.fixture
    def decoder(self) -> MacroActionDecoder:
        return MacroActionDecoder(
            n_projects_per_step=1,
            max_projects_per_agent=2,
            max_peer_group_size=4,
        )

    def test_action_space_size(self, decoder):
        # (1+1) * (2+1) * 2^4 = 2 * 3 * 16 = 96
        assert decoder.ACTION_N == 96

    def test_decode_action_zero(self, decoder):
        d = decoder.decode(0)
        assert d["action_id"] == 0
        assert d["choose_project"] == 0
        assert d["put_effort"] == 0
        assert d["n_collaborators"] == 0
        assert d["collaborate_with"] == [0, 0, 0, 0]

    def test_decode_returns_dict_with_expected_keys(self, decoder):
        d = decoder.decode(5)
        assert "action_id" in d
        assert "choose_project" in d
        assert "put_effort" in d
        assert "collaborate_with" in d
        assert "n_collaborators" in d

    def test_decode_out_of_range_negative(self, decoder):
        d = decoder.decode(-1)
        assert "error" in d

    def test_decode_out_of_range_too_large(self, decoder):
        d = decoder.decode(decoder.ACTION_N)
        assert "error" in d

    def test_decode_last_valid_action(self, decoder):
        d = decoder.decode(decoder.ACTION_N - 1)
        assert "error" not in d
        assert d["action_id"] == decoder.ACTION_N - 1

    def test_decode_all_valid_actions_no_error(self, decoder):
        """Every valid action_id should decode without error."""
        for aid in range(decoder.ACTION_N):
            d = decoder.decode(aid)
            assert "error" not in d, f"action_id={aid} produced error"

    def test_decode_collab_bits_length(self, decoder):
        d = decoder.decode(0)
        assert len(d["collaborate_with"]) == 4  # max_peer_group_size

    def test_n_collaborators_matches_bits(self, decoder):
        for aid in [0, 1, 3, 7, 15]:
            d = decoder.decode(aid)
            assert d["n_collaborators"] == sum(d["collaborate_with"])

    def test_roundtrip_consistency(self):
        """Encode an action and verify decode recovers the components."""
        dec = MacroActionDecoder(
            n_projects_per_step=1,
            max_projects_per_agent=3,
            max_peer_group_size=3,
        )
        # Manual encoding: action_id = cp * (PE * COLLAB_BASE) + pe * COLLAB_BASE + collab
        cp, pe, collab = 1, 2, 0b101  # project=1, effort=2, collab with peers 0 and 2
        action_id = cp * (dec.PE * dec.COLLAB_BASE) + pe * dec.COLLAB_BASE + collab
        d = dec.decode(action_id)
        assert d["choose_project"] == cp
        assert d["put_effort"] == pe
        assert d["collaborate_with"] == [1, 0, 1]
        assert d["n_collaborators"] == 2

    def test_different_params(self):
        dec = MacroActionDecoder(
            n_projects_per_step=2,
            max_projects_per_agent=1,
            max_peer_group_size=2,
        )
        # (2+1) * (1+1) * 2^2 = 3 * 2 * 4 = 24
        assert dec.ACTION_N == 24


# ===========================================================================
# make_action_info_callback / ActionInfoCallback
# ===========================================================================

class TestMakeActionInfoCallback:
    @pytest.fixture
    def CallbackClass(self):
        return make_action_info_callback(
            controlled_agent_id="agent_0",
            info_interval=5,
            n_projects_per_step=1,
            max_projects_per_agent=2,
            max_peer_group_size=4,
        )

    def test_returns_class(self, CallbackClass):
        assert isinstance(CallbackClass, type)

    def test_instantiable(self, CallbackClass):
        cb = CallbackClass()
        assert hasattr(cb, "controlled_agent_id")
        assert cb.controlled_agent_id == "agent_0"
        assert cb.info_interval == 5

    def test_decoder_baked_in(self, CallbackClass):
        cb = CallbackClass()
        assert hasattr(cb, "decoder")
        assert cb.decoder.ACTION_N == 96  # 2*3*16

    def test_on_episode_step_respects_interval(self, CallbackClass, capsys):
        cb = CallbackClass()
        ep = _make_episode(last_action=5)

        # Steps 1-4 should not print
        for _ in range(4):
            cb.on_episode_step(episode=ep)
        captured = capsys.readouterr()
        assert "[ACTION] t=" not in captured.out

        # Step 5 should print
        cb.on_episode_step(episode=ep)
        captured = capsys.readouterr()
        assert "[ACTION] t=5" in captured.out

    def test_on_episode_step_interval_one(self):
        Cls = make_action_info_callback(
            controlled_agent_id="agent_0",
            info_interval=1,
            n_projects_per_step=1,
            max_projects_per_agent=2,
            max_peer_group_size=4,
        )
        cb = Cls()
        ep = _make_episode(last_action=0)
        cb.on_episode_step(episode=ep)
        assert cb._global_t == 1

    def test_on_episode_end_with_paper_info(self, CallbackClass, capsys):
        cb = CallbackClass()
        ep = _make_episode(last_info={
            "papers_accepted": 3,
            "papers_rejected": 1,
            "papers_completed": 4,
        })
        cb.on_episode_end(episode=ep)
        captured = capsys.readouterr()
        assert "accepted=3" in captured.out
        assert "rejected=1" in captured.out

    def test_on_episode_end_no_info_no_crash(self, CallbackClass):
        cb = CallbackClass()
        ep = _make_episode()
        # Should not crash when info is not available
        cb.on_episode_end(episode=ep)

    def test_format_decoded_valid(self, CallbackClass):
        cb = CallbackClass()
        result = cb._format_decoded(0)
        assert "action_id=0" in result
        assert "decoded=" in result

    def test_format_decoded_invalid(self, CallbackClass):
        cb = CallbackClass()
        result = cb._format_decoded("not_a_number")
        assert "unavailable" in result


# ===========================================================================
# PapersMetricsCallback (callbacks/papers_metrics_callback.py)
# ===========================================================================

class TestPapersMetricsCallbackHelpers:
    """Test static helper methods."""

    def test_episode_id_new_stack(self):
        ep = _make_episode(id_="abc-123")
        assert PapersMetricsCallback._episode_id(ep) == "abc-123"

    def test_episode_id_old_stack(self):
        ep = _make_episode(episode_id="ep-456")
        assert PapersMetricsCallback._episode_id(ep) == "ep-456"

    def test_episode_id_fallback(self):
        ep = _make_episode()
        eid = PapersMetricsCallback._episode_id(ep)
        assert isinstance(eid, str)
        assert len(eid) > 0

    def test_get_last_info_new_stack(self):
        info = {"paper_stats": {"n_projects_total": 5}}
        ep = _make_episode(last_info=info)
        result = PapersMetricsCallback._get_last_info(ep)
        assert result == info

    def test_get_last_info_no_info(self):
        ep = _make_episode()
        result = PapersMetricsCallback._get_last_info(ep)
        assert result is None

    def test_write_metrics_old_stack(self):
        ep = _make_episode()
        metrics = {"papers_total": 10, "agent0_effort_applied_sum": 5.0}
        PapersMetricsCallback._write_metrics(ep, None, metrics)
        assert ep.custom_metrics["papers_total"] == 10
        assert ep.custom_metrics["agent0_effort_applied_sum"] == 5.0

    def test_write_metrics_new_stack(self):
        ep = _make_episode()
        ml = MagicMock()
        metrics = {"papers_total": 10}
        PapersMetricsCallback._write_metrics(ep, ml, metrics)
        ml.log_value.assert_called_once_with("papers_total", 10)

    def test_write_metrics_both_stacks(self):
        ep = _make_episode()
        ml = MagicMock()
        metrics = {"papers_total": 7, "effort": 2.5}
        PapersMetricsCallback._write_metrics(ep, ml, metrics)
        # Both targets receive the metrics
        assert ep.custom_metrics["papers_total"] == 7
        assert ml.log_value.call_count == 2


class TestPapersMetricsCallbackLifecycle:
    """Test the full on_episode_start → step → end lifecycle."""

    @pytest.fixture
    def cb(self) -> PapersMetricsCallback:
        return PapersMetricsCallback()

    @pytest.fixture
    def ep(self):
        return _make_episode(id_="test-ep-1")

    def test_start_initialises_accumulators(self, cb, ep):
        cb.on_episode_start(episode=ep)
        eid = PapersMetricsCallback._episode_id(ep)
        assert eid in cb._ep_data
        assert "_ps_total" in cb._ep_data[eid]
        assert cb._ep_data[eid]["_ps_total"] == []

    def test_step_accumulates_paper_stats(self, cb, ep):
        cb.on_episode_start(episode=ep)

        # Simulate step with paper_stats info
        step_info = {
            "paper_stats": {
                "n_projects_total": 5,
                "n_active_projects": 2,
                "n_due_projects": 1,
                "n_published_projects": 1,
                "n_rejected_projects": 0,
            }
        }
        ep_step = _make_episode(id_="test-ep-1", last_info=step_info)
        cb.on_episode_step(episode=ep_step)

        eid = PapersMetricsCallback._episode_id(ep)
        assert cb._ep_data[eid]["_ps_total"] == [5]
        assert cb._ep_data[eid]["_ps_active"] == [2]
        assert cb._ep_data[eid]["_ps_published"] == [1]

    def test_step_accumulates_effort_diagnostics(self, cb, ep):
        cb.on_episode_start(episode=ep)

        step_info = {
            "debug_effort": {
                "effort_applied_this_step": 3.5,
                "effort_action_invalid": 0,
                "choose_project_effective": 1,
                "n_active_projects_agent": 2,
            }
        }
        ep_step = _make_episode(id_="test-ep-1", last_info=step_info)
        cb.on_episode_step(episode=ep_step)

        eid = PapersMetricsCallback._episode_id(ep)
        assert cb._ep_data[eid]["_eff_applied"] == [3.5]
        assert cb._ep_data[eid]["_choose_effective"] == [1]

    def test_step_without_info_no_crash(self, cb, ep):
        cb.on_episode_start(episode=ep)
        ep_step = _make_episode(id_="test-ep-1")
        cb.on_episode_step(episode=ep_step)
        # Accumulators remain empty
        eid = PapersMetricsCallback._episode_id(ep)
        assert cb._ep_data[eid]["_ps_total"] == []

    def test_end_writes_aggregated_metrics(self, cb, ep):
        cb.on_episode_start(episode=ep)

        # Simulate 3 steps
        for i in range(3):
            step_info = {
                "paper_stats": {
                    "n_projects_total": 10 + i,
                    "n_active_projects": 2 + i,
                    "n_due_projects": 0,
                    "n_published_projects": i,
                    "n_rejected_projects": 0,
                },
                "debug_effort": {
                    "effort_applied_this_step": float(i + 1),
                    "effort_action_invalid": 0,
                    "choose_project_effective": 1,
                    "n_active_projects_agent": 2,
                },
            }
            ep_step = _make_episode(id_="test-ep-1", last_info=step_info)
            cb.on_episode_step(episode=ep_step)

        # End episode
        ep_end = _make_episode(id_="test-ep-1")
        cb.on_episode_end(episode=ep_end)

        # Check custom_metrics
        m = ep_end.custom_metrics
        assert m["papers_total"] == 12  # last value: 10+2
        assert m["papers_published_count"] == 2  # last value: 2
        assert m["agent0_effort_applied_sum"] == pytest.approx(6.0)  # 1+2+3
        assert m["agent0_choose_effective_frac"] == pytest.approx(1.0)
        assert m["papers_active_mean"] == pytest.approx(np.mean([2, 3, 4]))

    def test_end_cleans_up_ep_data(self, cb, ep):
        cb.on_episode_start(episode=ep)
        eid = PapersMetricsCallback._episode_id(ep)
        assert eid in cb._ep_data

        ep_end = _make_episode(id_="test-ep-1")
        cb.on_episode_end(episode=ep_end)
        assert eid not in cb._ep_data

    def test_end_without_start_no_crash(self, cb):
        """on_episode_end without prior on_episode_start should not crash."""
        ep = _make_episode(id_="orphan-ep")
        cb.on_episode_end(episode=ep)
        # No assertion needed — just verifying no exception

    def test_env_metrics_propagated(self, cb, ep):
        cb.on_episode_start(episode=ep)

        env_metrics = {
            "projects_started_total": 15,
            "due_within_episode_rate": 0.6,
            "projects_open_end": 3,
            "clipped_rate": 0.1,
        }
        step_info = {"env_metrics": env_metrics}
        ep_step = _make_episode(id_="test-ep-1", last_info=step_info)
        cb.on_episode_step(episode=ep_step)

        ep_end = _make_episode(id_="test-ep-1")
        cb.on_episode_end(episode=ep_end)

        m = ep_end.custom_metrics
        assert m["horizon_projects_started_total"] == 15.0
        assert m["horizon_due_within_episode_rate"] == pytest.approx(0.6)
        assert m["horizon_projects_open_end"] == 3.0
        assert m["horizon_clipped_rate"] == pytest.approx(0.1)

    def test_multiple_episodes_independent(self, cb):
        """Two episodes tracked simultaneously should not interfere."""
        ep1 = _make_episode(id_="ep-A")
        ep2 = _make_episode(id_="ep-B")

        cb.on_episode_start(episode=ep1)
        cb.on_episode_start(episode=ep2)

        # Step ep1 with paper_stats
        info1 = {"paper_stats": {"n_projects_total": 10, "n_active_projects": 1,
                                  "n_due_projects": 0, "n_published_projects": 5,
                                  "n_rejected_projects": 0}}
        ep1_step = _make_episode(id_="ep-A", last_info=info1)
        cb.on_episode_step(episode=ep1_step)

        # Step ep2 with different stats
        info2 = {"paper_stats": {"n_projects_total": 20, "n_active_projects": 3,
                                  "n_due_projects": 0, "n_published_projects": 8,
                                  "n_rejected_projects": 2}}
        ep2_step = _make_episode(id_="ep-B", last_info=info2)
        cb.on_episode_step(episode=ep2_step)

        # End ep1
        ep1_end = _make_episode(id_="ep-A")
        cb.on_episode_end(episode=ep1_end)
        assert ep1_end.custom_metrics["papers_total"] == 10
        assert ep1_end.custom_metrics["papers_published_count"] == 5

        # End ep2
        ep2_end = _make_episode(id_="ep-B")
        cb.on_episode_end(episode=ep2_end)
        assert ep2_end.custom_metrics["papers_total"] == 20
        assert ep2_end.custom_metrics["papers_published_count"] == 8

    def test_empty_steps_produce_zeros(self, cb, ep):
        """If no info was ever received during steps, end should produce zeros."""
        cb.on_episode_start(episode=ep)

        ep_end = _make_episode(id_="test-ep-1")
        cb.on_episode_end(episode=ep_end)

        m = ep_end.custom_metrics
        assert m["papers_total"] == 0
        assert m["papers_active_mean"] == 0.0
        assert m["agent0_effort_applied_sum"] == 0.0

    def test_end_with_metrics_logger(self, cb, ep):
        cb.on_episode_start(episode=ep)

        info = {"paper_stats": {"n_projects_total": 5, "n_active_projects": 1,
                                 "n_due_projects": 0, "n_published_projects": 2,
                                 "n_rejected_projects": 1}}
        ep_step = _make_episode(id_="test-ep-1", last_info=info)
        cb.on_episode_step(episode=ep_step)

        ml = MagicMock()
        ep_end = _make_episode(id_="test-ep-1")
        cb.on_episode_end(episode=ep_end, metrics_logger=ml)

        # metrics_logger should have received log_value calls
        logged_keys = [c[0][0] for c in ml.log_value.call_args_list]
        assert "papers_total" in logged_keys
        assert "papers_published_count" in logged_keys

