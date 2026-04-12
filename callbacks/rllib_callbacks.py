from ray.rllib.algorithms.callbacks import DefaultCallbacks


class EpisodeMetricsCallback(DefaultCallbacks):
    """RLlib callback that ensures per-episode reward/length are exposed as custom metrics.

    This makes RLlib aggregate them into result['custom_metrics'] / result['info'] so
    you can always see per-episode stats even if the collector otherwise doesn't
    populate the top-level `episode_*` keys for short tests.
    """

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        # Try common attributes first (RLlib provides these in most versions)
        length = getattr(episode, "length", None)
        total_reward = getattr(episode, "total_reward", None)

        # Fallback: some RLlib versions expose get_total_reward() or total_reward_for
        if total_reward is None:
            try:
                total_reward = episode.total_reward_for(episode._agent_to_last())
            except Exception:
                # best-effort: use aggregate of per-agent rewards (if available)
                try:
                    # episode.agent_rewards is a dict(agent_id -> list of rewards) in some versions
                    agent_rewards = getattr(episode, "agent_rewards", None)
                    if agent_rewards:
                        total_reward = sum(sum(v) for v in agent_rewards.values())
                except Exception:
                    total_reward = None

        # Ensure numeric types when possible
        try:
            if length is not None:
                length = int(length)
        except Exception:
            length = None
        try:
            if total_reward is not None:
                total_reward = float(total_reward)
        except Exception:
            total_reward = None

        # Insert into custom_metrics so RLlib aggregates them.
        if total_reward is not None:
            episode.custom_metrics["reported_episode_reward"] = total_reward
        if length is not None:
            episode.custom_metrics["reported_episode_length"] = length

        # Also add to user_data for debugging if needed
        episode.user_data["reported_episode_reward"] = total_reward
        episode.user_data["reported_episode_length"] = length

        # Visible print for worker logs (helps debugging in subprocess logs)
        print(f"[CALLBACK] on_episode_end: env_index={env_index} reward={total_reward} length={length}")

