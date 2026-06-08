from ray.rllib.algorithms.callbacks import DefaultCallbacks


class EpisodeMetricsCallback(DefaultCallbacks):
    """RLlib callback exposing per-episode reward/length as custom metrics."""

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        length = getattr(episode, "length", None)
        total_reward = getattr(episode, "total_reward", None)

        if total_reward is None:
            try:
                total_reward = episode.total_reward_for(episode._agent_to_last())
            except Exception:
                try:
                    agent_rewards = getattr(episode, "agent_rewards", None)
                    if agent_rewards:
                        total_reward = sum(sum(v) for v in agent_rewards.values())
                except Exception:
                    total_reward = None

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

        if total_reward is not None:
            episode.custom_metrics["reported_episode_reward"] = total_reward
        if length is not None:
            episode.custom_metrics["reported_episode_length"] = length

        episode.user_data["reported_episode_reward"] = total_reward
        episode.user_data["reported_episode_length"] = length
        print(f"[CALLBACK] on_episode_end: env_index={env_index} reward={total_reward} length={length}")

