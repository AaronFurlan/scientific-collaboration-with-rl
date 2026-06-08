import json
import pandas as pd
import os
import argparse
import re
from datetime import datetime
from tqdm import tqdm

def find_log_files_for_seed(base_dir, seed):
    """Find actions and observations files for given seed in subdirectories."""
    if not os.path.exists(base_dir):
        return []

    candidates = []

    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path):
            files = os.listdir(full_path)
            actions_file = None
            obs_file = None

            for f in files:
                if f.endswith("_actions.jsonl") and f"s{seed}" in f:
                    actions_file = os.path.join(full_path, f)
                elif f.endswith("_observations.jsonl") and f"s{seed}" in f:
                    obs_file = os.path.join(full_path, f)

            if actions_file and obs_file:
                candidates.append((actions_file, obs_file))

    return candidates

def build_reward_dataframe(reward_steps, agents, seed):
    """Build DataFrame of accumulated rewards per agent per step with archetype."""
    agent_archetype = {}
    for a in agents:
        for k, v in a.items():
            if v is not None:
                agent_archetype[k] = v.get("archetype", "rl_agent")

    records = []
    for step_idx, step in enumerate(reward_steps):
        for agent_id, data in step.items():
            if data is not None:
                obs = data.get("observation", None)
                if obs and "accumulated_rewards" in obs:
                    archetype = agent_archetype.get(agent_id, None)
                    if archetype is not None:
                        records.append({
                            "step": step_idx,
                            "archetype": archetype,
                            "agent_id": agent_id,
                            "accumulated_rewards": obs["accumulated_rewards"][0],
                            "h_index": obs["peer_h_index"][0],
                            "age": obs["age"][0],
                            "seed": seed,
                        })

    return pd.DataFrame(records)

def build_reward_summary_by_archetype(reward_steps, agents, seed, strategy):
    """Build summary DataFrame with mean/std rewards per archetype per step, using ffill for dead agents."""
    agent_archetype = {}
    for a in agents:
        if not isinstance(a, dict):
            continue
        for agent_id, v in a.items():
            if v is None:
                continue
            if isinstance(v, dict):
                # Heuristic agents have 'archetype', our controlled RL agent might not
                agent_archetype[agent_id] = v.get("archetype", "rl_agent")

    if not agent_archetype:
        raise ValueError("No archetypes found in agents data.")

    # Track last known reward per agent for forward-filling
    agent_last_reward = {}

    records = []
    for step_idx, step in enumerate(reward_steps):
        if not isinstance(step, dict):
            continue

        arch_rewards = {}

        for agent_id, data in step.items():
            if agent_id not in agent_archetype:
                # Fallback
                agent_archetype[agent_id] = "Unknown"

            archetype = agent_archetype[agent_id]

            if data is None:
                # Agent is dead/inactive - use last known reward (ffill)
                if agent_id in agent_last_reward:
                    reward = agent_last_reward[agent_id]
                    arch_rewards.setdefault(archetype, []).append(reward)
                continue

            obs = data.get("observation", {}) if isinstance(data, dict) else {}
            if "accumulated_rewards" not in obs:
                # No reward data - try to use last known
                if agent_id in agent_last_reward:
                    reward = agent_last_reward[agent_id]
                    arch_rewards.setdefault(archetype, []).append(reward)
                continue

            reward = obs["accumulated_rewards"][0]
            agent_last_reward[agent_id] = reward  # Update last known reward
            arch_rewards.setdefault(archetype, []).append(reward)

        for archetype, rewards in arch_rewards.items():
            s = pd.Series(rewards)
            records.append({
                "step": step_idx,
                "archetype": archetype,
                "mean_reward": float(s.mean()),
                "std_reward": float(s.std(ddof=1)) if len(rewards) > 1 else 0.0,
                "n_agents": len(rewards),
                "seed": int(seed),
                "strategy": strategy,
            })

    return pd.DataFrame(records)

def main():
    parser = argparse.ArgumentParser(description="Process RL simulation log files and generate summary Parquet files.")
    parser.add_argument("--log-dir", type=str, default="test_results", help="Path to the directory containing the simulation log subdirectories (default: 'test_results')")
    parser.add_argument("--out-base-dir", type=str, default="results", help="Base directory where the result Parquet files will be saved (default: 'results')")
    parser.add_argument("--start-seed", type=int, default=501, help="Starting seed for evaluation (default: 501)")
    parser.add_argument("--num-seeds", type=int, default=20, help="Number of seeds to evaluate (default: 20)")
    parser.add_argument("--strategy-name", type=str, default="random", help="Name to use for strategy in output files (default: 'random')")

    args = parser.parse_args()

    seeds = range(args.start_seed, args.start_seed + args.num_seeds)

    log_dir = args.log_dir
    strategy_name = args.strategy_name

    # Create timestamped subfolder in results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_base_dir, f"aggregation_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Aggregation started. Results will be saved to: {out_dir}")

    all_summaries = []
    all_rewards = []

    print(f"\nProcessing seeds for strategy: {strategy_name}")

    for seed in tqdm(seeds, desc="Seeds", unit="seed"):
        file_pairs = find_log_files_for_seed(log_dir, seed)

        if not file_pairs:
            # tqdm.write preserves the progress bar
            # tqdm.write(f"  [Warning] Skipping seed {seed}: Files not found in {log_dir}")
            continue

        # ROBUSTNESS: Detect if there are multiple runs per seed (e.g., 5 different PPO agents)
        # or just one run per seed (e.g., random agent baseline)
        #
        # Strategy:
        # - If there's only 1 file pair per seed → use original seed (e.g., random agent: 501-600)
        # - If there are multiple file pairs per seed → use unique_seed (e.g., PPO: 5×20)
        #
        # This ensures backward compatibility with random agent while fixing PPO aggregation.

        multiple_runs_per_seed = len(file_pairs) > 1

        for run_idx, (actions_file, obs_file) in enumerate(file_pairs):
            # Only create unique seed if there are multiple runs for the same seed
            if multiple_runs_per_seed:
                # Multiple agents trained on the same seed (e.g., 5 PPO agents on seed 501)
                # Create unique identifier: seed 501 run 0 -> 50100, run 1 -> 50101, etc.
                unique_seed = seed * 100 + run_idx
            else:
                # Single agent per seed (e.g., random agent baseline)
                # Keep original seed: seed 501 -> 501
                unique_seed = seed

            with open(actions_file, "r") as f:
                actions = [json.loads(line) for line in f]
            with open(obs_file, "r") as f:
                observations = [json.loads(line) for line in f]

            if not observations or not actions:
                continue

            df_all = build_reward_dataframe(observations, actions, unique_seed)
            df_summary = build_reward_summary_by_archetype(observations, actions, unique_seed, strategy_name)

            all_rewards.append(df_all)
            all_summaries.append(df_summary)

    if all_summaries:
        df_all_concat = pd.concat(all_rewards, ignore_index=True)
        df_summary_concat = pd.concat(all_summaries, ignore_index=True)

        summary_out = os.path.join(out_dir, f"summary_by_archetype_{strategy_name}.parquet")
        trajectories_out = os.path.join(out_dir, f"trajectories_{strategy_name}.parquet")

        df_summary_concat.to_parquet(summary_out, index=False)
        df_all_concat.to_parquet(trajectories_out, index=False)

        # Print aggregation statistics
        n_unique_seeds = df_all_concat['seed'].nunique()

        # Count controlled agent runs (rl_agent or random)
        controlled_archetypes = ['rl_agent', 'random']
        controlled_data = df_all_concat[df_all_concat['archetype'].isin(controlled_archetypes)]

        if not controlled_data.empty:
            n_controlled_runs = controlled_data.groupby('seed')['agent_id'].nunique().sum()
            controlled_archetype = controlled_data['archetype'].iloc[0]
        else:
            n_controlled_runs = 0
            controlled_archetype = "unknown"

        print(f"\n{'='*60}")
        print(f"AGGREGATION SUMMARY")
        print(f"{'='*60}")
        print(f"  Strategy: {strategy_name}")
        print(f"  Total unique seeds: {n_unique_seeds}")
        print(f"  Total controlled agent runs ({controlled_archetype}): {n_controlled_runs}")
        print(f"  Requested seed range: {args.start_seed} to {args.start_seed + args.num_seeds - 1}")
        print(f"{'='*60}\n")
        print(f"  DONE: Saved summary to {summary_out}")
        print(f"  DONE: Saved trajectories to {trajectories_out}")
    else:
        print(f"  [Error] No data found for strategy {strategy_name}")

    print(f"\nAll strategies processed. Final results in: {out_dir}")

if __name__ == "__main__":
    main()
