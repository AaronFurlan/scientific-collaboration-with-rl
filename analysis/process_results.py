import json
import pandas as pd
import argparse
import os

def build_reward_dataframe(reward_steps, agents, seed):
    """
    Builds a DataFrame of accumulated rewards per agent per step,
    annotated with archetype.
    """
    agent_archetype = {}
    for a in agents:
        for k, v in a.items():
            if v is not None:
                agent_archetype[k] = v["archetype"]

    records = []
    for step_idx, step in enumerate(reward_steps):
        for agent_id, data in step.items():
            if data is not None:
                data = data.get("observation", None)
                if data and "accumulated_rewards" in data:
                    archetype = agent_archetype.get(agent_id, None)
                    if archetype is not None:
                        records.append({
                            "step": step_idx,
                            "archetype": archetype,
                            "agent_id": agent_id,
                            "accumulated_rewards": data["accumulated_rewards"][0],
                            "h_index": data["peer_h_index"][0],
                            "age": data["age"][0],
                            # "accumulated_citations": len(data.get("citations", [])),
                            # "societal_value": data['societal_value_score'],
                            "seed": seed,
                        })

    return pd.DataFrame(records)



def build_reward_summary_by_archetype(reward_steps, agents, seed, strategy):
    """
    Returns a DataFrame with rows:
      step, archetype, mean_reward, std_reward, n_agents, seed, strategy

    Raises:
        ValueError if any agent has no archetype mapping.
    """
    # === 1. Build agent_id -> archetype map ===
    agent_archetype = {}
    for a in agents:
        if not isinstance(a, dict):
            continue
        for agent_id, v in a.items():
            if v is None:
                continue
            if isinstance(v, dict) and "archetype" in v:
                agent_archetype[agent_id] = v["archetype"]

    if not agent_archetype:
        raise ValueError("No archetypes found in agents data.")

    # === 2. Build per-step stats grouped by archetype ===
    records = []
    for step_idx, step in enumerate(reward_steps):
        if not isinstance(step, dict):
            continue

        # Gather rewards per archetype for this step
        arch_rewards = {}
        for agent_id, data in step.items():
            if data is None:
                continue

            # Every agent MUST have an archetype
            if agent_id not in agent_archetype:
                raise ValueError(f"Missing archetype for agent_id '{agent_id}' at step {step_idx}")

            obs = data.get("observation", {}) if isinstance(data, dict) else {}
            if "accumulated_rewards" not in obs:
                continue

            reward = obs["accumulated_rewards"][0]
            archetype = agent_archetype[agent_id]
            arch_rewards.setdefault(archetype, []).append(reward)

        # Compute mean/std per archetype
        for archetype, rewards in arch_rewards.items():
            s = pd.Series(rewards)
            records.append({
                "step": step_idx,
                "archetype": archetype,
                "mean_reward": float(s.mean()),
                "std_reward": float(s.std(ddof=1)),  # sample std
                "n_agents": len(rewards),
                "seed": int(seed),
                "strategy": strategy,
            })

    return pd.DataFrame(records)

def find_result_files(log_base_dir, prefix, strategy, seed):
    """
    Find action and observation files in various directory structures.

    Tries multiple patterns:
    1. Flat structure: log_base_dir/{prefix}_{strategy}_seed{seed}_*.jsonl
    2. Subdirectory: log_base_dir/{prefix}_{strategy}_seed{seed}/{prefix}_{strategy}_seed{seed}_*.jsonl
    3. Timestamped subdirectory: log_base_dir/{prefix}_{strategy}_s{seed}_*/{prefix}_{strategy}_s{seed}_*.jsonl

    Returns:
        tuple: (actions_file, obs_file) or (None, None) if not found
    """
    import glob

    patterns = [
        # Pattern 1: Flat structure with "seed{N}"
        (os.path.join(log_base_dir, f"{prefix}_{strategy}_seed{seed}_actions.jsonl"),
         os.path.join(log_base_dir, f"{prefix}_{strategy}_seed{seed}_observations.jsonl")),

        # Pattern 2: Subdirectory with "seed{N}"
        (os.path.join(log_base_dir, f"{prefix}_{strategy}_seed{seed}", f"{prefix}_{strategy}_seed{seed}_actions.jsonl"),
         os.path.join(log_base_dir, f"{prefix}_{strategy}_seed{seed}", f"{prefix}_{strategy}_seed{seed}_observations.jsonl")),

        # Pattern 3: Timestamped subdirectory with "s{N}"
        (glob.glob(os.path.join(log_base_dir, f"{prefix}_{strategy}_s{seed}_*", f"{prefix}_{strategy}_s{seed}_actions.jsonl")),
         glob.glob(os.path.join(log_base_dir, f"{prefix}_{strategy}_s{seed}_*", f"{prefix}_{strategy}_s{seed}_observations.jsonl"))),
    ]

    for actions_pattern, obs_pattern in patterns:
        # Handle glob results
        actions_file = actions_pattern[0] if isinstance(actions_pattern, list) and actions_pattern else actions_pattern
        obs_file = obs_pattern[0] if isinstance(obs_pattern, list) and obs_pattern else obs_pattern

        if isinstance(actions_file, str) and isinstance(obs_file, str):
            if os.path.exists(actions_file) and os.path.exists(obs_file):
                return actions_file, obs_file

    return None, None

def main(out_base_dir=".", log_base_dir="../log", prefix="balanced", seed_start=0, seed_end=10):
    """
    Process simulation results and save summary parquet files.

    Args:
        out_base_dir: Output directory for parquet files (default: current dir)
        log_base_dir: Base directory containing log files (default: ../log)
        prefix: Prefix for file names (default: "balanced")
        seed_start: Starting seed number (default: 0)
        seed_end: Ending seed number (exclusive, default: 10)
    """
    os.makedirs(out_base_dir, exist_ok=True)

    dfs = {}
    dfs_all = {}
    for name in ["multiply", "evenly", "by_effort"]:
        all_summaries = []
        all_rewards = []
        for seed in range(seed_start, seed_end):
            actions_file, obs_file = find_result_files(log_base_dir, prefix, name, seed)

            if actions_file is None or obs_file is None:
                print(f"WARNING: Files for {prefix}_{name}_seed{seed} not found, skipping")
                continue

            print(f"Processing {prefix}_{name}_seed{seed}...")
            print(f"  Actions: {actions_file}")
            print(f"  Observations: {obs_file}")

            with open(actions_file, "r") as f:
                balanced_actions = [json.loads(line) for line in f]
            with open(obs_file, "r") as f:
                balanced_observations = [json.loads(line) for line in f]

            df_all = build_reward_dataframe(balanced_observations, balanced_actions, seed)
            df_summary = build_reward_summary_by_archetype(
                balanced_observations, balanced_actions, seed, name
            )
            all_rewards.append(df_all)
            all_summaries.append(df_summary)

        if not all_summaries:
            print(f"WARNING: No data found for strategy '{name}', skipping.")
            continue

        df_all = pd.concat(all_rewards, ignore_index=True)
        df_summary_all = pd.concat(all_summaries, ignore_index=True)
        dfs[name] = df_summary_all

        summary_path = os.path.join(out_base_dir, f"reward_summary_by_archetype_{name}.parquet")
        df_summary_all.to_parquet(summary_path, index=False)
        print(f"Saved {name} summary to {summary_path} "
            f"({len(df_summary_all)} records across {len(all_summaries)} seeds).")

        dfs_all[name] = df_all
        trajectories_path = os.path.join(out_base_dir, f"reward_trajectories_{name}.parquet")
        df_all.to_parquet(trajectories_path, index=False)
        print(f"Saved {name} simulation to {trajectories_path} "
                f"({len(df_all)} records).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process simulation results and generate parquet files")
    parser.add_argument("--out-base-dir", type=str, default=".",
                        help="Output directory for parquet files (default: current directory)")
    parser.add_argument("--log-base-dir", type=str, default="../log",
                        help="Base directory containing log files (default: ../log)")
    parser.add_argument("--prefix", type=str, default="balanced",
                        help="Prefix for file names (default: balanced). Use 'random' for random agent results.")
    parser.add_argument("--seed-start", type=int, default=0,
                        help="Starting seed number (default: 0)")
    parser.add_argument("--seed-end", type=int, default=10,
                        help="Ending seed number (exclusive, default: 10)")

    args = parser.parse_args()
    main(
        out_base_dir=args.out_base_dir,
        log_base_dir=args.log_base_dir,
        prefix=args.prefix,
        seed_start=args.seed_start,
        seed_end=args.seed_end
    )