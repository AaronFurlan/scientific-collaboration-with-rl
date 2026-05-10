import json
import pandas as pd
import os
import argparse
import re
from datetime import datetime
from tqdm import tqdm

def find_latest_log_files(base_dir, strategy, seed):
    """
    Finds the latest actions and observations files for a given strategy and seed.
    Looks for subdirectories like rl_ppo_{strategy}_s{seed}_{timestamp}_s{seed}/
    """
    pattern = re.compile(f"rl_ppo_{strategy}_s{seed}_(\\d{{8}}_\\d{{6}})_s{seed}")
    
    candidates = []
    if not os.path.exists(base_dir):
        return None, None

    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path):
            match = pattern.match(entry)
            if match:
                timestamp_str = match.group(1)
                try:
                    ts = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    candidates.append((ts, full_path))
                except ValueError:
                    continue
    
    if not candidates:
        return None, None
    
    # Sort by timestamp descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    latest_dir = candidates[0][1]
    
    actions_file = os.path.join(latest_dir, f"rl_ppo_{strategy}_s{seed}_actions.jsonl")
    obs_file = os.path.join(latest_dir, f"rl_ppo_{strategy}_s{seed}_observations.jsonl")
    
    if os.path.exists(actions_file) and os.path.exists(obs_file):
        return actions_file, obs_file
    
    return None, None

def build_reward_dataframe(reward_steps, agents, seed):
    """
    Builds a DataFrame of accumulated rewards per agent per step,
    annotated with archetype (or 'RL_Agent' if it's the controlled agent).
    """
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
    """
    Returns a DataFrame with rows:
      step, archetype, mean_reward, std_reward, n_agents, seed, strategy
    """
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

    records = []
    for step_idx, step in enumerate(reward_steps):
        if not isinstance(step, dict):
            continue

        arch_rewards = {}
        for agent_id, data in step.items():
            if data is None:
                continue

            if agent_id not in agent_archetype:
                # Fallback
                agent_archetype[agent_id] = "Unknown"

            obs = data.get("observation", {}) if isinstance(data, dict) else {}
            if "accumulated_rewards" not in obs:
                continue

            reward = obs["accumulated_rewards"][0]
            archetype = agent_archetype[agent_id]
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
    
    args = parser.parse_args()

    seeds = range(args.start_seed, args.start_seed + args.num_seeds)
    strategies = ["by_effort"]
    
    log_dir = args.log_dir
    
    # Create timestamped subfolder in results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_base_dir, f"aggregation_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Aggregation started. Results will be saved to: {out_dir}")

    for name in strategies:
        all_summaries = []
        all_rewards = []
        
        print(f"\nProcessing strategy: {name}")
        
        for seed in tqdm(seeds, desc=f"Seeds ({name})", unit="seed"):
            actions_file, obs_file = find_latest_log_files(log_dir, name, seed)
            
            if not actions_file or not obs_file:
                # tqdm.write preserves the progress bar
                # tqdm.write(f"  [Warning] Skipping seed {seed}: Files not found in {log_dir}")
                continue
            
            with open(actions_file, "r") as f:
                actions = [json.loads(line) for line in f]
            with open(obs_file, "r") as f:
                observations = [json.loads(line) for line in f]

            if not observations or not actions:
                continue

            df_all = build_reward_dataframe(observations, actions, seed)
            df_summary = build_reward_summary_by_archetype(observations, actions, seed, name)
            
            all_rewards.append(df_all)
            all_summaries.append(df_summary)
        
        if all_summaries:
            df_all_concat = pd.concat(all_rewards, ignore_index=True)
            df_summary_concat = pd.concat(all_summaries, ignore_index=True)
            
            summary_out = os.path.join(out_dir, f"rl_summary_by_archetype_{name}.parquet")
            trajectories_out = os.path.join(out_dir, f"rl_trajectories_{name}.parquet")
            
            df_summary_concat.to_parquet(summary_out, index=False)
            df_all_concat.to_parquet(trajectories_out, index=False)
            
            print(f"  DONE: Saved RL summary to {summary_out}")
            print(f"  DONE: Saved RL trajectories to {trajectories_out}")
        else:
            print(f"  [Error] No data found for strategy {name}")

    print(f"\nAll strategies processed. Final results in: {out_dir}")

if __name__ == "__main__":
    main()
