import json
import pandas as pd
import os
import argparse

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
    parser.add_argument("log_dir", type=str, help="Path to the directory containing the simulation log files (.jsonl)")
    parser.add_argument("--out-dir", type=str, default="results", help="Directory where the result Parquet files will be saved (default: 'results')")
    
    args = parser.parse_args()

    # RL Evaluation seeds as per your proposal: 101 to 110
    seeds = range(101, 111)
    
    # Prefix from run_policy_simulation_with_rlagent.py (default or your convention)
    # Hint: You should name your logs consistently, e.g. "rl_ppo_multiply_s101_actions.jsonl"
    strategies = ["multiply", "evenly", "by_effort"]
    
    log_dir = args.log_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    for name in strategies:
        all_summaries = []
        all_rewards = []
        
        print(f"\nProcessing strategy: {name}")
        
        for seed in seeds:
            # IMPORTANT: Matches the file name pattern you should use in your eval loop
            actions_file = os.path.join(log_dir, f"rl_ppo_{name}_s{seed}_actions.jsonl")
            obs_file = os.path.join(log_dir, f"rl_ppo_{name}_s{seed}_observations.jsonl")
            
            if not os.path.exists(actions_file) or not os.path.exists(obs_file):
                print(f"  [Warning] Skipping seed {seed}: Files not found ({actions_file})")
                continue
            
            print(f"  Reading seed {seed}...")
            with open(actions_file, "r") as f:
                actions = [json.loads(line) for line in f]
            with open(obs_file, "r") as f:
                observations = [json.loads(line) for line in f]

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

if __name__ == "__main__":
    main()
