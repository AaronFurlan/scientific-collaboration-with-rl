import wandb
import os
import sys
import ray
import pandas as pd
import itertools

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.train_rl_agent import main as train_main

def profile_parallel_settings():
    # 1. Define the grid of configurations
    num_workers_list = [8, 10]
    num_envs_per_worker_list = [1, 2]
    rollout_fragment_length_list = [100, 200, 400]
    
    iterations = 5 # Reduziert von 15, da 1 Iteration ~1.8 min dauert.
    experiment_name = "parallelization-profiling-v1"
    
    # Base parameters
    base_params = {
        "algo": "PPO",
        "iterations": iterations,
        "framework": "torch",
        "policy_config_name": "Balanced",
        "group_policy_homogenous": False,
        "seed": 42,
        "n_agents": 400,
        "start_agents": 100,
        "max_steps": 600,
        "max_rewardless_steps": 50,
        "n_groups": 10,
        "max_peer_group_size": 100,
        "n_projects_per_step": 1,
        "max_projects_per_agent": 8,
        "max_agent_age": 750,
        "acceptance_threshold": 0.44,
        "reward_function": "by_effort",
        "prestige_threshold": 0.4,
        "novelty_threshold": 0.4,
        "effort_threshold": 38,
        "controlled_agent_id": "agent_0",
        "wandb_mode": "online",
        "wandb_project": "game-of-science-profiling",
        "wandb_group": experiment_name,
        "train_batch_size": 4000, 
        "evaluation_interval": 0, # Minimal overhead
        "save_every_n_iters": 0,    # Minimal overhead
    }

    results = []

    # Cartesian product of configurations
    configs = list(itertools.product(num_workers_list, num_envs_per_worker_list, rollout_fragment_length_list))
    
    print(f"Starting profiling with {len(configs)} configurations...")

    for nw, nepw, rfl in configs:
        params = base_params.copy()
        params["num_workers"] = nw
        params["num_envs_per_worker"] = nepw
        params["rollout_fragment_length"] = rfl
        
        print(f"\n>>> PROFILING CONFIG: num_workers={nw}, num_envs_per_worker={nepw}, rollout_fragment_length={rfl}")
        
        try:
            # We need to capture metrics somehow. 
            # Since train_main doesn't return metrics, and it finishes by wandb.finish()
            # we might need to rely on wandb's API or modify train_main to return something.
            # However, for now, let's assume we want to print a table at the end.
            # I will modify train_rl_agent.main to return the history or the last metrics.
            
            history = train_main(**params)
            
            if history:
                # Extract requested metrics from the last few iterations to get stable values
                last_iters = history[-3:] # Benutze die letzten 3 Iterationen für den Durchschnitt
                avg_env_steps_per_sec = sum(h.get("env_steps_per_sec", 0) for h in last_iters if h.get("env_steps_per_sec")) / len(last_iters)
                avg_sample_time_ms = sum(h.get("sample_time_ms", 0) for h in last_iters if h.get("sample_time_ms")) / len(last_iters)
                avg_train_time_ms = sum(h.get("train_time_ms", 0) for h in last_iters if h.get("train_time_ms")) / len(last_iters)
                avg_return = sum(h.get("episode_reward_mean", 0) for h in last_iters if h.get("episode_reward_mean") is not None) / len(last_iters)
                
                res = {
                    "num_workers": nw,
                    "num_envs_per_worker": nepw,
                    "rollout_fragment_length": rfl,
                    "env_steps_per_sec": avg_env_steps_per_sec,
                    "sample_time_ms": avg_sample_time_ms,
                    "train_time_ms": avg_train_time_ms,
                    "episode_return_mean": avg_return
                }
                results.append(res)
                print(f"Result: {res}")
            
        except Exception as e:
            print(f"Error profiling config {nw}/{nepw}/{rfl}: {e}")
            if ray.is_initialized():
                ray.shutdown()
        finally:
            if ray.is_initialized():
                ray.shutdown()

    # 5. Output summary table
    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*80)
        print("PROFILING SUMMARY")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        # Log summary table to WandB (optional, in a separate run or just print)
        df.to_csv("profiling_summary.csv", index=False)
    else:
        print("No results collected.")

if __name__ == "__main__":
    profile_parallel_settings()
