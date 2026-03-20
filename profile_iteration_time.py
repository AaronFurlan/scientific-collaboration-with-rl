import time
import argparse
import os
import pandas as pd
from train_rl_agent import main as train_main
import ray

def profile_iteration_time():
    parser = argparse.ArgumentParser(description="Profile iteration time for different RL parameters.")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations per config")
    args = parser.parse_args()

    # Parameter, die wir analysieren wollen
    # n_agents, max_peer_group_size, train-batch-size, max_steps, n_groups, num_workers, evaluation_interval

    # Baseline mit reduzierten Werten für schnellere Durchläufe
    default_params = {
        "algo": "PPO",
        "iterations": args.iterations,
        "framework": "torch",
        "policy_config_name": "Balanced",
        "group_policy_homogenous": False,
        "seed": 42,
        "n_agents": 100,        # Reduziert von 2000
        "start_agents": 50,     # Reduziert von 200
        "max_steps": 100,       # Reduziert von 600
        "max_rewardless_steps": 100,
        "n_groups": 20,         # Reduziert von 50
        "max_peer_group_size": 10, # Reduziert von 40
        "n_projects_per_step": 1,
        "max_projects_per_agent": 6,
        "max_agent_age": 750,
        "acceptance_threshold": 0.44,
        "reward_function": "by_effort",
        "prestige_threshold": 0.2,
        "novelty_threshold": 0.8,
        "effort_threshold": 22,
        "controlled_agent_id": "agent_0",
        "wandb_mode": "disabled",
        "train_batch_size": 4000, # Reduziert von 4000
        "save_every_n_iters": 0,
        "num_workers": 2,       # Reduziert von 8
        "evaluation_interval": 0, # Evaluation standardmäßig aus für Profiling
    }

    test_scenarios = [
        {"name": "Baseline", "params": {}},
        {"name": "n_agents", "params": {"n_agents": 110}},
        {"name": "max_peer_group_size", "params": {"max_peer_group_size": 20}},
        {"name": "train_batch_size", "params": {"train_batch_size": 4010}},
        {"name": "max_steps", "params": {"max_steps": 210}},
        {"name": "n_groups", "params": {"n_groups": 30}},
        {"name": "num_workers", "params": {"num_workers": 5}},
        {"name": "evaluation_interval", "params": {"evaluation_interval": 2}},
    ]

    results = []

    for scenario in test_scenarios:
        print(f"\n>>> Running Scenario: {scenario['name']}")
        params = default_params.copy()
        params.update(scenario["params"])
        
        start_time = time.time()
        try:
            # Note: main() calls ray.init() and ray.shutdown()
            train_main(**params)
        except Exception as e:
            print(f"Error in scenario {scenario['name']}: {e}")
            continue
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / args.iterations
        
        print(f"<<< Finished {scenario['name']}: Total={total_time:.2f}s, Avg={avg_time:.2f}s/iter")
        
        results.append({
            "Scenario": scenario["name"],
            "Total Time": total_time,
            "Avg Time/Iter": avg_time,
            **scenario["params"]
        })

    df = pd.DataFrame(results)
    print("\n=== PROFILING RESULTS ===")
    print(df.to_string(index=False))
    df.to_csv("profiling_results.csv", index=False)
    print("\nResults saved to profiling_results.csv")

if __name__ == "__main__":
    profile_iteration_time()
