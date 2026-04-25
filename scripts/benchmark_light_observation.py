"""
Benchmarking script for PeerGroupEnvironment lightweight observations.

Compares environment step time with and without the use_light_policy_obs optimization.

Recommended command:
python scripts/benchmark_light_observation.py --num-seeds 5 --steps-per-seed 300
"""
import os
import sys
import time
import numpy as np
import argparse
from typing import Dict, Any, List

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.env.peer_group_environment import PeerGroupEnvironment
from src.agent_policies import careerist_policy, orthodox_scientist_policy, mass_producer_policy

def run_experiment(use_light: bool, args: argparse.Namespace) -> Dict[str, Any]:
    """Runs the simulation with or without light observations and measures timing."""
    all_step_times = []
    per_seed_avg = []
    total_steps_executed = 0
    
    # Instrumentation
    call_counts = {"full_obs": 0, "light_obs": 0, "mask": 0, "running_projects": 0, "open_projects": 0}
    timings = {"full_obs": 0.0, "light_obs": 0.0, "mask": 0.0, "running_projects": 0.0, "open_projects": 0.0}

    def wrap(fn, key):
        def wrapped(*args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            timings[key] += time.perf_counter() - start
            call_counts[key] += 1
            return result
        return wrapped

    # Environment config
    env_config = {
        "start_agents": args.start_agents,
        "max_agents": args.max_agents,
        "max_steps": args.max_steps,
        "max_peer_group_size": args.max_peer_group_size,
        "n_groups": args.n_groups,
        "max_projects_per_agent": args.max_projects_per_agent,
        "n_projects_per_step": args.n_projects_per_step,
        "use_light_policy_obs": use_light
    }

    # Archetype policies
    archetypes = {
        "careerist": lambda obs, mask: careerist_policy(obs, mask, prestige_threshold=args.prestige_threshold),
        "orthodox_scientist": lambda obs, mask: orthodox_scientist_policy(obs, mask, novelty_threshold=args.novelty_threshold),
        "mass_producer": lambda obs, mask: mass_producer_policy(obs, mask, effort_threshold=args.effort_threshold),
    }
    policy_names = list(archetypes.keys())

    for seed in range(42, 42 + args.num_seeds):
        env = PeerGroupEnvironment(**env_config)
        
        # Monkey-patch the environment methods
        env._get_observation = wrap(env._get_observation, "full_obs")
        env._get_policy_observation = wrap(env._get_policy_observation, "light_obs")
        env._get_action_mask = wrap(env._get_action_mask, "mask")
        env._get_running_projects_obs = wrap(env._get_running_projects_obs, "running_projects")
        env._get_open_projects_obs = wrap(env._get_open_projects_obs, "open_projects")
        
        obs_dict, _ = env.reset(seed=seed)
        
        # Fixed policy assignment (round-robin)
        agent_policies = {}
        for i, agent_id in enumerate(env.possible_agents):
            agent_policies[agent_id] = policy_names[i % len(policy_names)]

        seed_step_times = []
        for step in range(args.steps_per_seed):
            active_agents = env.agents
            actions = {}
            
            # 1. Generate actions (Action generation is intentionally outside the timed block)
            for agent_id in active_agents:
                # Use public observation and mask
                if agent_id not in obs_dict:
                    # If an agent is in env.agents but not in obs_dict (e.g. just activated),
                    # we must provide an action anyway.
                    mask = env._get_action_mask(agent_id)
                    # We might need an observation too for policy
                    if use_light:
                        obs = env._get_policy_observation(agent_id)
                    else:
                        obs = env._get_observation(agent_id)
                else:
                    # Check if it's the structure {"observation": ..., "action_mask": ...}
                    if isinstance(obs_dict[agent_id], dict) and "observation" in obs_dict[agent_id]:
                        obs = obs_dict[agent_id]["observation"]
                        mask = obs_dict[agent_id]["action_mask"]
                    else:
                        # Fallback if reset/step format ever changes back to raw obs
                        obs = obs_dict[agent_id]
                        mask = env._get_action_mask(agent_id)
                
                idx = env.agent_to_id[agent_id]
                # Even if not active in this step, we provide a dummy action to avoid KeyErrors
                # in the environment's step logic if it iterates over all agents.
                if env.active_agents[idx] == 0:
                    actions[agent_id] = {
                        "choose_project": 0,
                        "collaborate_with": np.zeros(env.max_peer_group_size, dtype=np.int8),
                        "put_effort": 0
                    }
                    continue

                policy_fn = archetypes[agent_policies[agent_id]]
                actions[agent_id] = policy_fn(obs, mask)

            # Ensure ALL agents in possible_agents have an action if they might be accessed
            # The error happened with agent_100 which is > start_agents.
            # In env.step, it iterates over actions.items() but also access f"agent_{gm_idx}"
            # which depends on peer_groups.
            
            # 2. Step the environment (The benchmark measures env.step() only)
            step_start = time.perf_counter()
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            step_end = time.perf_counter()
            
            total_steps_executed += 1
            elapsed = step_end - step_start
            
            # Warmup logic
            if step >= args.warmup_steps:
                all_step_times.append(elapsed)
                seed_step_times.append(elapsed)
            
            if all(terminated.values()) or all(truncated.values()):
                break
        
        if seed_step_times:
            per_seed_avg.append(np.mean(seed_step_times))
                
    return {
        "avg": np.mean(all_step_times) if all_step_times else 0,
        "median": np.median(all_step_times) if all_step_times else 0,
        "p95": np.percentile(all_step_times, 95) if all_step_times else 0,
        "total_time": np.sum(all_step_times),
        "total_steps": total_steps_executed,
        "per_seed_avg": per_seed_avg,
        "profiling": {
            "call_counts": call_counts,
            "timings": timings
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark light observation optimization.")
    
    # Benchmarking control
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--steps-per-seed", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=0, help="Exclude first N steps per seed from timing.")
    
    # Env config
    parser.add_argument("--start-agents", type=int, default=100)
    parser.add_argument("--max-agents", type=int, default=400)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--max-peer-group-size", type=int, default=40)
    parser.add_argument("--n-groups", type=int, default=10)
    parser.add_argument("--max-projects-per-agent", type=int, default=8)
    parser.add_argument("--n-projects-per-step", type=int, default=1)
    
    # Heuristic thresholds
    parser.add_argument("--prestige-threshold", type=float, default=0.29)
    parser.add_argument("--novelty-threshold", type=float, default=0.4)
    parser.add_argument("--effort-threshold", type=int, default=35)

    args = parser.parse_args()

    print("========================================")
    print("BENCHMARKING LIGHT OBSERVATION")
    print("========================================")
    print(f"Seeds: {args.num_seeds}, Steps per seed: {args.steps_per_seed}")
    print(f"Agents: {args.start_agents} -> {args.max_agents}")
    print("Running Experiment A (Full Observation)...")
    res_a = run_experiment(use_light=False, args=args)
    
    print("Running Experiment B (Light Observation)...")
    res_b = run_experiment(use_light=True, args=args)

    if res_a["total_steps"] != res_b["total_steps"]:
        print(f"\nWARNING: Total steps executed differ! A: {res_a['total_steps']}, B: {res_b['total_steps']}")
        print("This may happen due to stochasticity in agent lifecycle even with same seeds.")

    print("\n==============================")
    print("TIMING COMPARISON")
    print("==============================")
    
    print("Full Observation (A):")
    print(f"  avg step time:    {res_a['avg']*1000:.3f} ms")
    print(f"  median step time: {res_a['median']*1000:.3f} ms")
    print(f"  p95 step time:    {res_a['p95']*1000:.3f} ms")
    print(f"  total steps:      {res_a['total_steps']}")
    
    print("\nLight Observation (B):")
    print(f"  avg step time:    {res_b['avg']*1000:.3f} ms")
    print(f"  median step time: {res_b['median']*1000:.3f} ms")
    print(f"  p95 step time:    {res_b['p95']*1000:.3f} ms")
    print(f"  total steps:      {res_b['total_steps']}")

    speedup_abs = (res_a['avg'] - res_b['avg']) * 1000
    speedup_rel = (res_a['avg'] / res_b['avg']) if res_b['avg'] > 0 else 0
    reduction = (1 - res_b['avg'] / res_a['avg']) * 100 if res_a['avg'] > 0 else 0

    print("\nSpeedup:")
    print(f"  absolute:  {speedup_abs:.3f} ms saved per step")
    print(f"  relative:  {speedup_rel:.2f}x faster")
    print(f"  reduction: {reduction:.2f}% reduction in average step time")

    print("\n==============================")
    print("FUNCTION PROFILING")
    print("==============================")
    
    for name, res in [("Full Observation Mode (A)", res_a), ("Light Observation Mode (B)", res_b)]:
        print(f"\n{name}:")
        prof = res["profiling"]
        counts = prof["call_counts"]
        times = prof["timings"]
        total_step_time = res["total_time"]
        
        for key in ["full_obs", "light_obs", "mask", "running_projects", "open_projects"]:
            c = counts[key]
            t = times[key]
            avg = (t / c * 1000) if c > 0 else 0
            pct = (t / total_step_time * 100) if total_step_time > 0 else 0
            print(f"  _{key}:")
            print(f"    calls:      {c}")
            print(f"    total time: {t*1000:.3f} ms")
            print(f"    avg time:   {avg:.6f} ms")
            print(f"    % of step:  {pct:.2f}%")
        
        obs_building_total = times["full_obs"] + times["light_obs"] + times["mask"]
        obs_building_pct = (obs_building_total / total_step_time * 100) if total_step_time > 0 else 0
        print(f"  ----------------------------")
        print(f"  Total observation building: {obs_building_total*1000:.3f} ms ({obs_building_pct:.2f}% of step)")

    print("\nKey observations:")
    
    # 1. Is _get_observation still called in light mode?
    prof_a = res_a["profiling"]["call_counts"]
    prof_b = res_b["profiling"]["call_counts"]
    
    if prof_b["full_obs"] > 0:
        print(f"  - _get_observation IS still called in light mode ({prof_b['full_obs']} times).")
        print("    Check if the environment's step() logic or another component is forcing full observations.")
    else:
        print("  - _get_observation is NOT called in light mode (Success).")
        
    # 2. Is _get_policy_observation actually cheaper?
    timings_a = res_a["profiling"]["timings"]
    timings_b = res_b["profiling"]["timings"]
    
    avg_full = (timings_a["full_obs"] / prof_a["full_obs"] * 1000) if prof_a["full_obs"] > 0 else 0
    avg_light = (timings_b["light_obs"] / prof_b["light_obs"] * 1000) if prof_b["light_obs"] > 0 else 0
    
    if avg_light > 0 and avg_full > 0:
        reduction_per_call = (1 - avg_light / avg_full) * 100
        print(f"  - _get_policy_observation is {reduction_per_call:.2f}% cheaper than _get_observation per call.")
        print(f"    (Full: {avg_full:.4f} ms vs Light: {avg_light:.4f} ms)")
    
    # 3. Where is the remaining time spent?
    total_time_b = res_b["total_time"]
    obs_time_b = timings_b["full_obs"] + timings_b["light_obs"] + timings_b["mask"]
    other_time_pct = (1 - obs_time_b / total_time_b) * 100 if total_time_b > 0 else 0
    
    avg_running = (timings_b["running_projects"] / prof_b["running_projects"] * 1000) if prof_b["running_projects"] > 0 else 0
    pct_running = (timings_b["running_projects"] / total_time_b * 100) if total_time_b > 0 else 0
    
    print(f"  - _get_running_projects_obs takes {avg_running:.4f} ms per call ({pct_running:.2f}% of step).")
    print("    This function is still called for all agents and seems to be a major shared bottleneck.")

    if other_time_pct > 10:
        print(f"  - {other_time_pct:.2f}% of time in light mode is spent OUTSIDE observation building.")
        print("    This includes collaboration logic, project lifecycle, and rewards.")
    elif other_time_pct < 0:
         print(f"  - Note: Measured observation time exceeds total step time ({other_time_pct:.2f}%).")
         print("    This is a common artifact of high-resolution timing on many small calls (overhead).")
    else:
        print(f"  - Only {other_time_pct:.2f}% of time is spent outside observation building.")

    print("\nNOTE: This benchmark measures the full env.step() time.")
    print("Earlier profiling showed that Observation Building takes ~80% of total step time.")
    print("The theoretical maximum reduction for this optimization is also ~80%.")

    print("\nPer-Seed Average Step Time (ms):")
    print("Full (A):  ", [f"{v*1000:.2f}" for v in res_a['per_seed_avg']])
    print("Light (B): ", [f"{v*1000:.2f}" for v in res_b['per_seed_avg']])
    print("==============================\n")
    
    if reduction > 0:
        print(f"SUCCESS: Light observation reduces average step time by {reduction:.2f}%")
    else:
        print("NOTICE: No speedup detected. This might happen if the environment scale is too small or other components dominate.")

if __name__ == "__main__":
    main()
