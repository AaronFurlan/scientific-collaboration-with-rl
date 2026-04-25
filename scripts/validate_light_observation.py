"""
Validation script for PeerGroupEnvironment lightweight observations.

Verifies that _get_policy_observation produced identical decisions for fixed 
archetype policies compared to the full _get_observation.

Recommended fast command:
python scripts/validate_light_observation.py --num-seeds 5 --max-steps 200 --step-stride 5 --agent-sample-size 10

Stricter command:
python scripts/validate_light_observation.py --num-seeds 10 --max-steps 600 --step-stride 1 --agent-sample-size -1
"""
import os
import sys
import math
import numpy as np
import argparse
from typing import Dict, Any, List

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.env.peer_group_environment import PeerGroupEnvironment
from src.agent_policies import careerist_policy, orthodox_scientist_policy, mass_producer_policy

def compare_actions(a1: Dict[str, Any], a2: Dict[str, Any]) -> bool:
    """Compare two action dictionaries for equality."""
    if a1["choose_project"] != a2["choose_project"]:
        return False
    if a1["put_effort"] != a2["put_effort"]:
        return False
    if not np.array_equal(a1["collaborate_with"], a2["collaborate_with"]):
        return False
    return True

def validate_observations(
    num_seeds=5, 
    max_steps=100, 
    stop_on_mismatch=False,
    agent_sample_size=10,
    step_stride=5,
    env_config=None
):
    if env_config is None:
        env_config = {}

    prestige_threshold = env_config.get("prestige_threshold", 0.29)
    novelty_threshold = env_config.get("novelty_threshold", 0.4)
    effort_threshold = env_config.get("effort_threshold", 35)

    # Map archetype names to policy functions
    archetypes = {
        "careerist": lambda obs, mask: careerist_policy(obs, mask, prestige_threshold=prestige_threshold),
        "orthodox_scientist": lambda obs, mask: orthodox_scientist_policy(obs, mask, novelty_threshold=novelty_threshold),
        "mass_producer": lambda obs, mask: mass_producer_policy(obs, mask, effort_threshold=effort_threshold),
    }

    stats = {
        "total_actions": 0,
        "matching_actions": 0,
        "matching_choose_project": 0,
        "matching_put_effort": 0,
        "matching_collaborate": 0,
    }

    per_policy_stats = {name: {
        "total": 0,
        "matching": 0,
    } for name in archetypes}

    # Expected evaluation count
    if agent_sample_size == -1:
        print(f"Approx validation checks: depends on active agent count (seeds: {num_seeds}, steps: {max_steps}, stride: {step_stride})")
    else:
        num_steps_to_validate = math.ceil(max_steps / step_stride)
        approx_checks = num_seeds * num_steps_to_validate * agent_sample_size
        print(f"Approx validation checks: {approx_checks}")

    print(f"Starting validation across {num_seeds} seeds, {max_steps} steps each.")

    for seed in range(101, 101 + num_seeds):
        print(f"--- Seed {seed} ---")
        # Use a local RNG for deterministic sampling per seed
        rng = np.random.default_rng(seed)

        env = PeerGroupEnvironment(
            start_agents=env_config.get("start_agents", 100),
            max_agents=env_config.get("max_agents", 400),
            max_steps=max_steps,
            max_peer_group_size=env_config.get("max_peer_group_size", 40),
            n_groups=env_config.get("n_groups", 10),
            max_projects_per_agent=env_config.get("max_projects_per_agent", 8),
            n_projects_per_step=env_config.get("n_projects_per_step", 1),
            use_light_policy_obs=True
        )
        
        obs_dict, info = env.reset(seed=seed)
        
        # Assign archetypes to agents (simple round-robin for validation)
        agent_policies = {}
        policy_names = list(archetypes.keys())
        for i, agent_id in enumerate(env.possible_agents):
            agent_policies[agent_id] = policy_names[i % len(policy_names)]

        for step in range(max_steps):
            active_agents = env.agents
            actions = {}
            
            # Sampling check
            should_validate_step = (step % step_stride == 0)
            
            # Determine which agents to validate this step
            agents_to_check = []
            if should_validate_step:
                # Only check active agents
                truly_active = [a for a in active_agents if env.active_agents[env.agent_to_id[a]] == 1]
                if agent_sample_size == -1 or len(truly_active) <= agent_sample_size:
                    agents_to_check = truly_active
                else:
                    agents_to_check = rng.choice(truly_active, size=agent_sample_size, replace=False).tolist()

            # We must provide actions for ALL agents in env.agents to avoid KeyError in step()
            for agent_id in active_agents:
                idx = env.agent_to_id[agent_id]
                
                # Default action (do nothing) if agent is not active or we don't have a policy
                mask = env._get_action_mask(agent_id)
                actions[agent_id] = {
                    "choose_project": 0,
                    "collaborate_with": np.zeros_like(mask["collaborate_with"]),
                    "put_effort": 0
                }

                if env.active_agents[idx] == 0:
                    continue
                
                policy_name = agent_policies[agent_id]
                policy_fn = archetypes[policy_name]
                
                if agent_id in agents_to_check:
                    # Get both observations
                    full_obs = env._get_observation(agent_id)
                    light_obs = env._get_policy_observation(agent_id)
                    
                    # Get actions
                    action_full = policy_fn(full_obs, mask)
                    action_light = policy_fn(light_obs, mask)
                    
                    # Update stats
                    stats["total_actions"] += 1
                    per_policy_stats[policy_name]["total"] += 1
                    
                    match_cp = action_full["choose_project"] == action_light["choose_project"]
                    match_pe = action_full["put_effort"] == action_light["put_effort"]
                    match_cb = np.array_equal(action_full["collaborate_with"], action_light["collaborate_with"])
                    
                    if match_cp: stats["matching_choose_project"] += 1
                    if match_pe: stats["matching_put_effort"] += 1
                    if match_cb: stats["matching_collaborate"] += 1
                    
                    if match_cp and match_pe and match_cb:
                        stats["matching_actions"] += 1
                        per_policy_stats[policy_name]["matching"] += 1
                    else:
                        print(f"MISMATCH detected! Seed: {seed}, Step: {step}, Agent: {agent_id}, Policy: {policy_name}")
                        print(f"  Full action:  {action_full}")
                        print(f"  Light action: {action_light}")
                        
                        # Log key differences
                        full_keys = set(full_obs.keys())
                        light_keys = set(light_obs.keys())
                        
                        print(f"  Keys in full obs:  {sorted(list(full_keys))}")
                        print(f"  Keys in light obs: {sorted(list(light_keys))}")
                        
                        if full_keys != light_keys:
                            print(f"  Keys missing in light: {full_keys - light_keys}")
                            print(f"  Extra keys in light:   {light_keys - full_keys}")
                        
                        for key in light_keys:
                            if key in full_obs:
                                if not np.array_equal(full_obs[key], light_obs[key]) and key != "project_opportunities" and key != "running_projects":
                                    print(f"  Value difference in key '{key}':")
                                    print(f"    Full:  {full_obs[key]}")
                                    print(f"    Light: {light_obs[key]}")
                        
                        if stop_on_mismatch:
                            print("Stopping on mismatch.")
                            return
                    
                    # For simulation to continue, we use light action to stay on the "optimized" path
                    actions[agent_id] = action_light
                else:
                    # Just run the policy normally to advance env
                    obs = env._get_policy_observation(agent_id)
                    actions[agent_id] = policy_fn(obs, mask)

            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            if all(terminated.values()) or all(truncated.values()):
                break

    # Summary
    print("\n" + "="*40)
    print("VALIDATION RESULTS")
    print("="*40)
    total = stats["total_actions"]
    if total > 0:
        print(f"Overall match rate:           {stats['matching_actions']/total*100:6.2f}% ({stats['matching_actions']}/{total})")
        print(f"choose_project match rate:    {stats['matching_choose_project']/total*100:6.2f}%")
        print(f"put_effort match rate:        {stats['matching_put_effort']/total*100:6.2f}%")
        print(f"collaborate_with match rate:  {stats['matching_collaborate']/total*100:6.2f}%")
        
        print("\nPer-Policy breakdown:")
        for name, p_stats in per_policy_stats.items():
            rate = p_stats["matching"] / p_stats["total"] * 100 if p_stats["total"] > 0 else 0
            print(f"  {name:20}: {rate:6.2f}% ({p_stats['matching']}/{p_stats['total']})")
    else:
        print("No actions were recorded.")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--stop-on-mismatch", action="store_true")
    
    # Sampling options
    parser.add_argument("--agent-sample-size", type=int, default=10)
    parser.add_argument("--step-stride", type=int, default=5)

    # Env config matching training defaults
    parser.add_argument("--start-agents", type=int, default=100)
    parser.add_argument("--max-agents", type=int, default=400)
    parser.add_argument("--max-peer-group-size", type=int, default=40)
    parser.add_argument("--n-groups", type=int, default=10)
    parser.add_argument("--max-projects-per-agent", type=int, default=8)
    parser.add_argument("--n-projects-per-step", type=int, default=1)
    parser.add_argument("--prestige-threshold", type=float, default=0.29)
    parser.add_argument("--novelty-threshold", type=float, default=0.4)
    parser.add_argument("--effort-threshold", type=int, default=35)

    args = parser.parse_args()

    env_config = {
        "start_agents": args.start_agents,
        "max_agents": args.max_agents,
        "max_peer_group_size": args.max_peer_group_size,
        "n_groups": args.n_groups,
        "max_projects_per_agent": args.max_projects_per_agent,
        "n_projects_per_step": args.n_projects_per_step,
        "prestige_threshold": args.prestige_threshold,
        "novelty_threshold": args.novelty_threshold,
        "effort_threshold": args.effort_threshold,
    }

    validate_observations(
        num_seeds=args.num_seeds, 
        max_steps=args.max_steps, 
        stop_on_mismatch=args.stop_on_mismatch,
        agent_sample_size=args.agent_sample_size,
        step_stride=args.step_stride,
        env_config=env_config
    )
