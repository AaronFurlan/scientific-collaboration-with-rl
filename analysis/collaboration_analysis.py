"""
Collaboration Action Analysis

Analyzes the collaborate_with action behavior of PPO agents.

This module provides functions to:
1. Load and process collaboration data at step-level and peer-level
2. Analyze collaboration patterns and decision factors
3. Find thresholds, correlations, and recurring situations
4. Check for slot bias and project choice dependencies
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import warnings
import re

warnings.filterwarnings('ignore')


# ============================================================================
# Helper Functions
# ============================================================================

def get_val(d, k, default=0):
    """Extract value from dict, handling list values."""
    v = d.get(k, default)
    return v[0] if isinstance(v, list) else v


def extract_seed_from_path(path_str):
    """Extract seed number from directory path."""
    match = re.search(r's(\d+)', str(path_str))
    return int(match.group(1)) if match else None


def safe_divide(numerator, denominator, default=0):
    """Safe division to avoid division by zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def get_scalar(value, default=np.nan):
    """Extract a scalar from nested observation values."""
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, list):
        if len(value) == 0:
            return default
        if isinstance(value[0], list):
            # [[3.0]] -> 3.0
            return get_scalar(value[0], default)
        else:
            # [3.0] -> 3.0
            return value[0]
    if isinstance(value, dict):
        values = list(value.values())
        if len(values) == 0:
            return default
        return get_scalar(values[0], default)
    return default


def get_centroid(obs, key, default=(np.nan, np.nan)):
    """Extract a 2D centroid from observation fields.

    Handles:
    - [[x, y]] -> (x, y)
    - [x, y] -> (x, y)
    - np.array([[x, y]]) -> (x, y)
    - np.array([x, y]) -> (x, y)
    - {"x": x, "y": y} -> (x, y)
    - {"0": x, "1": y} -> (x, y)

    Returns (np.nan, np.nan) if extraction fails.
    """
    value = obs.get(key, None)

    if value is None:
        return default

    # Convert to numpy array for uniform handling
    try:
        arr = np.asarray(value, dtype=float)
    except (ValueError, TypeError):
        return default

    # Flatten and check size
    if arr.size < 2:
        return default

    arr_flat = arr.reshape(-1)
    x, y = float(arr_flat[0]), float(arr_flat[1])

    # Return NaN if values are invalid
    if not (np.isfinite(x) and np.isfinite(y)):
        return default

    return x, y


# ============================================================================
# Data Loading Functions
# ============================================================================

def build_collaboration_step_dataframe(actions_path, observations_path, seed=None, reward_scheme=None):
    """
    Build step-level dataframe: Does agent collaborate with any peer?

    Returns one row per timestep for agent_0.
    Target: collaborates_any (0 or 1)
    """
    actions_path = Path(actions_path)
    observations_path = Path(observations_path)
    rows = []

    with open(actions_path, 'r') as f_act, open(observations_path, 'r') as f_obs:
        for line_act, line_obs in zip(f_act, f_obs):
            try:
                action_line = json.loads(line_act)
                obs_line = json.loads(line_obs)

                agent_id = 'agent_0'
                if agent_id not in action_line or agent_id not in obs_line:
                    continue

                agent_action = action_line[agent_id]
                agent_obs_data = obs_line[agent_id]

                # Handle both observation formats
                if 'observation' in agent_obs_data:
                    agent_obs = agent_obs_data['observation']
                    action_mask = agent_obs_data.get('action_mask', {})
                else:
                    agent_obs = agent_obs_data
                    action_mask = agent_obs_data.get('action_mask', {})

                # Extract collaborate_with action
                if isinstance(agent_action, dict):
                    collab_action = agent_action.get('collaborate_with', [])
                    choose_project_action = agent_action.get('choose_project', 0)
                else:
                    continue

                if isinstance(collab_action, (int, float)):
                    collab_action = [collab_action]
                if isinstance(choose_project_action, list):
                    choose_project_action = choose_project_action[0]

                # Extract peer group and masks
                peer_group = agent_obs.get('peer_group', [])
                if isinstance(peer_group, dict):
                    peer_group = list(peer_group.values())

                collab_mask = action_mask.get('collaborate_with', [])
                if isinstance(collab_mask, dict):
                    collab_mask = list(collab_mask.values())

                # Ensure same length
                max_peers = max(len(peer_group), len(collab_action), len(collab_mask))
                peer_group = list(peer_group) + [0] * (max_peers - len(peer_group))
                collab_action = list(collab_action) + [0] * (max_peers - len(collab_action))
                collab_mask = list(collab_mask) + [0] * (max_peers - len(collab_mask))

                # Count peers
                n_available_peers = sum(1 for pg in peer_group if pg > 0)
                n_valid_collab_actions = sum(1 for cm in collab_mask if cm > 0)
                n_selected_peers = sum(
                    1 for i, ca in enumerate(collab_action)
                    if peer_group[i] > 0 and collab_mask[i] > 0 and ca > 0.5
                )

                collaborates_any = 1 if n_selected_peers > 0 else 0
                collaboration_rate = safe_divide(n_selected_peers, n_available_peers, 0)

                # Extract peer features
                peer_reputation = agent_obs.get('peer_reputation', [])
                peer_centroids = agent_obs.get('peer_centroids', [])

                if isinstance(peer_reputation, dict):
                    peer_reputation = list(peer_reputation.values())
                if isinstance(peer_centroids, dict):
                    peer_centroids = list(peer_centroids.values())

                peer_reputation = list(peer_reputation) + [0] * (max_peers - len(peer_reputation))
                peer_centroids = list(peer_centroids) + [[0, 0]] * (max_peers - len(peer_centroids))

                # Extract self centroid using helper function
                # Try 'self_centroid' first, then 'self_centroids'
                self_x, self_y = get_centroid(agent_obs, 'self_centroid', (np.nan, np.nan))
                if np.isnan(self_x) or np.isnan(self_y):
                    self_x, self_y = get_centroid(agent_obs, 'self_centroids', (np.nan, np.nan))

                # Compute aggregated peer features
                selected_peer_reputations = []
                unselected_peer_reputations = []
                selected_distances = []
                unselected_distances = []
                all_peer_reputations = []
                all_peer_distances = []

                for i in range(max_peers):
                    if peer_group[i] <= 0 or collab_mask[i] <= 0:
                        continue

                    rep = peer_reputation[i]
                    centroid = peer_centroids[i] if i < len(peer_centroids) else [0, 0]
                    if isinstance(centroid, (int, float)):
                        centroid = [centroid, 0]
                    elif len(centroid) < 2:
                        centroid = list(centroid) + [0] * (2 - len(centroid))

                    peer_x, peer_y = centroid[0], centroid[1]

                    # Calculate distance only if all coordinates are valid
                    if np.isfinite(self_x) and np.isfinite(self_y) and np.isfinite(peer_x) and np.isfinite(peer_y):
                        distance = np.sqrt((peer_x - self_x)**2 + (peer_y - self_y)**2)
                    else:
                        distance = np.nan

                    all_peer_reputations.append(rep)
                    all_peer_distances.append(distance)

                    if collab_action[i] > 0.5:
                        selected_peer_reputations.append(rep)
                        selected_distances.append(distance)
                    else:
                        unselected_peer_reputations.append(rep)
                        unselected_distances.append(distance)

                # Aggregates
                mean_selected_peer_reputation = np.mean(selected_peer_reputations) if selected_peer_reputations else np.nan
                mean_unselected_peer_reputation = np.mean(unselected_peer_reputations) if unselected_peer_reputations else np.nan
                max_selected_peer_reputation = np.max(selected_peer_reputations) if selected_peer_reputations else np.nan
                mean_selected_distance = np.mean(selected_distances) if selected_distances else np.nan
                mean_unselected_distance = np.mean(unselected_distances) if unselected_distances else np.nan
                min_selected_distance = np.min(selected_distances) if selected_distances else np.nan

                mean_peer_reputation = np.mean(all_peer_reputations) if all_peer_reputations else 0
                max_peer_reputation = np.max(all_peer_reputations) if all_peer_reputations else 0
                std_peer_reputation = np.std(all_peer_reputations) if all_peer_reputations else 0
                mean_peer_distance = np.mean(all_peer_distances) if all_peer_distances else 0
                min_peer_distance = np.min(all_peer_distances) if all_peer_distances else 0
                max_peer_distance = np.max(all_peer_distances) if all_peer_distances else 0

                # Workload features
                running_projects = agent_obs.get('running_projects', {})
                if isinstance(running_projects, dict):
                    running_projects = list(running_projects.values())

                n_running_projects = len(running_projects)
                progress_vals = []
                time_left_vals = []
                remaining_effort_vals = []
                effort_per_time_left_vals = []

                for rp in running_projects:
                    req_eff = get_val(rp, 'required_effort', 1)
                    cur_eff = get_val(rp, 'current_effort', 0)
                    tl = get_val(rp, 'time_left', 1)
                    progress = cur_eff / max(req_eff, 1)
                    remaining_eff = max(req_eff - cur_eff, 0)
                    eff_per_tl = safe_divide(remaining_eff, tl, 0)
                    progress_vals.append(progress)
                    time_left_vals.append(tl)
                    remaining_effort_vals.append(remaining_eff)
                    effort_per_time_left_vals.append(eff_per_tl)

                mean_running_progress = np.mean(progress_vals) if progress_vals else 0
                max_running_progress = np.max(progress_vals) if progress_vals else 0
                mean_running_time_left = np.mean(time_left_vals) if time_left_vals else 0
                min_running_time_left = np.min(time_left_vals) if time_left_vals else 0
                mean_remaining_effort = np.mean(remaining_effort_vals) if remaining_effort_vals else 0
                total_remaining_effort = np.sum(remaining_effort_vals) if remaining_effort_vals else 0
                mean_effort_per_time_left = np.mean(effort_per_time_left_vals) if effort_per_time_left_vals else 0

                # Project opportunity features
                project_opps = agent_obs.get('project_opportunities', {})
                if isinstance(project_opps, dict):
                    project_opps = list(project_opps.values())

                if len(project_opps) > 0:
                    opp = project_opps[0]
                    opp_required_effort = get_val(opp, 'required_effort', 0)
                    opp_prestige = get_val(opp, 'prestige', 0)
                    opp_novelty = get_val(opp, 'novelty', 0)
                    opp_time_window = get_val(opp, 'time_window', 1)
                    opp_effort_per_time = safe_divide(opp_required_effort, opp_time_window, 0)
                else:
                    opp_required_effort = 0
                    opp_prestige = 0
                    opp_novelty = 0
                    opp_time_window = 0
                    opp_effort_per_time = 0

                # Agent features
                age = get_val(agent_obs, 'age', 0)
                accumulated_rewards = get_val(agent_obs, 'accumulated_rewards', 0)
                accepted_project = 1 if choose_project_action > 0 else 0

                # Build row
                row = {
                    'collaborates_any': collaborates_any,
                    'n_selected_peers': n_selected_peers,
                    'collaboration_rate': collaboration_rate,
                    'n_available_peers': n_available_peers,
                    'n_valid_collab_actions': n_valid_collab_actions,
                    'mean_selected_peer_reputation': mean_selected_peer_reputation,
                    'mean_unselected_peer_reputation': mean_unselected_peer_reputation,
                    'max_selected_peer_reputation': max_selected_peer_reputation,
                    'mean_selected_distance': mean_selected_distance,
                    'mean_unselected_distance': mean_unselected_distance,
                    'min_selected_distance': min_selected_distance,
                    'mean_peer_reputation': mean_peer_reputation,
                    'max_peer_reputation': max_peer_reputation,
                    'std_peer_reputation': std_peer_reputation,
                    'mean_peer_distance': mean_peer_distance,
                    'min_peer_distance': min_peer_distance,
                    'max_peer_distance': max_peer_distance,
                    'n_running_projects': n_running_projects,
                    'mean_running_progress': mean_running_progress,
                    'max_running_progress': max_running_progress,
                    'mean_running_time_left': mean_running_time_left,
                    'min_running_time_left': min_running_time_left,
                    'mean_remaining_effort': mean_remaining_effort,
                    'total_remaining_effort': total_remaining_effort,
                    'mean_effort_per_time_left': mean_effort_per_time_left,
                    'opportunity_required_effort': opp_required_effort,
                    'opportunity_prestige': opp_prestige,
                    'opportunity_novelty': opp_novelty,
                    'opportunity_time_window': opp_time_window,
                    'opportunity_effort_per_time': opp_effort_per_time,
                    'age': age,
                    'accumulated_rewards': accumulated_rewards,
                    'choose_project_action': choose_project_action,
                    'accepted_project': accepted_project,
                }

                if seed is not None:
                    row['seed'] = seed
                if reward_scheme is not None:
                    row['reward_scheme'] = reward_scheme

                rows.append(row)

            except (json.JSONDecodeError, KeyError, ValueError, IndexError, TypeError):
                continue

    return pd.DataFrame(rows)


def build_collaboration_peer_dataframe(actions_path, observations_path, seed=None, reward_scheme=None):
    """
    Build peer-level dataframe: Which peers are selected?

    Returns one row per valid peer-slot per timestep.
    Target: selected_peer (0 or 1)
    """
    actions_path = Path(actions_path)
    observations_path = Path(observations_path)
    rows = []

    with open(actions_path, 'r') as f_act, open(observations_path, 'r') as f_obs:
        for line_act, line_obs in zip(f_act, f_obs):
            try:
                action_line = json.loads(line_act)
                obs_line = json.loads(line_obs)

                agent_id = 'agent_0'
                if agent_id not in action_line or agent_id not in obs_line:
                    continue

                agent_action = action_line[agent_id]
                agent_obs_data = obs_line[agent_id]

                if 'observation' in agent_obs_data:
                    agent_obs = agent_obs_data['observation']
                    action_mask = agent_obs_data.get('action_mask', {})
                else:
                    agent_obs = agent_obs_data
                    action_mask = agent_obs_data.get('action_mask', {})

                if isinstance(agent_action, dict):
                    collab_action = agent_action.get('collaborate_with', [])
                    choose_project_action = agent_action.get('choose_project', 0)
                else:
                    continue

                if isinstance(collab_action, (int, float)):
                    collab_action = [collab_action]
                if isinstance(choose_project_action, list):
                    choose_project_action = choose_project_action[0]

                peer_group = agent_obs.get('peer_group', [])
                if isinstance(peer_group, dict):
                    peer_group = list(peer_group.values())

                collab_mask = action_mask.get('collaborate_with', [])
                if isinstance(collab_mask, dict):
                    collab_mask = list(collab_mask.values())

                peer_reputation = agent_obs.get('peer_reputation', [])
                peer_centroids = agent_obs.get('peer_centroids', [])

                if isinstance(peer_reputation, dict):
                    peer_reputation = list(peer_reputation.values())
                if isinstance(peer_centroids, dict):
                    peer_centroids = list(peer_centroids.values())

                # Extract self centroid using helper function
                # Try 'self_centroid' first, then 'self_centroids'
                self_x, self_y = get_centroid(agent_obs, 'self_centroid', (np.nan, np.nan))
                if np.isnan(self_x) or np.isnan(self_y):
                    self_x, self_y = get_centroid(agent_obs, 'self_centroids', (np.nan, np.nan))

                max_peers = max(len(peer_group), len(collab_action), len(collab_mask), len(peer_reputation), len(peer_centroids))
                peer_group = list(peer_group) + [0] * (max_peers - len(peer_group))
                collab_action = list(collab_action) + [0] * (max_peers - len(collab_action))
                collab_mask = list(collab_mask) + [0] * (max_peers - len(collab_mask))
                peer_reputation = list(peer_reputation) + [0] * (max_peers - len(peer_reputation))
                peer_centroids = list(peer_centroids) + [[0, 0]] * (max_peers - len(peer_centroids))

                # Workload and opportunity features
                running_projects = agent_obs.get('running_projects', {})
                if isinstance(running_projects, dict):
                    running_projects = list(running_projects.values())

                n_running_projects = len(running_projects)
                progress_vals = []
                remaining_effort_vals = []

                for rp in running_projects:
                    req_eff = get_val(rp, 'required_effort', 1)
                    cur_eff = get_val(rp, 'current_effort', 0)
                    progress = cur_eff / max(req_eff, 1)
                    remaining_eff = max(req_eff - cur_eff, 0)
                    progress_vals.append(progress)
                    remaining_effort_vals.append(remaining_eff)

                mean_running_progress = np.mean(progress_vals) if progress_vals else 0
                max_running_progress = np.max(progress_vals) if progress_vals else 0
                total_remaining_effort = np.sum(remaining_effort_vals) if remaining_effort_vals else 0

                project_opps = agent_obs.get('project_opportunities', {})
                if isinstance(project_opps, dict):
                    project_opps = list(project_opps.values())

                if len(project_opps) > 0:
                    opp = project_opps[0]
                    opp_prestige = get_val(opp, 'prestige', 0)
                    opp_novelty = get_val(opp, 'novelty', 0)
                else:
                    opp_prestige = 0
                    opp_novelty = 0

                age = get_val(agent_obs, 'age', 0)
                accumulated_rewards = get_val(agent_obs, 'accumulated_rewards', 0)
                accepted_project = 1 if choose_project_action > 0 else 0

                # Compute ranks
                valid_reputations = [
                    (i, peer_reputation[i])
                    for i in range(max_peers)
                    if peer_group[i] > 0 and collab_mask[i] > 0
                ]
                sorted_by_rep = sorted(valid_reputations, key=lambda x: x[1], reverse=True)
                reputation_rank_map = {idx: rank for rank, (idx, _) in enumerate(sorted_by_rep)}

                valid_distances = []
                for i in range(max_peers):
                    if peer_group[i] <= 0 or collab_mask[i] <= 0:
                        continue

                    centroid = peer_centroids[i] if i < len(peer_centroids) else [0, 0]
                    if isinstance(centroid, (int, float)):
                        centroid = [centroid, 0]
                    elif len(centroid) < 2:
                        centroid = list(centroid) + [0] * (2 - len(centroid))

                    peer_x, peer_y = centroid[0], centroid[1]

                    # Calculate distance only if all coordinates are valid
                    if np.isfinite(self_x) and np.isfinite(self_y) and np.isfinite(peer_x) and np.isfinite(peer_y):
                        distance = np.sqrt((peer_x - self_x)**2 + (peer_y - self_y)**2)
                    else:
                        distance = np.nan
                    valid_distances.append((i, distance))

                sorted_by_dist = sorted(valid_distances, key=lambda x: x[1])
                distance_rank_map = {idx: rank for rank, (idx, _) in enumerate(sorted_by_dist)}

                # Create row per valid peer
                for i in range(max_peers):
                    peer_available = peer_group[i] > 0
                    action_valid = collab_mask[i] > 0

                    if not peer_available or not action_valid:
                        continue

                    selected_peer = 1 if collab_action[i] > 0.5 else 0
                    rep = peer_reputation[i]
                    centroid = peer_centroids[i] if i < len(peer_centroids) else [0, 0]
                    if isinstance(centroid, (int, float)):
                        centroid = [centroid, 0]
                    elif len(centroid) < 2:
                        centroid = list(centroid) + [0] * (2 - len(centroid))

                    peer_x, peer_y = centroid[0], centroid[1]

                    # Calculate distance only if all coordinates are valid
                    if np.isfinite(self_x) and np.isfinite(self_y) and np.isfinite(peer_x) and np.isfinite(peer_y):
                        distance = np.sqrt((peer_x - self_x)**2 + (peer_y - self_y)**2)
                    else:
                        distance = np.nan

                    reputation_rank = reputation_rank_map.get(i, -1)
                    distance_rank = distance_rank_map.get(i, -1)
                    is_top_reputation_peer = 1 if reputation_rank == 0 else 0
                    is_closest_peer = 1 if distance_rank == 0 else 0
                    is_far_peer = 1 if distance_rank >= len(valid_distances) - 1 else 0

                    row = {
                        'selected_peer': selected_peer,
                        'peer_slot_index': i,
                        'peer_available': 1,
                        'action_valid': 1,
                        'peer_reputation': rep,
                        'peer_centroid_x': peer_x,
                        'peer_centroid_y': peer_y,
                        'self_centroid_x': self_x,
                        'self_centroid_y': self_y,
                        'distance_to_self': distance,
                        'peer_reputation_rank': reputation_rank,
                        'peer_distance_rank': distance_rank,
                        'is_top_reputation_peer': is_top_reputation_peer,
                        'is_closest_peer': is_closest_peer,
                        'is_far_peer': is_far_peer,
                        'n_running_projects': n_running_projects,
                        'mean_running_progress': mean_running_progress,
                        'max_running_progress': max_running_progress,
                        'total_remaining_effort': total_remaining_effort,
                        'opportunity_prestige': opp_prestige,
                        'opportunity_novelty': opp_novelty,
                        'age': age,
                        'accumulated_rewards': accumulated_rewards,
                        'choose_project_action': choose_project_action,
                        'accepted_project': accepted_project,
                    }

                    if seed is not None:
                        row['seed'] = seed
                    if reward_scheme is not None:
                        row['reward_scheme'] = reward_scheme

                    rows.append(row)

            except (json.JSONDecodeError, KeyError, ValueError, IndexError, TypeError):
                continue

    return pd.DataFrame(rows)


def build_collaboration_dataframes_for_seeds(
    json_dir,
    reward_scheme,
    seeds,
    algo="ppo",
    use_all_matching_dirs=True,
):
    """
    Load collaboration data for multiple seeds.

    If use_all_matching_dirs=True:
        process all matching subdirectories per seed.
    If use_all_matching_dirs=False:
        process only the newest matching subdirectory per seed.

    Returns:
        step_df, peer_df
    """
    base_path = Path(json_dir)

    if not base_path.exists():
        print(f"Error: Directory {json_dir} does not exist!")
        return pd.DataFrame(), pd.DataFrame()

    all_step_dfs = []
    all_peer_dfs = []

    all_subdirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
    print(f"Found {len(all_subdirs)} subdirectories in {json_dir}")

    processed_runs = 0
    skipped_runs = 0

    for seed in seeds:
        pattern = f"rl_{algo}_{reward_scheme}_s{seed}_"
        matching_dirs = [d for d in all_subdirs if d.name.startswith(pattern)]

        if not matching_dirs:
            print(f"Warning: No subdirectory found for seed {seed}")
            continue

        matching_dirs = sorted(matching_dirs, key=lambda x: x.name)

        if not use_all_matching_dirs:
            matching_dirs = [matching_dirs[-1]]
            print(f"Info: Using newest for seed {seed}: {matching_dirs[0].name}")
        else:
            print(f"Info: Processing {len(matching_dirs)} runs for seed {seed}")

        for run_idx, subdir in enumerate(matching_dirs):
            actions_files = sorted(subdir.glob("*_actions.jsonl"))
            obs_files = sorted(subdir.glob("*_observations.jsonl"))

            if not actions_files or not obs_files:
                print(f"Warning: Missing files in {subdir.name}")
                skipped_runs += 1
                continue

            actions_path = actions_files[0]
            obs_path = obs_files[0]

            run_id = subdir.name

            print(f"Processing seed {seed}, run {run_idx + 1}/{len(matching_dirs)}: {run_id}")

            step_df = build_collaboration_step_dataframe(
                actions_path=actions_path,
                observations_path=obs_path,
                seed=seed,
                reward_scheme=reward_scheme,
            )

            peer_df = build_collaboration_peer_dataframe(
                actions_path=actions_path,
                observations_path=obs_path,
                seed=seed,
                reward_scheme=reward_scheme,
            )

            # Add run identifier so repeated checkpoints/runs are distinguishable.
            if not step_df.empty:
                step_df["run_id"] = run_id
                step_df["run_idx_for_seed"] = run_idx
                all_step_dfs.append(step_df)
                print(f"  -> Step-level: {len(step_df)} samples")

            if not peer_df.empty:
                peer_df["run_id"] = run_id
                peer_df["run_idx_for_seed"] = run_idx
                all_peer_dfs.append(peer_df)
                print(f"  -> Peer-level: {len(peer_df)} samples")

            processed_runs += 1

    if all_step_dfs:
        combined_step_df = pd.concat(all_step_dfs, ignore_index=True)
        print(f"\nTotal processed runs: {processed_runs}")
        print(f"Skipped runs: {skipped_runs}")
        print(f"Total step-level: {len(combined_step_df)}")
        print(f"Unique seeds: {combined_step_df['seed'].nunique()}")
        print(f"Unique runs: {combined_step_df['run_id'].nunique()}")
        print(f"Collaboration rate: {combined_step_df['collaborates_any'].mean():.3f}")
    else:
        combined_step_df = pd.DataFrame()

    if all_peer_dfs:
        combined_peer_df = pd.concat(all_peer_dfs, ignore_index=True)
        print(f"\nTotal peer-level: {len(combined_peer_df)}")
        print(f"Unique seeds: {combined_peer_df['seed'].nunique()}")
        print(f"Unique runs: {combined_peer_df['run_id'].nunique()}")
        print(f"Selection rate: {combined_peer_df['selected_peer'].mean():.3f}")
    else:
        combined_peer_df = pd.DataFrame()

    return combined_step_df, combined_peer_df


# ============================================================================
# Analysis Functions
# ============================================================================

def summarize_collaboration_by_feature(df, target_col, features):
    """Compare target=1 vs target=0 by features."""
    positive = df[df[target_col] == 1]
    negative = df[df[target_col] == 0]

    summary = []
    for feat in features:
        if feat not in df.columns:
            continue

        pos_mean = positive[feat].mean() if len(positive) > 0 else np.nan
        pos_std = positive[feat].std() if len(positive) > 0 else np.nan
        neg_mean = negative[feat].mean() if len(negative) > 0 else np.nan
        neg_std = negative[feat].std() if len(negative) > 0 else np.nan
        diff = pos_mean - neg_mean

        summary.append({
            'feature': feat,
            f'{target_col}=1_mean': pos_mean,
            f'{target_col}=1_std': pos_std,
            f'{target_col}=0_mean': neg_mean,
            f'{target_col}=0_std': neg_std,
            'difference': diff,
        })

    summary_df = pd.DataFrame(summary)
    return summary_df.sort_values('difference', ascending=False, key=abs)


def compute_collaboration_correlations(df, target_col, features):
    """Compute correlations with target."""
    correlations = {}
    for feat in features:
        if feat in df.columns and feat != target_col:
            correlations[feat] = df[feat].corr(df[target_col])

    return pd.Series(correlations).sort_values(ascending=False, key=abs)


def find_simple_collaboration_thresholds(df, target_col, features, min_samples_leaf=30):
    """Find quantile-based thresholds."""
    results = []

    for feat in features:
        if feat not in df.columns or feat == target_col:
            continue

        for q in [0.25, 0.5, 0.75]:
            try:
                threshold = df[feat].quantile(q)
                below = df[df[feat] <= threshold]
                above = df[df[feat] > threshold]

                if len(below) < min_samples_leaf or len(above) < min_samples_leaf:
                    continue

                rate_below = below[target_col].mean()
                rate_above = above[target_col].mean()
                diff = rate_above - rate_below

                results.append({
                    'feature': feat,
                    'threshold': threshold,
                    'n_below': len(below),
                    'n_above': len(above),
                    f'{target_col}_rate_below': rate_below,
                    f'{target_col}_rate_above': rate_above,
                    'difference': diff,
                })
            except:
                continue

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('difference', ascending=False, key=abs)

    return results_df


def fit_collaboration_logistic_regression(df, target_col, features):
    """Fit logistic regression."""
    available_features = [f for f in features if f in df.columns and f != target_col]

    if not available_features:
        return {'warning': 'No features available'}

    if df[target_col].nunique() < 2:
        return {'warning': f'Target has only one class: {df[target_col].unique()}'}

    X = df[available_features].fillna(0)
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]

    balanced_acc = balanced_accuracy_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred_proba)

    coef_df = pd.DataFrame({
        'feature': available_features,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', ascending=False, key=abs)

    return {
        'coefficients': coef_df,
        'balanced_accuracy': balanced_acc,
        'roc_auc': roc_auc,
        'intercept': model.intercept_[0],
    }


def fit_small_collaboration_decision_tree(df, target_col, features, max_depth=3):
    """Fit small decision tree."""
    available_features = [f for f in features if f in df.columns and f != target_col]

    if not available_features:
        return {'warning': 'No features available'}

    if df[target_col].nunique() < 2:
        return {'warning': f'Target has only one class: {df[target_col].unique()}'}

    X = df[available_features].fillna(0)
    y = df[target_col]

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight='balanced',
        min_samples_leaf=30,
        random_state=42
    )
    tree.fit(X, y)

    y_pred = tree.predict(X)
    balanced_acc = balanced_accuracy_score(y, y_pred)
    rules = export_text(tree, feature_names=available_features)

    return {
        'rules': rules,
        'balanced_accuracy': balanced_acc,
    }


def find_recurring_collaboration_situations(df, target_col):
    """Find recurring situation patterns."""
    df_copy = df.copy()

    # Create categories
    if 'peer_reputation' in df_copy.columns:
        df_copy['reputation_level'] = pd.cut(df_copy['peer_reputation'], bins=3, labels=['low', 'medium', 'high'])
    elif 'mean_peer_reputation' in df_copy.columns:
        df_copy['reputation_level'] = pd.cut(df_copy['mean_peer_reputation'], bins=3, labels=['low', 'medium', 'high'])
    else:
        df_copy['reputation_level'] = 'unknown'

    if 'distance_to_self' in df_copy.columns:
        df_copy['distance_level'] = pd.cut(df_copy['distance_to_self'], bins=3, labels=['close', 'medium', 'far'])
    elif 'mean_peer_distance' in df_copy.columns:
        df_copy['distance_level'] = pd.cut(df_copy['mean_peer_distance'], bins=3, labels=['close', 'medium', 'far'])
    else:
        df_copy['distance_level'] = 'unknown'

    if 'total_remaining_effort' in df_copy.columns:
        df_copy['workload_level'] = pd.cut(df_copy['total_remaining_effort'], bins=3, labels=['low', 'medium', 'high'])
    else:
        df_copy['workload_level'] = 'unknown'

    if 'opportunity_prestige' in df_copy.columns:
        df_copy['project_quality_level'] = pd.cut(df_copy['opportunity_prestige'], bins=3, labels=['low', 'medium', 'high'])
    else:
        df_copy['project_quality_level'] = 'unknown'

    if 'age' in df_copy.columns:
        df_copy['episode_phase'] = pd.cut(df_copy['age'], bins=3, labels=['early', 'mid', 'late'])
    else:
        df_copy['episode_phase'] = 'unknown'

    if 'n_selected_peers' in df_copy.columns:
        df_copy['collaboration_size_level'] = pd.cut(
            df_copy['n_selected_peers'],
            bins=[0, 1, 2, 100],
            labels=['none', 'small', 'large'],
            include_lowest=True
        )
    else:
        df_copy['collaboration_size_level'] = 'unknown'

    category_cols = [c for c in [
        'reputation_level', 'distance_level', 'workload_level',
        'project_quality_level', 'episode_phase', 'collaboration_size_level'
    ] if c in df_copy.columns]

    grouped = df_copy.groupby(category_cols, observed=True).agg({
        target_col: ['count', 'mean']
    }).reset_index()

    grouped.columns = category_cols + ['n_samples', f'{target_col}_rate']
    return grouped.sort_values('n_samples', ascending=False)


def analyze_collaboration_conditioned_on_project_choice(step_df, peer_df):
    """Analyze collaboration conditioned on project choice."""
    results = {}

    print("\n=== Conditioned on Project Choice ===")

    # Step-level
    results['step_all'] = {
        'n': len(step_df),
        'collab_rate': step_df['collaborates_any'].mean() if len(step_df) > 0 else np.nan,
    }

    if 'accepted_project' in step_df.columns:
        step_acc = step_df[step_df['accepted_project'] == 1]
        step_rej = step_df[step_df['accepted_project'] == 0]

        results['step_accepted'] = {
            'n': len(step_acc),
            'collab_rate': step_acc['collaborates_any'].mean() if len(step_acc) > 0 else np.nan,
        }

        results['step_rejected'] = {
            'n': len(step_rej),
            'collab_rate': step_rej['collaborates_any'].mean() if len(step_rej) > 0 else np.nan,
        }

        print(f"Step all: n={len(step_df)}, rate={step_df['collaborates_any'].mean():.3f}")
        print(f"Step accepted: n={len(step_acc)}, rate={step_acc['collaborates_any'].mean():.3f}")
        print(f"Step rejected: n={len(step_rej)}, rate={step_rej['collaborates_any'].mean():.3f}")

    # Peer-level
    if 'accepted_project' in peer_df.columns:
        peer_acc = peer_df[peer_df['accepted_project'] == 1]
        peer_rej = peer_df[peer_df['accepted_project'] == 0]

        results['peer_accepted'] = {
            'n': len(peer_acc),
            'sel_rate': peer_acc['selected_peer'].mean() if len(peer_acc) > 0 else np.nan,
        }

        results['peer_rejected'] = {
            'n': len(peer_rej),
            'sel_rate': peer_rej['selected_peer'].mean() if len(peer_rej) > 0 else np.nan,
        }

        print(f"Peer accepted: n={len(peer_acc)}, rate={peer_acc['selected_peer'].mean():.3f}")
        print(f"Peer rejected: n={len(peer_rej)}, rate={peer_rej['selected_peer'].mean():.3f}")

    return results


def run_collaboration_analysis(step_df, peer_df):
    """Run complete collaboration analysis."""
    results = {}

    print("\n" + "="*80)
    print("STEP-LEVEL ANALYSIS: collaborates_any")
    print("="*80)

    step_features = [
        'n_available_peers', 'n_valid_collab_actions',
        'mean_peer_reputation', 'max_peer_reputation', 'std_peer_reputation',
        'mean_peer_distance', 'min_peer_distance', 'max_peer_distance',
        'n_running_projects', 'mean_running_progress', 'max_running_progress',
        'mean_running_time_left', 'min_running_time_left',
        'mean_remaining_effort', 'total_remaining_effort', 'mean_effort_per_time_left',
        'opportunity_required_effort', 'opportunity_prestige', 'opportunity_novelty',
        'opportunity_time_window', 'opportunity_effort_per_time',
        'age', 'accumulated_rewards',
    ]

    print("Computing feature summary...")
    results['step_feature_summary'] = summarize_collaboration_by_feature(step_df, 'collaborates_any', step_features)

    print("Computing correlations...")
    results['step_correlations'] = compute_collaboration_correlations(step_df, 'collaborates_any', step_features)

    print("Finding thresholds...")
    results['step_threshold_report'] = find_simple_collaboration_thresholds(step_df, 'collaborates_any', step_features)

    print("Fitting logistic regression...")
    results['step_logistic_report'] = fit_collaboration_logistic_regression(step_df, 'collaborates_any', step_features)

    print("Fitting decision tree...")
    results['step_tree_report'] = fit_small_collaboration_decision_tree(step_df, 'collaborates_any', step_features)

    print("Finding situations...")
    results['step_situation_report'] = find_recurring_collaboration_situations(step_df, 'collaborates_any')

    print("\n" + "="*80)
    print("PEER-LEVEL ANALYSIS: selected_peer")
    print("="*80)

    peer_features = [
        'peer_slot_index', 'peer_reputation', 'peer_reputation_rank',
        'is_top_reputation_peer', 'is_closest_peer', 'is_far_peer',
        'distance_to_self', 'peer_distance_rank',
        'peer_centroid_x', 'peer_centroid_y', 'self_centroid_x', 'self_centroid_y',
        'n_running_projects', 'mean_running_progress', 'max_running_progress',
        'total_remaining_effort', 'opportunity_prestige', 'opportunity_novelty',
        'age', 'accumulated_rewards',
    ]

    print("Computing feature summary...")
    results['peer_feature_summary'] = summarize_collaboration_by_feature(peer_df, 'selected_peer', peer_features)

    print("Computing correlations...")
    results['peer_correlations'] = compute_collaboration_correlations(peer_df, 'selected_peer', peer_features)

    print("Finding thresholds...")
    results['peer_threshold_report'] = find_simple_collaboration_thresholds(peer_df, 'selected_peer', peer_features)

    print("Fitting logistic regression...")
    results['peer_logistic_report'] = fit_collaboration_logistic_regression(peer_df, 'selected_peer', peer_features)

    print("Fitting decision tree...")
    results['peer_tree_report'] = fit_small_collaboration_decision_tree(peer_df, 'selected_peer', peer_features)

    print("Finding situations...")
    results['peer_situation_report'] = find_recurring_collaboration_situations(peer_df, 'selected_peer')

    print("\n" + "="*80)
    print("CONDITIONED ANALYSIS")
    print("="*80)

    results['conditioned_on_project_choice'] = analyze_collaboration_conditioned_on_project_choice(step_df, peer_df)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

    return results


def analyze_slot_bias(peer_df):
    """Check for slot bias in peer selection."""
    print("\n" + "="*80)
    print("SLOT BIAS ANALYSIS")
    print("="*80)

    slot_analysis = peer_df.groupby('peer_slot_index').agg({
        'selected_peer': ['count', 'mean', 'sum']
    }).reset_index()

    slot_analysis.columns = ['peer_slot_index', 'n_samples', 'selection_rate', 'n_selected']
    slot_analysis = slot_analysis.sort_values('peer_slot_index')

    print(slot_analysis)
    print("\nInterpretation: Significant variation suggests slot bias.")

    return slot_analysis


def find_recurring_situations_peer_level(peer_df, min_samples=30):
    """
    Find recurring situation patterns for peer-level selection decisions.

    Focus on key features:
    - peer_reputation: binned into low/medium/high
    - distance_to_self: binned into close/medium/far
    - peer_slot_index: binned into early/mid/late slots

    Parameters
    ----------
    peer_df : DataFrame
        Peer-level dataframe with selection decisions
    min_samples : int, default=30
        Minimum number of samples required per situation group

    Returns
    -------
    DataFrame
        Grouped situations with selection rates, sorted by sample count
    """
    df = peer_df.copy()

    # Overall selection rate for computing deviations
    overall_selection_rate = df['selected_peer'].mean()

    # 1. Reputation level - data-driven tertile bins
    if 'peer_reputation' in df.columns:
        try:
            df['reputation_level'] = pd.qcut(
                df['peer_reputation'],
                q=3,
                labels=['low', 'medium', 'high'],
                duplicates='drop'
            )
        except ValueError as e:
            print(f"Warning: Could not create tertile bins for 'peer_reputation': {e}")
            df['reputation_level'] = 'unknown'

    # 2. Distance level - data-driven tertile bins
    if 'distance_to_self' in df.columns:
        try:
            df['distance_level'] = pd.qcut(
                df['distance_to_self'],
                q=3,
                labels=['close', 'medium', 'far'],
                duplicates='drop'
            )
        except ValueError as e:
            print(f"Warning: Could not create tertile bins for 'distance_to_self': {e}")
            df['distance_level'] = 'unknown'

    # 3. Slot position - binned into early/mid/late
    if 'peer_slot_index' in df.columns:
        max_slot = df['peer_slot_index'].max()
        if max_slot >= 3:
            df['slot_position'] = pd.cut(
                df['peer_slot_index'],
                bins=[-0.1, max_slot/3, 2*max_slot/3, max_slot+1],
                labels=['early', 'mid', 'late']
            )
        else:
            df['slot_position'] = 'unknown'

    # Collect situation columns that were successfully created
    situation_cols = []
    for col in ['reputation_level', 'distance_level', 'slot_position']:
        if col in df.columns:
            situation_cols.append(col)

    if not situation_cols:
        print("Warning: No situation dimensions could be created")
        return pd.DataFrame()

    # Check for unexpected NaNs in situation columns
    n_before = len(df)
    df_clean = df.dropna(subset=situation_cols)
    n_after = len(df_clean)
    if n_after < n_before:
        print(f"Warning: Lost {n_before - n_after} rows ({100*(n_before-n_after)/n_before:.1f}%) due to NaN in situation columns")

    # Group by situations
    agg_dict = {'selected_peer': ['count', 'mean']}

    # Add mean values for key features if they exist
    if 'peer_reputation' in df_clean.columns:
        agg_dict['peer_reputation'] = 'mean'
    if 'distance_to_self' in df_clean.columns:
        agg_dict['distance_to_self'] = 'mean'
    if 'peer_slot_index' in df_clean.columns:
        agg_dict['peer_slot_index'] = 'mean'

    grouped = df_clean.groupby(situation_cols, observed=True).agg(agg_dict)

    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns]
    grouped = grouped.reset_index()

    # Rename standard columns
    grouped = grouped.rename(columns={
        'selected_peer_count': 'n_samples',
        'selected_peer_mean': 'selection_rate',
    })

    # Filter groups by minimum sample size
    grouped = grouped[grouped['n_samples'] >= min_samples]

    # Compute deviation from overall mean
    grouped['selection_deviation'] = grouped['selection_rate'] - overall_selection_rate

    # Sort by absolute deviation (most extreme situations first), then by sample size
    grouped = grouped.sort_values(
        ['selection_deviation', 'n_samples'],
        ascending=[False, False],
        key=lambda col: col.abs() if col.name == 'selection_deviation' else col
    )

    return grouped


def validate_centroid_extraction(peer_df):
    """Validate that centroid extraction worked correctly.

    This function checks:
    - Descriptive statistics of self_centroid_x and self_centroid_y
    - Count of NaN values in centroid and distance fields
    - Number of unique self-centroids
    - Warnings if centroids are all zero or all NaN
    """
    print("\n" + "="*80)
    print("CENTROID EXTRACTION VALIDATION")
    print("="*80)

    # Check if required columns exist
    required_cols = ['self_centroid_x', 'self_centroid_y', 'distance_to_self']
    missing_cols = [col for col in required_cols if col not in peer_df.columns]

    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
        return

    # Descriptive statistics
    print("\nSelf Centroid Descriptive Statistics:")
    print(peer_df[['self_centroid_x', 'self_centroid_y']].describe())

    # Count NaN values
    print("\nNaN Counts:")
    print(f"  self_centroid_x: {peer_df['self_centroid_x'].isna().sum():,} / {len(peer_df):,} ({100*peer_df['self_centroid_x'].isna().sum()/len(peer_df):.2f}%)")
    print(f"  self_centroid_y: {peer_df['self_centroid_y'].isna().sum():,} / {len(peer_df):,} ({100*peer_df['self_centroid_y'].isna().sum()/len(peer_df):.2f}%)")
    print(f"  distance_to_self: {peer_df['distance_to_self'].isna().sum():,} / {len(peer_df):,} ({100*peer_df['distance_to_self'].isna().sum()/len(peer_df):.2f}%)")

    # Count unique centroids
    unique_centroids = peer_df[['self_centroid_x', 'self_centroid_y']].dropna().drop_duplicates()
    print(f"\nUnique Self Centroids: {len(unique_centroids):,}")

    # Warnings
    print("\nValidation Checks:")

    # Check if all zeros
    non_nan_x = peer_df['self_centroid_x'].dropna()
    non_nan_y = peer_df['self_centroid_y'].dropna()

    if len(non_nan_x) > 0 and len(non_nan_y) > 0:
        if (non_nan_x == 0.0).all() and (non_nan_y == 0.0).all():
            print("  ⚠ WARNING: self_centroid_x and self_centroid_y are both constant 0.0!")
            print("    -> Centroid parsing likely failed")
        elif non_nan_x.std() == 0 and non_nan_y.std() == 0:
            print("  ⚠ WARNING: self_centroid_x and self_centroid_y have zero variance!")
            print("    -> All centroids are identical")
        else:
            print("  ✓ PASS: Self centroids have non-zero variance")

    # Check if all NaN
    if peer_df['distance_to_self'].isna().all():
        print("  ⚠ WARNING: distance_to_self is completely NaN!")
        print("    -> Distance calculation failed")
    elif peer_df['distance_to_self'].isna().mean() > 0.5:
        print(f"  ⚠ WARNING: distance_to_self is {100*peer_df['distance_to_self'].isna().mean():.1f}% NaN")
    else:
        print(f"  ✓ PASS: distance_to_self is {100*(1-peer_df['distance_to_self'].isna().mean()):.1f}% valid")

    # Distance statistics
    if not peer_df['distance_to_self'].isna().all():
        print("\nDistance to Self Statistics:")
        print(peer_df['distance_to_self'].describe())

    print("\n" + "="*80)
