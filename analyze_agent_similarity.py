"""
Script to analyze similarity between RL agent and archetypes in Game of Science.

Assumptions:
1. choose_project: 0 = "No project", index > 0 => project_opportunities["project_{index-1}"]
2. collaborate_with: binary vector, only slots with action_mask["collaborate_with"] > 0 are valid.
3. put_effort: 0 = "No effort", index > 0 => maps to running_projects in alphabetical order of keys (e.g., project_0, project_1).
4. Features are normalized where appropriate (prestige, novelty).
5. Aggregates are computed per-timestep and overall.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Loads a JSONL file into a list of dictionaries."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def safe_mean(values: List[float]) -> float:
    """Computes mean safely, returning 0.0 if empty."""
    return float(np.mean(values)) if values else 0.0

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Computes Euclidean distance between two vectors."""
    return float(np.linalg.norm(a - b))

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Computes Cosine similarity between two vectors."""
    a_reshaped = a.reshape(1, -1)
    b_reshaped = b.reshape(1, -1)
    return float(cosine_similarity(a_reshaped, b_reshaped)[0, 0])

def extract_project_choice_features(action: Dict[str, Any], obs_entry: Dict[str, Any]) -> Dict[str, float]:
    """Extracts features related to project choice."""
    choice = action.get("choose_project", 0)
    features = {
        "choose_project_binary": 1.0 if choice > 0 else 0.0,
        "chosen_project_required_effort": 0.0,
        "chosen_project_prestige": 0.0,
        "chosen_project_novelty": 0.0,
        "chosen_project_time_window": 0.0
    }
    
    if choice > 0:
        proj_key = f"project_{choice - 1}"
        opportunities = obs_entry.get("observation", {}).get("project_opportunities", {})
        proj_data = opportunities.get(proj_key)
        if proj_data:
            features["chosen_project_required_effort"] = float(np.asarray(proj_data.get("required_effort", [0])).item())
            features["chosen_project_prestige"] = float(np.asarray(proj_data.get("prestige", [0.0])).item())
            features["chosen_project_novelty"] = float(np.asarray(proj_data.get("novelty", [0.0])).item())
            features["chosen_project_time_window"] = float(np.asarray(proj_data.get("time_window", [0])).item())
            
    return features

def extract_valid_collaboration_features(action: Dict[str, Any], obs_entry: Dict[str, Any]) -> Dict[str, float]:
    """Extracts features related to collaboration."""
    collab_bits = np.asarray(action.get("collaborate_with", []), dtype=np.int8)
    mask = np.asarray(obs_entry.get("action_mask", {}).get("collaborate_with", []), dtype=np.int8)
    obs = obs_entry.get("observation", {})
    
    # Only consider valid slots
    valid_indices = np.where(mask > 0)[0]
    if len(collab_bits) == 0:
        return {
            "n_selected_collaborators": 0.0,
            "mean_selected_peer_reputation": 0.0,
            "mean_selected_peer_distance_to_self_centroid": 0.0
        }
    
    selected_indices = [i for i in valid_indices if i < len(collab_bits) and collab_bits[i] == 1]
    
    n_selected = len(selected_indices)
    reps = []
    dists = []
    
    peer_reputation = np.asarray(obs.get("peer_reputation", []))
    peer_centroids = np.asarray(obs.get("peer_centroids", []))
    self_centroid = np.asarray(obs.get("self_centroid", [0, 0])).reshape(-1)
    
    for idx in selected_indices:
        if idx < len(peer_reputation):
            reps.append(float(peer_reputation[idx]))
        if idx < len(peer_centroids):
            p_centroid = peer_centroids[idx]
            dists.append(float(np.linalg.norm(p_centroid - self_centroid)))
            
    return {
        "n_selected_collaborators": float(n_selected),
        "mean_selected_peer_reputation": safe_mean(reps),
        "mean_selected_peer_distance_to_self_centroid": safe_mean(dists)
    }

def extract_effort_features(action: Dict[str, Any], obs_entry: Dict[str, Any]) -> Dict[str, float]:
    """Extracts features related to effort allocation."""
    effort_idx = action.get("put_effort", 0)
    obs = obs_entry.get("observation", {})
    running_projs = obs.get("running_projects", {})
    
    # Sort keys to ensure deterministic mapping
    proj_keys = sorted(running_projs.keys())
    
    features = {
        "effort_is_none": 1.0 if effort_idx == 0 else 0.0,
        "effort_is_active": 1.0 if effort_idx > 0 else 0.0,
        "effort_target_required_effort": 0.0,
        "effort_target_prestige": 0.0,
        "effort_target_novelty": 0.0,
        "effort_target_time_left": 0.0,
        "effort_target_current_effort": 0.0,
        "effort_target_n_contributors": 0.0,
        "effort_target_mean_peer_fit": 0.0
    }
    
    if effort_idx > 0 and (effort_idx - 1) < len(proj_keys):
        target_key = proj_keys[effort_idx - 1]
        p = running_projs[target_key]
        features["effort_target_required_effort"] = float(np.asarray(p.get("required_effort", [0])).item())
        features["effort_target_prestige"] = float(np.asarray(p.get("prestige", [0.0])).item())
        features["effort_target_novelty"] = float(np.asarray(p.get("novelty", [0.0])).item())
        features["effort_target_time_left"] = float(np.asarray(p.get("time_left", [0])).item())
        features["effort_target_current_effort"] = float(np.asarray(p.get("current_effort", [0.0])).item())
        
        contributors = np.asarray(p.get("contributors", []))
        features["effort_target_n_contributors"] = float(np.sum(contributors))
        
        peer_fit = np.asarray(p.get("peer_fit", []))
        # Filter peer_fit for active contributors
        active_fit = peer_fit[contributors > 0] if len(contributors) == len(peer_fit) else peer_fit
        features["effort_target_mean_peer_fit"] = safe_mean(active_fit.tolist())
        
    return features

def build_action_feature_vector(action_entry: Dict[str, Any], observation_entry: Dict[str, Any]) -> Optional[np.ndarray]:
    """Combines all extracted features into a single vector."""
    if action_entry is None or observation_entry is None:
        return None
        
    f_proj = extract_project_choice_features(action_entry, observation_entry)
    f_collab = extract_valid_collaboration_features(action_entry, observation_entry)
    f_effort = extract_effort_features(action_entry, observation_entry)
    
    # Merge and convert to vector
    all_features = {**f_proj, **f_collab, **f_effort}
    # Ensure stable order by sorting keys
    sorted_values = [all_features[k] for k in sorted(all_features.keys())]
    return np.array(sorted_values, dtype=np.float32)

def compute_rl_vs_archetype_distances(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Computes similarity metrics between RL agent and archetypes."""
    results = []
    
    timesteps = feature_df['timestep'].unique()
    for t in timesteps:
        t_data = feature_df[feature_df['timestep'] == t]
        rl_agents = t_data[t_data['archetype'] == 'rl_agent']
        
        if rl_agents.empty:
            continue
            
        # Usually one RL agent, but handle multiple if exists
        for _, rl_row in rl_agents.iterrows():
            rl_vec = rl_row['feature_vector']
            
            archetypes = ['careerist', 'orthodox_scientist', 'mass_producer']
            for arch in archetypes:
                arch_agents = t_data[t_data['archetype'] == arch]
                if arch_agents.empty:
                    continue
                
                eucs = []
                coss = []
                for _, arch_row in arch_agents.iterrows():
                    arch_vec = arch_row['feature_vector']
                    eucs.append(euclidean_distance(rl_vec, arch_vec))
                    coss.append(cosine_sim(rl_vec, arch_vec))
                
                results.append({
                    "timestep": t,
                    "archetype": arch,
                    "mean_euclidean": np.mean(eucs),
                    "std_euclidean": np.std(eucs),
                    "mean_cosine": np.mean(coss),
                    "std_cosine": np.std(coss),
                    "n_comparisons": len(eucs)
                })
                
    return pd.DataFrame(results)

def plot_similarity_over_time(results_df: pd.DataFrame, output_dir: Path, prefix: str, rolling_window: int = 10):
    """Generates plots for similarity metrics over time, including rolling averages."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize prefix for filename (remove potential directory separators)
    filename_prefix = Path(prefix).name
    
    # Common plotting parameters
    colors = {
        "careerist": "tab:blue",
        "orthodox_scientist": "tab:green",
        "mass_producer": "tab:red",
        "unknown": "tab:gray"
    }

    # Helper function for plotting a metric
    def _plot_metric(metric_name: str, ylabel: str, title_prefix: str, suffix: str):
        plt.figure(figsize=(14, 7))
        for arch in results_df['archetype'].unique():
            data = results_df[results_df['archetype'] == arch].sort_values('timestep')
            color = colors.get(arch, None)
            
            # Original data with low alpha
            plt.plot(data['timestep'], data[metric_name], alpha=0.2, color=color)
            
            # Rolling mean
            rolling = data[metric_name].rolling(window=rolling_window, min_periods=1, center=True).mean()
            plt.plot(data['timestep'], rolling, label=f"{arch} (avg)", linewidth=2.5, color=color)
            
        plt.title(f"{title_prefix}: RL Agent vs Archetypes (Window={rolling_window})\n{filename_prefix}")
        plt.xlabel("Timestep")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{filename_prefix}_{suffix}.png", dpi=150)
        plt.close()

    _plot_metric("mean_euclidean", "Euclidean Distance (lower is more similar)", "Mean Euclidean Distance", "euclidean_similarity")
    _plot_metric("mean_cosine", "Cosine Similarity (higher is more similar)", "Mean Cosine Similarity", "cosine_similarity")

    # Distribution plot (Boxplot)
    plt.figure(figsize=(10, 6))
    data_to_plot = [results_df[results_df['archetype'] == arch]['mean_cosine'].dropna() for arch in results_df['archetype'].unique()]
    labels = list(results_df['archetype'].unique())
    
    # Use tick_labels for modern matplotlib, but keep labels for compatibility if needed
    try:
        plt.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
    except TypeError:
        plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
    plt.title(f"Distribution of Cosine Similarity by Archetype\n{filename_prefix}")
    plt.ylabel("Mean Cosine Similarity")
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{filename_prefix}_cosine_distribution.png", dpi=150)
    plt.close()

def main(actions_path: str, observations_path: str, output_prefix: str = "similarity_analysis", rolling_window: int = 10):
    """Main execution flow."""
    print(f"Loading data from {actions_path} and {observations_path}...")
    actions_data = load_jsonl(Path(actions_path))
    obs_data = load_jsonl(Path(observations_path))
    
    num_timesteps = min(len(actions_data), len(obs_data))
    all_agent_features = []
    
    print(f"Processing {num_timesteps} timesteps...")
    for t in range(num_timesteps):
        step_actions = actions_data[t]
        step_obs = obs_data[t]
        
        for agent_id, action in step_actions.items():
            if action is None:
                continue
            
            obs_entry = step_obs.get(agent_id)
            if obs_entry is None:
                continue
                
            vec = build_action_feature_vector(action, obs_entry)
            if vec is not None:
                all_agent_features.append({
                    "timestep": t,
                    "agent_id": agent_id,
                    "archetype": action.get("archetype", "unknown"),
                    "feature_vector": vec
                })
                
    feature_df = pd.DataFrame(all_agent_features)
    if feature_df.empty:
        print("No features extracted. Check input data.")
        return

    print("Computing similarities...")
    results_df = compute_rl_vs_archetype_distances(feature_df)
    
    if results_df.empty:
        print("No similarity results computed. Check archetype names and RL agent presence.")
        return
        
    # Per-timestep results
    results_df.to_csv(f"{output_prefix}_timestep_results.csv", index=False)
    
    # Overall summary
    summary = results_df.groupby('archetype').agg({
        'mean_euclidean': 'mean',
        'mean_cosine': 'mean',
        'n_comparisons': 'sum'
    }).reset_index()
    summary.rename(columns={
        'mean_euclidean': 'overall_mean_euclidean',
        'mean_cosine': 'overall_mean_cosine'
    }, inplace=True)
    
    summary.to_csv(f"{output_prefix}_summary.csv", index=False)
    print("\nOverall Summary:")
    print(summary)
    
    print("\nGenerating plots...")
    plot_similarity_over_time(results_df, Path("results"), output_prefix, rolling_window=rolling_window)
    print(f"Analysis complete. Results saved with prefix: {output_prefix}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze similarity between RL agent and archetypes.")
    parser.add_argument("--actions", type=str, required=True, help="Path to actions.jsonl")
    parser.add_argument("--obs", type=str, required=True, help="Path to observations.jsonl")
    parser.add_argument("--output", type=str, default="similarity_analysis", help="Prefix for output files")
    parser.add_argument("--rolling", type=int, default=10, help="Window size for rolling average plots")
    
    args = parser.parse_args()
    main(args.actions, args.obs, args.output, rolling_window=args.rolling)
