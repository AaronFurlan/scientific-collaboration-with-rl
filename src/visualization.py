from collections import Counter, defaultdict

from src.env.area import Area
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import pandas as pd
import os
from datetime import datetime

from src.visualization_utils import get_path_for_strat
from src.analysis import (
    contributors_per_project,
    success_rate_over_time,
    quality_metrics_over_time,
    collab_metrics_over_time,
    projects_per_contributor,
    completion_per_project,
    quality_per_project
)
# Color palette for archetypes
THESIS_COLORS = {
    "careerist": "#2E5090",
    "orthodox_scientist": "#228B22",
    "mass_producer": "#D2691E",
    "rl_agent": "#DC143C",
    "random": "#FF8C00",
    "unknown": "#696969"
}
plt.rcParams['text.antialiased'] = True
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linewidth'] = 0.5

def save_plot(filename=None, output_dir="plots", dpi=600, bbox_inches="tight",
              timestamp=True, fig=None):
    """Save matplotlib figure as PNG and PDF."""
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename if not provided
    if filename is None:
        filename = f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        filename = filename.replace(".png", "").replace(".pdf", "")
        if timestamp:
            filename = f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    filepath_png = os.path.join(output_dir, f"{filename}.png")
    filepath_pdf = os.path.join(output_dir, f"{filename}.pdf")
    os.makedirs(os.path.dirname(filepath_png), exist_ok=True)
    if fig is None:
        fig = plt.gcf()
    fig.savefig(filepath_png, dpi=dpi, bbox_inches=bbox_inches)
    fig.savefig(filepath_pdf, bbox_inches=bbox_inches)
    print(f"Saved plot to: {filepath_png}")
    print(f"Saved plot to: {filepath_pdf}")

    return filepath_png, filepath_pdf

def topic_area_per_project(projects, actions, area_pickle_file, observations=None, filename=None, **kwargs):
    """Visualize project distribution in knowledge space by archetype."""
    area = Area.load(area_pickle_file)
    papers = []
    agent_to_archetype = {}

    # RL Agent ID identifizieren
    rl_agent_id = None

    for step in actions:
        for agent, action in step.items():
            if action is not None:
                arch = action.get("archetype", "rl_agent")
                if arch == "rl_agent":
                    rl_agent_id = agent
                if agent not in agent_to_archetype:
                    agent_to_archetype[agent] = arch

    for p in projects:
        # Bestimme den Haupt-Archetyp des Projekts
        contributor_archetypes = [agent_to_archetype.get(f"agent_{c}", "unknown") for c in p["contributors"]]

        # Prüfen ob der RL Agent beteiligt war
        is_rl_project = any(f"agent_{c}" == rl_agent_id for c in p["contributors"]) if rl_agent_id else False

        if is_rl_project:
            main_archetype = "rl_agent"
        else:
            counts = Counter(contributor_archetypes)
            if counts:
                main_archetype = counts.most_common(1)[0][0]
            else:
                main_archetype = "unknown"

        papers.append((*p["kene"], main_archetype))

    rl_agent_positions = []
    if observations and rl_agent_id:
        for step in observations:
            if rl_agent_id in step:
                obs = step[rl_agent_id].get("observation", {})
                centroid = obs.get("self_centroid")
                if centroid:
                    if isinstance(centroid, list) and len(centroid) > 0:
                        if isinstance(centroid[0], list):
                            rl_agent_positions.append(tuple(centroid[0]))
                        else:
                            rl_agent_positions.append(tuple(centroid))

    print("\n--- Publications per Archetype ---")
    arch_counts = Counter([p[2] for p in papers])
    for arch in ["careerist", "orthodox_scientist", "mass_producer", "rl_agent"]:
        print(f"{arch:18}: {arch_counts.get(arch, 0)}")

    visualize_knowledge_space(area, sampled_points=papers, agent_positions=rl_agent_positions, filename=filename)


def visualize_knowledge_space(area, resolution=200, sampled_points=None, bounds=None, agent_positions=None, filename=None):
    """Visualize knowledge space with optional agent trajectory."""
    if bounds is None:
        xmin, xmax = area.xlim
        ymin, ymax = area.ylim
    else:
        xmin, xmax, ymin, ymax = bounds

    x = np.linspace(xmin, xmax, resolution)
    y = np.linspace(ymin, ymax, resolution)
    X, Y = np.meshgrid(x, y)

    Z = 0
    for x0, y0, sigma, v in area.areas:
        Z += v * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2))
    Z = np.tanh(Z)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(
        Z,
        extent=(xmin, xmax, ymin, ymax),
        origin="lower",
        cmap="bwr",
        vmin=-1.0,
        vmax=1.0,
        alpha=0.3,
    )
    plt.colorbar(im, ax=ax, label="Scientific Value")

    if sampled_points is not None and len(sampled_points) > 0:
        has_category = len(sampled_points[0]) == 3

        if has_category:
            category_points = defaultdict(list)
            for px, py, cat in sampled_points:
                if xmin <= px <= xmax and ymin <= py <= ymax:
                    category_points[cat].append((px, py))

            colors = THESIS_COLORS.copy()

            for cat in ["careerist", "orthodox_scientist", "mass_producer", "unknown"]:
                if cat not in category_points:
                    if cat != "unknown":
                        print(f"Note: No papers found for archetype '{cat}' in this simulation.")
                    continue

                pts = category_points[cat]
                xs, ys = zip(*pts)
                ax.scatter(
                    xs, ys,
                    c=colors.get(cat, "gray"),
                    s=20,
                    alpha=0.6,
                    edgecolors="white",
                    linewidth=0.5,
                    label=cat.replace("_", " ")
                )

            # Dann RL Agent obenauf und größer/deutlicher
            if "rl_agent" in category_points:
                xs, ys = zip(*category_points["rl_agent"])
                ax.scatter(
                    xs, ys,
                    c=THESIS_COLORS["rl_agent"],
                    s=80,
                    alpha=1.0,
                    edgecolors="black",
                    linewidth=1.5,
                    marker="*",
                    label="RL Agent (PPO)"
                )
        else:
            px, py = zip(*[(px, py) for (px, py) in sampled_points if xmin <= px <= xmax and ymin <= py <= ymax])
            ax.scatter(px, py, c="black", s=10, edgecolors="white", label="Papers")

    ax.legend(loc="upper left", bbox_to_anchor=(1.15, 1))

    # Plot RL Agent trajectory/final position if provided
    if agent_positions and len(agent_positions) > 0:
        axs, ays = zip(*agent_positions)
        # Plot trajectory
        ax.plot(axs, ays, color="red", linestyle="--", linewidth=1, alpha=0.5, label="RL Agent Path")
        # Plot final position
        ax.scatter(axs[-1], ays[-1], color="red", marker="X", s=200, edgecolors="black", linewidth=2, label="RL Agent Final Pos", zorder=10)

    ax.set_title("Knowledge Space with Published Papers")
    ax.set_xlabel("Topic Dimension 1")
    ax.set_ylabel("Topic Dimension 2")

    # Update legend to include RL Agent trajectory if it was added
    ax.legend(loc="upper left", bbox_to_anchor=(1.15, 1))

    plt.tight_layout()

    if filename:
        save_plot(filename=filename, fig=fig)

    plt.show()

def animate_knowledge_space(projects, actions, area_pickle_file, interval=200, steps_per_frame=5, observations=None):
    """
    Erstellt eine Animation des Knowledge Space über die Zeit.
    """
    area = Area.load(area_pickle_file)
    agent_to_archetype = {}
    rl_agent_id = None

    for step in actions:
        for agent, action in step.items():
            if action is not None:
                arch = action.get("archetype", "rl_agent")
                if arch == "rl_agent":
                    rl_agent_id = agent
                if agent not in agent_to_archetype:
                    agent_to_archetype[agent] = arch

    # Extrahiere RL Agent Trajektorie
    rl_traj = []
    if observations and rl_agent_id:
        for step_idx, step in enumerate(observations):
            if rl_agent_id in step:
                obs = step[rl_agent_id].get("observation", {})
                centroid = obs.get("self_centroid")
                if centroid and isinstance(centroid, list) and len(centroid) > 0:
                    pos = centroid[0] if isinstance(centroid[0], list) else centroid
                    rl_traj.append({"time": step_idx, "x": pos[0], "y": pos[1]})

    # Vorbereitung der Paper-Daten mit Zeitstempel
    paper_data = []
    for p in projects:
        if not p.get("finished", True): continue

        contributor_archetypes = [agent_to_archetype.get(f"agent_{c}", "unknown") for c in p["contributors"]]
        is_rl_project = any(f"agent_{c}" == rl_agent_id for c in p["contributors"]) if rl_agent_id else False

        if is_rl_project:
            main_archetype = "rl_agent"
        else:
            counts = Counter(contributor_archetypes)
            main_archetype = counts.most_common(1)[0][0] if counts else "unknown"

        # Wir nehmen an, dass start_time + duration (oder ähnliches) das Publikationsdatum ist.
        # Da wir nur start_time haben und die Projekte oft kurz sind, nutzen wir start_time als Näherung
        # oder schauen, ob es ein 'finish_time' gibt.
        # Falls 'finish_time' nicht da ist, nutzen wir start_time.
        pub_time = p.get("finish_time", p.get("start_time", 0))
        paper_data.append({
            "x": p["kene"][0],
            "y": p["kene"][1],
            "archetype": main_archetype,
            "time": pub_time
        })

    # Sortieren nach Zeit
    paper_data.sort(key=lambda x: x["time"])

    # Plot Setup
    xmin, xmax = area.xlim
    ymin, ymax = area.ylim
    res = 100  # Etwas niedrigere Auflösung für schnellere Animation
    x = np.linspace(xmin, xmax, res)
    y = np.linspace(ymin, ymax, res)
    X, Y = np.meshgrid(x, y)
    Z = 0
    for x0, y0, sigma, v in area.areas:
        Z += v * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2))
    Z = np.tanh(Z)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(Z, extent=(xmin, xmax, ymin, ymax), origin="lower", cmap="bwr", vmin=-1.0, vmax=1.0, alpha=0.3)

    colors = THESIS_COLORS.copy()
    scatters = {}
    for arch, color in colors.items():
        if arch == "rl_agent":
            scatters[arch] = ax.scatter([], [], c=color, s=100, marker="*", edgecolors="black", label=arch, zorder=5)
        else:
            scatters[arch] = ax.scatter([], [], c=color, s=20, alpha=0.6, edgecolors="white", label=arch, zorder=3)

    # Agent Trajectory Line and Current Position Marker
    agent_path, = ax.plot([], [], color=THESIS_COLORS["rl_agent"], linestyle="--", linewidth=1, alpha=0.5, zorder=4, label="RL Agent Path")
    agent_pos = ax.scatter([], [], color=THESIS_COLORS["rl_agent"], marker="X", s=150, edgecolors="black", linewidth=2, zorder=10, label="RL Agent Current")

    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_xlabel("Topic Dimension 1")
    ax.set_ylabel("Topic Dimension 2")
    title = ax.set_title("Knowledge Space Evolution - Step 0")

    # Bestimme max_time aus Paper-Daten und Trajektorie
    paper_times = [p["time"] for p in paper_data] if paper_data else [0]
    traj_times = [t["time"] for t in rl_traj] if rl_traj else [0]
    max_time = max(max(paper_times), max(traj_times), 100)
    
    frames = range(0, int(max_time) + 1, steps_per_frame)

    def update(frame):
        current_papers = [p for p in paper_data if p["time"] <= frame]
        
        # Trajektorie bis zum aktuellen Frame
        current_traj = [t for t in rl_traj if t["time"] <= frame]
        if current_traj:
            tx = [t["x"] for t in current_traj]
            ty = [t["y"] for t in current_traj]
            agent_path.set_data(tx, ty)
            agent_pos.set_offsets([[tx[-1], ty[-1]]])
        else:
            agent_path.set_data([], [])
            agent_pos.set_offsets(np.empty((0, 2)))

        # Gruppieren nach Archetyp
        arch_groups = defaultdict(list)
        for p in current_papers:
            arch_groups[p["archetype"]].append((p["x"], p["y"]))

        for arch, pts in arch_groups.items():
            if arch in scatters:
                if pts:
                    scatters[arch].set_offsets(pts)
                else:
                    scatters[arch].set_offsets(np.empty((0, 2)))

        title.set_text(f"Knowledge Space Evolution - Step {frame}")
        return list(scatters.values()) + [agent_path, agent_pos, title]

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
    plt.close()  # Verhindert doppelte Anzeige im Notebook
    return ani

def plot_histograms(projects, actions, filename=None):
    """
    Plot various histogram metrics.

    Args:
        filename: Optional base filename for saving plots (will append metric names)
    """
    contributors_per_project(projects)
    success_rate_over_time(projects)
    quality_metrics_over_time(projects)
    collab_metrics_over_time(projects)
    projects_per_contributor(projects)
    completion_per_project(projects)
    quality_per_project(projects)

def plot_mean_reward_trajectories_with_rl(dfs_heuristic, df_rl, strategy_name="by_effort", normalize=True, filename=None):
    """
    Plot mean reward trajectories comparing heuristic agents with RL agent.

    Args:
        filename: Optional filename for saving the plot
    """
    df_h, df_r = dfs_heuristic.copy(), df_rl.copy()
    all_archetypes = sorted(list(df_h["archetype"].unique()))

    # Use thesis color scheme
    color_map = THESIS_COLORS.copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Normalize if requested
    if normalize and not df_h.empty and not df_r.empty:
        h_max = df_h["mean_reward"].max() if not df_h.empty else 0
        r_max = df_r["mean_reward"].max() if not df_r.empty else 0
        max_val = max(h_max, r_max)
        if max_val > 0:
            if not df_h.empty:
                df_h["mean_reward"] /= max_val
            if not df_r.empty:
                df_r["mean_reward"] /= max_val

    # Plot heuristic agents
    for archetype, group_arch in df_h.groupby("archetype"):
        summary_h = group_arch.groupby("step")["mean_reward"].mean().reset_index()
        plt.plot(summary_h["step"], summary_h["mean_reward"], label=archetype, color=color_map.get(archetype, "gray"), lw=2, alpha=0.7)

    # Plot controlled agent (RL or Random)
    if not df_r.empty:
        summary_rl = df_r.groupby("step").agg(mean_across_seeds=("mean_reward", "mean"), std_across_seeds=("mean_reward", "std")).reset_index()
        controlled_archetype = df_r["archetype"].iloc[0]  # Get the actual archetype name
        label = "Random Agent" if controlled_archetype == "random" else "RL Agent (PPO)"
        color = color_map.get(controlled_archetype, "red")
        plt.plot(summary_rl["step"], summary_rl["mean_across_seeds"], label=label, color=color, lw=4, zorder=5)
        plt.fill_between(summary_rl["step"], summary_rl["mean_across_seeds"] - summary_rl["std_across_seeds"], summary_rl["mean_across_seeds"] + summary_rl["std_across_seeds"], color=color, alpha=0.2, zorder=4)

    ax.set_title(f"Performance Comparison: Controlled Agent vs. Heuristics ({strategy_name})")
    ax.set_ylabel("Normalized Accumulated Reward" if normalize else "Accumulated Reward")
    ax.set_xlabel("Step")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()

    if filename:
        save_plot(filename=filename, fig=fig)

    plt.show()

def plot_rl_metrics_distribution(df_rl, strategy_name="by_effort", filename=None):
    """
    Plot distribution of RL agent metrics (H-Index and Lifespan).

    Args:
        filename: Optional filename for saving the plot
    """
    if df_rl.empty:
        print("No controlled agent data to plot distribution.")
        return

    # Support both 'rl_agent' and 'random' archetypes
    controlled_archetypes = ["rl_agent", "random"]
    final_states = df_rl[df_rl["archetype"].isin(controlled_archetypes)].groupby("seed").last().reset_index()

    if final_states.empty:
        print("No final states found for controlled agent.")
        return

    controlled_archetype = final_states["archetype"].iloc[0]
    agent_label = "Random Agent" if controlled_archetype == "random" else "RL Agent"

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.boxplot(y=final_states["h_index"], ax=axes[0], color="skyblue")
    sns.stripplot(y=final_states["h_index"], ax=axes[0], color="black", alpha=0.5)
    axes[0].set_title(f"{agent_label} H-Index Distribution ({strategy_name})")
    axes[0].set_ylabel("H-Index")
    sns.boxplot(y=final_states["age"], ax=axes[1], color="lightgreen")
    sns.stripplot(y=final_states["age"], ax=axes[1], color="black", alpha=0.5)
    axes[1].set_title(f"{agent_label} Lifespan Distribution ({strategy_name})")
    axes[1].set_ylabel("Age (steps)")
    plt.tight_layout()

    if filename:
        save_plot(filename=filename, fig=fig)

    plt.show()

def plot_all_archetypes_distribution(df_traj, strategy_name="by_effort", filename=None, algorithm=""):
    """
    Plot H-Index and Lifespan distributions for all archetypes (heuristics + controlled agent).

    Args:
        df_traj: DataFrame with trajectories for all agents (from trajectories_*.parquet)
        strategy_name: Name of the reward strategy
        filename: Optional filename for saving the plot
    """
    if df_traj.empty:
        print("No trajectory data to plot.")
        return

    # Get final state for each agent across all seeds
    final_states = df_traj.groupby(["seed", "agent_id", "archetype"]).last().reset_index()

    if final_states.empty:
        print("No final states found.")
        return

    # Sort archetypes for consistent ordering
    archetypes = sorted(final_states["archetype"].unique())

    # Use thesis color scheme
    color_map = THESIS_COLORS.copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Add grid to both axes
    for ax in axes:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')

    # H-Index distribution by archetype
    sns.boxplot(data=final_states, x="archetype", y="h_index", ax=axes[0], hue="archetype",
                palette=color_map, order=archetypes, legend=False)
    sns.stripplot(data=final_states, x="archetype", y="h_index", ax=axes[0],
                  color="black", alpha=0.3, size=2, order=archetypes)
    axes[0].set_title(f"H-Index Distribution {algorithm} ({strategy_name})")
    axes[0].set_xlabel("Archetype")
    axes[0].set_ylabel("H-Index")
    axes[0].tick_params(axis='x', rotation=45)

    # Lifespan distribution by archetype
    sns.boxplot(data=final_states, x="archetype", y="age", ax=axes[1], hue="archetype",
                palette=color_map, order=archetypes, legend=False)
    sns.stripplot(data=final_states, x="archetype", y="age", ax=axes[1],
                  color="black", alpha=0.3, size=2, order=archetypes)
    axes[1].set_title(f"Lifespan Distribution {algorithm} ({strategy_name})")
    axes[1].set_xlabel("Archetype")
    axes[1].set_ylabel("Age (steps)")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if filename:
        save_plot(filename=filename, fig=fig)

    plt.show()

    # Print summary statistics
    print(f"\n--- Summary Statistics by Archetype ({strategy_name}) ---")
    for archetype in archetypes:
        arch_data = final_states[final_states["archetype"] == archetype]
        n_agents = len(arch_data)
        h_mean = arch_data["h_index"].mean()
        h_std = arch_data["h_index"].std()
        age_mean = arch_data["age"].mean()
        age_std = arch_data["age"].std()
        print(f"{archetype:20} | n={n_agents:5} | H-Index: {h_mean:5.2f} ± {h_std:4.2f} | Age: {age_mean:6.1f} ± {age_std:5.1f}")

def aggregate_empirical_results(summary_path, trajectory_path, strat="by_effort"):
    """
    Aggregate and visualize empirical results.

    Args:
        filename_performance: Optional filename for performance plot
        filename_distribution: Optional filename for distribution plot
    """
    df_h, df_rl_summary, df_rl_traj = get_path_for_strat(summary_path, trajectory_path)
    if df_h is None or df_rl_summary is None or df_rl_traj is None:
        return

    # 1. Plot Performance Curves
    plot_mean_reward_trajectories_with_rl(df_h, df_rl_summary, strategy_name=strat, filename=summary_path)

    # 2. Plot Distribution Metrics
    plot_rl_metrics_distribution(df_rl_traj, strategy_name=strat, filename=summary_path)

    # 3. Statistical Summary
    if df_rl_summary.empty:
        print(f"\nNo controlled agent data found for {strat}")
        return

    final_step = df_rl_summary["step"].max()
    rl_final = df_rl_summary[df_rl_summary["step"] == final_step]["mean_reward"].mean()
    h_final = df_h[df_h["step"] == final_step].groupby("archetype")["mean_reward"].mean()

    controlled_archetype = df_rl_summary["archetype"].iloc[0]
    agent_label = "Random Agent" if controlled_archetype == "random" else "RL Agent"

    print(f"\n--- Final Mean Reward (Step {final_step}) for {strat} ---")
    print(f"{agent_label}: {rl_final:.2f}")
    for arch, val in h_final.items():
        print(f"{arch:18}: {val:.2f}")

    if not h_final.empty:
        best_h_val = h_final.max()
        diff = ((rl_final / best_h_val) - 1) * 100
        print(f"\n{agent_label} is {diff:+.1f}% better than the best heuristic ({h_final.idxmax()}).")
    else:
        print(f"\n{agent_label} final reward: {rl_final:.2f} (no heuristics to compare)")

def calculate_overall_mean_return(summary_path, trajectory_path, aggregation_dir, prefix, strategy="by_effort"):
    """
    Calculate overall mean return with standard deviation from parquet files.

    Args:
        aggregation_dir: Directory containing the results (e.g., "random_agent")
        prefix: Prefix for the files (e.g., "random")
        strategy: Strategy name (e.g., "by_effort")

    Returns:
        dict: Dictionary with statistics for each archetype including:
              - mean_return: normalized mean return
              - std_return: standard deviation
              - n_agents: number of agents
              - raw_mean: unnormalized mean
              - raw_std: unnormalized std
    """
    import os
    import pathlib

    # Try multiple possible base paths
    possible_bases = [".", "..", "../.."]
    summary_file = None
    trajectories_file = None

    for base in possible_bases:
        test_summary = os.path.join(base, summary_path)
        test_traj = os.path.join(base, trajectory_path)

        if os.path.exists(test_summary) and os.path.exists(test_traj):
            summary_file = test_summary
            trajectories_file = test_traj
            break

    if summary_file is None or trajectories_file is None:
        print(f"Files not found for {strategy}")
        print(f"  Searched in: {', '.join(possible_bases)}")
        print(f"  Looking for pattern: {summary_path}")
        print(f"  Looking for pattern: {trajectory_path}")
        print(f"  Current directory: {os.getcwd()}")
        return None

    # Load data
    df_summary = pd.read_parquet(summary_file)
    df_traj = pd.read_parquet(trajectories_file)

    # Get final step for each agent
    final_states = df_traj.groupby(['seed', 'agent_id', 'archetype']).last().reset_index()

    # Calculate statistics by archetype
    results = {}

    for archetype in final_states['archetype'].unique():
        arch_data = final_states[final_states['archetype'] == archetype]

        # Get final accumulated rewards
        final_rewards = arch_data['accumulated_rewards'].values

        # Calculate raw statistics
        raw_mean = final_rewards.mean()
        raw_std = final_rewards.std()
        n_agents = len(final_rewards)

        # Normalize by the maximum reward across ALL archetypes for fair comparison
        max_reward = final_states['accumulated_rewards'].max()

        if max_reward > 0:
            normalized_rewards = final_rewards / max_reward
            norm_mean = normalized_rewards.mean()
            norm_std = normalized_rewards.std()
        else:
            norm_mean = 0.0
            norm_std = 0.0

        results[archetype] = {
            'mean_return': norm_mean,
            'std_return': norm_std,
            'n_agents': n_agents,
            'raw_mean': raw_mean,
            'raw_std': raw_std,
            'max_reward_used': max_reward
        }

    # Print summary
    print(f"\n{'='*80}")
    print(f"Overall Mean Return Statistics - {strategy}")
    print(f"{'='*80}")
    print(f"{'Archetype':<20} | {'Norm. Mean':<12} | {'Norm. Std':<12} | {'Raw Mean':<12} | {'Raw Std':<12} | {'N':<6}")
    print(f"{'-'*80}")

    for archetype in sorted(results.keys()):
        stats = results[archetype]
        print(f"{archetype:<20} | "
              f"{stats['mean_return']:>11.4f} | "
              f"{stats['std_return']:>11.4f} | "
              f"{stats['raw_mean']:>11.2f} | "
              f"{stats['raw_std']:>11.2f} | "
              f"{stats['n_agents']:>6}")

    print(f"\nNormalization factor (max reward across all archetypes): {results[list(results.keys())[0]]['max_reward_used']:.2f}")

    return results


def plot_mean_reward_trajectories_corrected(df_traj, strategy_name="by_effort", normalize=True, filename=None, algorithm=""):
    """
    Corrected version that properly handles survivor bias and uses consistent normalization.

    Args:
        df_traj: DataFrame with trajectories (from trajectories_*.parquet)
        strategy_name: Name of the reward strategy
        normalize: Whether to normalize rewards
        filename: Optional filename for saving the plot
    """
    if df_traj.empty:
        print("No trajectory data to plot.")
        return

    # DEBUG: Check number of seeds per archetype
    print(f"\n{'='*80}")
    print(f"DEBUG: Seeds per Archetype in df_traj")
    print(f"{'='*80}")
    for archetype in sorted(df_traj['archetype'].unique()):
        arch_seeds = df_traj[df_traj['archetype'] == archetype]['seed'].unique()
        n_agents = df_traj[df_traj['archetype'] == archetype].groupby('seed')['agent_id'].nunique()
        print(f"{archetype:<20} | Seeds: {len(arch_seeds):<5} | Agents/seed: {n_agents.mean():.1f} ± {n_agents.std():.1f}")
        if archetype in ['rl_agent', 'random']:
            print(f"  -> Seed list: {sorted(arch_seeds)}")

    # Step 1: Calculate cumulative rewards per agent (grouped by seed, agent_id, archetype)
    # Sort by step to ensure correct cumsum
    df_sorted = df_traj.sort_values(['seed', 'agent_id', 'step']).copy()

    # If 'accumulated_rewards' already exists, use it; otherwise calculate it
    if 'accumulated_rewards' not in df_sorted.columns:
        print("Warning: 'accumulated_rewards' column not found, cannot plot.")
        return

    # Step 2: Get max step in dataset
    max_step = df_sorted['step'].max()

    # Step 3: Forward-fill dead agents - vectorized approach
    # Create a complete grid of all (seed, agent_id, archetype, step) combinations
    all_agents = df_sorted[['seed', 'agent_id', 'archetype']].drop_duplicates()
    all_steps = pd.DataFrame({'step': range(int(max_step) + 1)})

    # Cross join to get all combinations
    all_agents['key'] = 0
    all_steps['key'] = 0
    full_grid = all_agents.merge(all_steps, on='key').drop('key', axis=1)

    # Merge with actual data
    df_expanded = full_grid.merge(
        df_sorted[['seed', 'agent_id', 'step', 'accumulated_rewards']],
        on=['seed', 'agent_id', 'step'],
        how='left'
    )

    # Forward-fill accumulated_rewards within each agent group
    df_expanded = df_expanded.sort_values(['seed', 'agent_id', 'step'])
    df_expanded['accumulated_rewards'] = df_expanded.groupby(['seed', 'agent_id'])['accumulated_rewards'].ffill()

    # Fill any remaining NaN with 0 (for agents that haven't started yet)
    df_expanded['accumulated_rewards'] = df_expanded['accumulated_rewards'].fillna(0)

    # Step 4: Calculate mean and std per archetype per step
    summary = df_expanded.groupby(['archetype', 'step'])['accumulated_rewards'].agg(['mean', 'std']).reset_index()

    # Step 5: Get normalization factor (same as in calculate_overall_mean_return)
    final_states = df_traj.groupby(['seed', 'agent_id', 'archetype']).last().reset_index()
    max_reward = final_states['accumulated_rewards'].max()

    # Step 6: Normalize if requested
    if normalize and max_reward > 0:
        summary['mean'] = summary['mean'] / max_reward
        summary['std'] = summary['std'] / max_reward

    # Step 7: Plot
    archetypes = sorted(summary['archetype'].unique())

    # Use thesis color scheme
    color_map = THESIS_COLORS.copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    for archetype in archetypes:
        arch_data = summary[summary['archetype'] == archetype]
        color = color_map.get(archetype, "gray")

        # Determine line properties
        if archetype in ["rl_agent", "random"]:
            lw = 4
            zorder = 5
            alpha = 1.0
            label = "Random Agent" if archetype == "random" else "RL Agent (PPO)"
        else:
            lw = 2
            zorder = 3
            alpha = 0.7
            label = archetype

        plt.plot(arch_data['step'], arch_data['mean'], label=label, color=color, lw=lw, zorder=zorder, alpha=alpha)

        # Add std band for controlled agents
        if archetype in ["rl_agent", "random"] and not arch_data['std'].isna().all():
            plt.fill_between(
                arch_data['step'],
                arch_data['mean'] - arch_data['std'],
                arch_data['mean'] + arch_data['std'],
                color=color, alpha=0.2, zorder=zorder-1
            )

    ax.set_title(f"Performance Comparison: {algorithm} - {strategy_name}")
    ax.set_ylabel("Normalized Accumulated Reward" if normalize else "Accumulated Reward")
    ax.set_xlabel("Step")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()

    if filename:
        save_plot(filename=filename, fig=fig)

    plt.show()

    # Step 8: Debug comparison - print final values
    print(f"\n{'='*80}")
    print(f"Debug: Final Plotted Values vs. Overall Mean Return - {strategy_name}")
    print(f"{'='*80}")
    print(f"{'Archetype':<20} | {'Final Plot Value':<18} | {'Mean Return (Stat)':<20}")
    print(f"{'-'*80}")

    # Calculate overall mean return for comparison
    final_plot_values = {}
    for archetype in archetypes:
        arch_data = summary[summary['archetype'] == archetype]
        final_val = arch_data[arch_data['step'] == max_step]['mean'].values
        if len(final_val) > 0:
            final_plot_values[archetype] = final_val[0]
        else:
            final_plot_values[archetype] = None

    # Get mean return from final states
    mean_returns = {}
    for archetype in archetypes:
        arch_finals = final_states[final_states['archetype'] == archetype]['accumulated_rewards']
        if len(arch_finals) > 0:
            if normalize and max_reward > 0:
                mean_returns[archetype] = arch_finals.mean() / max_reward
            else:
                mean_returns[archetype] = arch_finals.mean()
        else:
            mean_returns[archetype] = None

    for archetype in sorted(archetypes):
        plot_val = final_plot_values.get(archetype, None)
        stat_val = mean_returns.get(archetype, None)

        if plot_val is not None and stat_val is not None:
            plot_str = f"{plot_val:.6f}"
            stat_str = f"{stat_val:.6f}"
        else:
            plot_str = "N/A"
            stat_str = "N/A"

        print(f"{archetype:<20} | {plot_str:<18} | {stat_str:<20}")

    print(f"\nNormalization factor: {max_reward:.2f}")
    print("Note: These should match if survivor bias is properly corrected.")


def visualize_policy_population(simulation_steps, filename=None):
    """
    Visualize population composition over time.

    Args:
        filename: Optional filename for saving the plot
    """
    records = []
    for step_idx, step in enumerate(simulation_steps):
        for agent_id, agent in step.items():
            if agent is not None:
                records.append({"step": step_idx, "archetype": agent.get("archetype", "rl_agent")})
    df = pd.DataFrame(records)
    counts = df.groupby(["step", "archetype"]).size().reset_index(name="count")
    pivot = counts.pivot(index="step", columns="archetype", values="count").fillna(0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    pivot.plot.line(ax=axes[0])
    axes[0].set_title("Population Composition Over Time")
    axes[0].set_ylabel("Agent Count")
    axes[0].legend(title="Archetype", bbox_to_anchor=(1.05, 1), loc="upper left")
    pivot.sum(axis=1).plot(ax=axes[1], color="black")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Total Agents")
    plt.tight_layout()

    if filename:
        save_plot(filename=filename, fig=fig)

    plt.show()

def plot_archetype_vs_controlled_agent(
    df_traj,
    strategy_name="by_effort",
    algorithm="PPO",
    normalize=True,
    normalization_factor=None,
    figsize=(14, 8),
    missing_mode="ffill",
    band="std",
    band_for_controlled_only=True,   # NEW
    clip_band_at_zero=False,
    std_ddof=1,
    filename=None
):
    """
    Plot fixed-policy archetypes against the controlled RL/random agent.

    Style is aligned with plot_agent_comparison:
    - same figsize default
    - same linewidth
    - same band logic
    - same axis labels style
    - same legend placement
    - same missing-value handling for accumulated rewards
    """

    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Allow both DataFrame and parquet path
    if isinstance(df_traj, (str, os.PathLike)):
        df_traj = pd.read_parquet(df_traj)

    if df_traj.empty:
        raise ValueError("df_traj is empty.")

    required_cols = {"seed", "agent_id", "archetype", "step", "accumulated_rewards"}
    missing_cols = required_cols - set(df_traj.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    colors = {
        "careerist": "#2E5090",
        "mass_producer": "#D2691E",
        "orthodox_scientist": "#228B22",
        "rl_agent": "#e74c3c",
        "random": "#e74c3c",
        "unknown": "#696969"
    }

    label_map = {
        "careerist": "Careerist",
        "mass_producer": "Mass Producer",
        "orthodox_scientist": "Orthodox Scientist",
        "rl_agent": f"RL Agent ({algorithm})",
        "random": "Random Agent",
        "unknown": "Unknown"
    }

    df = df_traj.copy()
    df = df.sort_values(["seed", "agent_id", "step"])

    df["__unit"] = (
        df["seed"].astype(str)
        + "::"
        + df["agent_id"].astype(str)
    )

    all_steps = list(range(int(df["step"].min()), int(df["step"].max()) + 1))

    # ------------------------------------------------------------------
    # Compute normalization factor
    # ------------------------------------------------------------------
    if normalize:
        if normalization_factor is None:
            normalization_factor = df["accumulated_rewards"].max()

        if normalization_factor <= 0:
            print("Warning: normalization_factor <= 0. Normalization disabled.")
            normalization_factor = 1.0
            normalize = False

        print(f"Global max reward after filtering: {normalization_factor:.6f}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    summary_rows = []
    stats = {}

    archetype_order = [
        "careerist",
        "mass_producer",
        "orthodox_scientist",
        "random",
        "rl_agent"
    ]

    present_archetypes = [
        a for a in archetype_order
        if a in df["archetype"].unique()
    ]

    # Add any unexpected archetypes at the end
    extra_archetypes = [
        a for a in sorted(df["archetype"].unique())
        if a not in present_archetypes
    ]
    present_archetypes.extend(extra_archetypes)

    for archetype in present_archetypes:
        arch_df = df[df["archetype"] == archetype].copy()

        if arch_df.empty:
            continue

        n_units = arch_df["__unit"].nunique()
        print(f"{archetype}: {n_units} evaluated units")

        # Select unique (step, __unit) combinations
        # Note: There should be no duplicates, but if they exist, take the last value
        grouped = (
            arch_df
            .groupby(["step", "__unit"], as_index=False)["accumulated_rewards"]
            .last()
        )

        reward_matrix = grouped.pivot(
            index="step",
            columns="__unit",
            values="accumulated_rewards"
        )

        reward_matrix = reward_matrix.reindex(all_steps)

        if missing_mode == "ffill":
            # Correct for accumulated rewards:
            # after an agent disappears, keep its last known accumulated reward.
            reward_matrix = reward_matrix.ffill().fillna(0.0)

        elif missing_mode == "zero":
            # More punitive; can distort accumulated rewards after death/removal.
            reward_matrix = reward_matrix.fillna(0.0)

        elif missing_mode == "drop":
            # Survivor-only behavior; can introduce survivor bias.
            pass

        else:
            raise ValueError("missing_mode must be 'ffill', 'zero', or 'drop'.")

        if normalize:
            reward_matrix = reward_matrix / normalization_factor

        mean = reward_matrix.mean(axis=1, skipna=True)

        if n_units > 1:
            std = reward_matrix.std(axis=1, skipna=True, ddof=std_ddof)
        else:
            std = pd.Series(0.0, index=reward_matrix.index)

        count = reward_matrix.count(axis=1)

        color = colors.get(archetype, "#95a5a6")
        label = label_map.get(archetype, archetype)

        is_controlled_agent = archetype in {"rl_agent", "random"}

        ax.plot(
            reward_matrix.index,
            mean,
            label=label,
            linewidth=2.5,
            color=color
        )

        # Draw uncertainty band with same logic as plot_agent_comparison
        draw_band = band is not None and (
                not band_for_controlled_only or is_controlled_agent
        )

        if draw_band:
            if band == "std":
                lower = mean - std
                upper = mean + std

            elif band == "sem":
                sem = std / np.sqrt(count.clip(lower=1))
                lower = mean - sem
                upper = mean + sem

            elif band == "ci95":
                sem = std / np.sqrt(count.clip(lower=1))
                lower = mean - 1.96 * sem
                upper = mean + 1.96 * sem

            elif band == "quantile":
                lower = reward_matrix.quantile(0.25, axis=1)
                upper = reward_matrix.quantile(0.75, axis=1)

            else:
                raise ValueError("band must be one of: 'std', 'sem', 'ci95', 'quantile', or None.")

            if clip_band_at_zero:
                lower = lower.clip(lower=0)

            ax.fill_between(
                reward_matrix.index,
                lower,
                upper,
                alpha=0.2,
                color=color
            )

        for step in reward_matrix.index:
            summary_rows.append({
                "archetype": archetype,
                "step": step,
                "mean": mean.loc[step],
                "std": std.loc[step],
                "count": int(count.loc[step])
            })

        final_step = reward_matrix.index[-1]

        stats[archetype] = {
            "final_step": final_step,
            "final_mean": float(mean.iloc[-1]),
            "final_std": float(std.iloc[-1]),
            "n_units": int(n_units)
        }

    summary = pd.DataFrame(summary_rows)

    ax.set_xlabel("Simulation Step", fontsize=14, fontweight="bold")
    ylabel = "Normalized Mean Accumulated Reward" if normalize else "Mean Accumulated Reward"
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
    ax.set_title("Agent Performance Comparison", fontsize=16, fontweight="bold", pad=20)
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if filename is not None:
        # Save as PNG and PDF
        filepath_png = filename.replace(".png", "").replace(".pdf", "") + ".png"
        filepath_pdf = filename.replace(".png", "").replace(".pdf", "") + ".pdf"
        fig.savefig(
            filepath_png,
            dpi=600,
            bbox_inches="tight",
            facecolor=fig.get_facecolor()
        )
        fig.savefig(
            filepath_pdf,
            bbox_inches="tight",
            facecolor=fig.get_facecolor()
        )
        print(f"Saved plot to: {filepath_png}")
        print(f"Saved plot to: {filepath_pdf}")

    print("\n" + "=" * 90)
    correction_text = f"with missing_mode='{missing_mode}'"
    print(f"Performance Statistics ({correction_text})")
    print("=" * 90)

    for archetype, s in stats.items():
        label = label_map.get(archetype, archetype)
        print(
            f"{label:<24} | "
            f"Mean: {s['final_mean']:.6f} | "
            f"Std: {s['final_std']:.6f} | "
            f"N: {s['n_units']}"
        )

    if normalize:
        print(f"\nNormalization factor: {normalization_factor:.6f}")

    return fig, summary, normalization_factor


def plot_archetype_vs_controlled_agent_from_summary(
    df_summary,
    algorithm="PPO",
    normalize=True,
    normalization_factor=None,
    figsize=(14, 8),
    band="std",
    band_for_controlled_only=True,
    clip_band_at_zero=False,
    filename=None
):
    """
    Plot fixed-policy archetypes against the controlled RL/random agent using pre-aggregated summary data.

    This function is optimized for summary files that contain:
    - step, archetype, mean_reward, std_reward, n_agents, seed

    Unlike plot_archetype_vs_controlled_agent which works with raw trajectories,
    this function uses pre-computed statistics from summary files.

    Parameters
    ----------
    df_summary : DataFrame or path
        Summary DataFrame or path to parquet file with columns:
        ['step', 'archetype', 'mean_reward', 'std_reward', 'n_agents', 'seed']
    algorithm : str
        Name of algorithm (e.g., "PPO", "APPO", "Random") for labeling
    normalize : bool
        Whether to normalize rewards by global max
    normalization_factor : float, optional
        Specific normalization factor. If None, computed from data.
    figsize : tuple
        Figure size
    band : str
        Type of uncertainty band: "std", "sem", "ci95", or None
    band_for_controlled_only : bool
        If True, only show bands for controlled agent (rl_agent/random)
    clip_band_at_zero : bool
        If True, clip lower band at 0
    filename : str, optional
        Path to save figure (both .png and .pdf)

    Returns
    -------
    fig : matplotlib.figure.Figure
    normalization_factor : float
    """
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Allow both DataFrame and parquet path
    if isinstance(df_summary, (str, os.PathLike)):
        df_summary = pd.read_parquet(df_summary)

    if df_summary.empty:
        raise ValueError("df_summary is empty.")

    required_cols = {"step", "archetype", "mean_reward", "std_reward", "n_agents", "seed"}
    missing_cols = required_cols - set(df_summary.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    colors = {
        "careerist": "#2E5090",
        "mass_producer": "#D2691E",
        "orthodox_scientist": "#228B22",
        "rl_agent": "#e74c3c",
        "random": "#e74c3c",
        "unknown": "#696969"
    }

    label_map = {
        "careerist": "Careerist",
        "mass_producer": "Mass Producer",
        "orthodox_scientist": "Orthodox Scientist",
        "rl_agent": f"RL Agent ({algorithm})",
        "random": "Random Agent",
        "unknown": "Unknown"
    }

    df = df_summary.copy()

    # Compute normalization factor
    if normalize:
        if normalization_factor is None:
            normalization_factor = df["mean_reward"].max()

        if normalization_factor <= 0:
            print("Warning: normalization_factor <= 0. Normalization disabled.")
            normalization_factor = 1.0
            normalize = False

        print(f"Global max reward in summary: {normalization_factor:.6f}")

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    archetype_order = [
        "careerist",
        "mass_producer",
        "orthodox_scientist",
        "random",
        "rl_agent"
    ]

    present_archetypes = [
        a for a in archetype_order
        if a in df["archetype"].unique()
    ]

    # Add any unexpected archetypes at the end
    extra_archetypes = [
        a for a in sorted(df["archetype"].unique())
        if a not in present_archetypes
    ]
    present_archetypes.extend(extra_archetypes)

    stats = {}

    for archetype in present_archetypes:
        arch_df = df[df["archetype"] == archetype].copy()

        if arch_df.empty:
            continue

        all_seeds = arch_df["seed"].unique()
        n_seeds = len(all_seeds)
        print(f"{archetype}: {n_seeds} seeds")

        # Get max step across all data (not just this archetype)
        max_step = df["step"].max()
        all_steps = list(range(0, max_step + 1))

        # Build matrices: rows = steps, columns = seeds
        # Apply ffill per seed to handle missing data
        seed_rewards = {}
        seed_n_agents = {}

        for seed in all_seeds:
            seed_df = arch_df[arch_df["seed"] == seed].copy()
            seed_df = seed_df.set_index("step").sort_index()

            # Reindex to all steps and forward-fill both reward and n_agents
            seed_series = seed_df["mean_reward"].reindex(all_steps)
            seed_series = seed_series.ffill().fillna(0.0)
            seed_rewards[seed] = seed_series

            n_agents_series = seed_df["n_agents"].reindex(all_steps)
            n_agents_series = n_agents_series.ffill().fillna(0.0)
            seed_n_agents[seed] = n_agents_series

        # Convert to DataFrames for easy aggregation
        reward_df = pd.DataFrame(seed_rewards)
        n_agents_df = pd.DataFrame(seed_n_agents)

        if normalize:
            reward_df = reward_df / normalization_factor

        # Aggregate across seeds using weighted mean
        # Weight each seed's mean_reward by its n_agents
        total_agents = n_agents_df.sum(axis=1)
        weighted_sum = (reward_df * n_agents_df).sum(axis=1)

        # Avoid division by zero
        mean = weighted_sum / total_agents.replace(0, np.nan)
        mean = mean.fillna(0.0)

        # For std, compute weighted standard deviation
        if n_seeds > 1:
            # Weighted variance formula: Σ(w_i * (x_i - μ)^2) / Σ(w_i)
            deviations_sq = (reward_df.subtract(mean, axis=0)) ** 2
            weighted_var = (deviations_sq * n_agents_df).sum(axis=1) / total_agents.replace(0, np.nan)
            std = np.sqrt(weighted_var).fillna(0.0)
        else:
            std = pd.Series(0.0, index=reward_df.index)

        is_controlled_agent = archetype in {"rl_agent", "random"}

        if archetype in {"rl_agent", "random"}:
            label = f"RL Agent ({algorithm})" if archetype == "rl_agent" else "Random Agent"
        else:
            label = archetype.replace("_", " ").title()

        color = colors.get(archetype, "#95a5a6")

        ax.plot(
            mean.index,
            mean,
            label=label,
            linewidth=2.5,
            color=color
        )

        # Draw uncertainty band
        draw_band = band is not None and (
            not band_for_controlled_only or is_controlled_agent
        )

        if draw_band:
            if band == "std":
                lower = mean - std
                upper = mean + std

            elif band == "sem":
                # SEM = std / sqrt(n_seeds)
                sem = std / np.sqrt(n_seeds)
                lower = mean - sem
                upper = mean + sem

            elif band == "ci95":
                sem = std / np.sqrt(n_seeds)
                lower = mean - 1.96 * sem
                upper = mean + 1.96 * sem

            else:
                raise ValueError("band must be one of: 'std', 'sem', 'ci95', or None.")

            if clip_band_at_zero:
                lower = lower.clip(lower=0)

            ax.fill_between(
                mean.index,
                lower,
                upper,
                alpha=0.2,
                color=color
            )

        final_step = mean.index[-1]

        stats[archetype] = {
            "final_step": final_step,
            "final_mean": float(mean.iloc[-1]),
            "final_std": float(std.iloc[-1]),
            "n_seeds": n_seeds
        }

    ax.set_xlabel("", fontsize=14, fontweight="bold")
    ylabel = "Normalized Mean Accumulated Reward" if normalize else "Mean Accumulated Reward"
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
    ax.set_title("Agent Performance Comparison", fontsize=16, fontweight="bold", pad=20)
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if filename is not None:
        # Save as PNG and PDF
        filepath_png = filename.replace(".png", "").replace(".pdf", "") + ".png"
        filepath_pdf = filename.replace(".png", "").replace(".pdf", "") + ".pdf"
        fig.savefig(
            filepath_png,
            dpi=600,
            bbox_inches="tight",
            facecolor=fig.get_facecolor()
        )
        fig.savefig(
            filepath_pdf,
            bbox_inches="tight",
            facecolor=fig.get_facecolor()
        )
        print(f"Saved plot to: {filepath_png}")
        print(f"Saved plot to: {filepath_pdf}")

    print("\n" + "=" * 90)
    print("Performance Statistics (from summary)")
    print("=" * 90)

    for archetype, s in stats.items():
        label = label_map.get(archetype, archetype)
        print(
            f"{label:<24} | "
            f"Mean: {s['final_mean']:.6f} | "
            f"Std: {s['final_std']:.6f} | "
            f"Seeds: {s['n_seeds']}"
        )

    if normalize:
        print(f"\nNormalization factor: {normalization_factor:.6f}")

    return fig, normalization_factor


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_archetype_vs_controlled_agent_subplots(
    trajectory_dict,
    strategy_name="by_effort",
    normalize=True,
    normalization_factor=None,
    figsize=(24, 7),
    missing_mode="ffill",   # "ffill", "zero", or "drop"
    band="ci95",             # "std", "ci95", "sem", "quantile", or None
    band_for_controlled_only=True,
    clip_band_at_zero=False,
    std_ddof=1,
    sharey=True,
    filename=None
):
    """
    Plot multiple 'archetype vs controlled agent' comparisons side by side.

    Parameters
    ----------
    trajectory_dict : dict
        Example:
        {
            "PPO": ppo_trajectory_df,
            "APPO": appo_trajectory_df,
            "Random": random_trajectory_df
        }

    Returns
    -------
    fig : matplotlib.figure.Figure
    all_summaries : dict
        Summary dataframe per subplot.
    normalization_factor : float
        Shared normalization factor used across all subplots.
    """

    colors = {
        "careerist": "#2E5090",
        "mass_producer": "#D2691E",
        "orthodox_scientist": "#228B22",
        "rl_agent": "#e74c3c",
        "random": "#e74c3c",
        "unknown": "#696969"
    }

    def _load_if_needed(df_or_path):
        if isinstance(df_or_path, (str, os.PathLike)):
            return pd.read_parquet(df_or_path)
        return df_or_path.copy()

    # ------------------------------------------------------------------
    # Load all data first
    # ------------------------------------------------------------------
    prepared = {}
    for algo_name, df_or_path in trajectory_dict.items():
        df = _load_if_needed(df_or_path)

        if df.empty:
            raise ValueError(f"{algo_name}: dataframe is empty.")

        required_cols = {"seed", "agent_id", "archetype", "step", "accumulated_rewards"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"{algo_name}: missing required columns: {missing_cols}")

        df = df.sort_values(["seed", "agent_id", "step"]).copy()
        df["__unit"] = df["seed"].astype(str) + "::" + df["agent_id"].astype(str)

        prepared[algo_name] = df

    if len(prepared) == 0:
        raise ValueError("No valid data provided.")

    # ------------------------------------------------------------------
    # Shared normalization factor across all subplots
    # ------------------------------------------------------------------
    if normalize:
        if normalization_factor is None:
            normalization_factor = max(
                df["accumulated_rewards"].max()
                for df in prepared.values()
            )

        if normalization_factor <= 0:
            print("Warning: normalization_factor <= 0. Normalization disabled.")
            normalization_factor = 1.0
            normalize = False

        print(f"Shared normalization factor: {normalization_factor:.6f}")
    else:
        normalization_factor = None

    # ------------------------------------------------------------------
    # Create subplot figure
    # ------------------------------------------------------------------
    n_plots = len(prepared)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize, sharey=sharey)

    if n_plots == 1:
        axes = [axes]

    all_summaries = {}

    archetype_order = [
        "careerist",
        "mass_producer",
        "orthodox_scientist",
        "random",
        "rl_agent"
    ]

    for idx, (algo_name, df) in enumerate(prepared.items()):
        ax = axes[idx]
        all_steps = list(range(int(df["step"].min()), int(df["step"].max()) + 1))
        summary_rows = []

        present_archetypes = [
            a for a in archetype_order
            if a in df["archetype"].unique()
        ]

        extra_archetypes = [
            a for a in sorted(df["archetype"].unique())
            if a not in present_archetypes
        ]
        present_archetypes.extend(extra_archetypes)

        for archetype in present_archetypes:
            arch_df = df[df["archetype"] == archetype].copy()
            if arch_df.empty:
                continue

            # Select unique (step, __unit) combinations
            # If duplicates exist, take the last value
            grouped = (
                arch_df
                .groupby(["step", "__unit"], as_index=False)["accumulated_rewards"]
                .last()
            )

            reward_matrix = grouped.pivot(
                index="step",
                columns="__unit",
                values="accumulated_rewards"
            )

            reward_matrix = reward_matrix.reindex(all_steps)

            if missing_mode == "ffill":
                reward_matrix = reward_matrix.ffill().fillna(0.0)
            elif missing_mode == "zero":
                reward_matrix = reward_matrix.fillna(0.0)
            elif missing_mode == "drop":
                pass
            else:
                raise ValueError("missing_mode must be 'ffill', 'zero', or 'drop'.")

            if normalize:
                reward_matrix = reward_matrix / normalization_factor

            mean = reward_matrix.mean(axis=1, skipna=True)

            n_units = reward_matrix.shape[1]
            if n_units > 1:
                std = reward_matrix.std(axis=1, skipna=True, ddof=std_ddof)
            else:
                std = pd.Series(0.0, index=reward_matrix.index)

            count = reward_matrix.count(axis=1)

            is_controlled_agent = archetype in {"rl_agent", "random"}

            if archetype in {"rl_agent", "random"}:
                label = "Controlled Agent"
            else:
                label = archetype.replace("_", " ").title()

            color = colors.get(archetype, "#95a5a6")

            ax.plot(
                reward_matrix.index,
                mean,
                label=label,
                linewidth=2.5,
                color=color
            )

            draw_band = band is not None and (
                not band_for_controlled_only or is_controlled_agent
            )

            if draw_band:
                if band == "std":
                    lower = mean - std
                    upper = mean + std

                elif band == "sem":
                    sem = std / np.sqrt(count.clip(lower=1))
                    lower = mean - sem
                    upper = mean + sem

                elif band == "ci95":
                    sem = std / np.sqrt(count.clip(lower=1))
                    lower = mean - 1.96 * sem
                    upper = mean + 1.96 * sem

                elif band == "quantile":
                    lower = reward_matrix.quantile(0.25, axis=1)
                    upper = reward_matrix.quantile(0.75, axis=1)

                else:
                    raise ValueError("band must be one of: 'std', 'sem', 'ci95', 'quantile', or None.")

                if clip_band_at_zero:
                    lower = lower.clip(lower=0)

                ax.fill_between(
                    reward_matrix.index,
                    lower,
                    upper,
                    alpha=0.2,
                    color=color
                )

            for step in reward_matrix.index:
                summary_rows.append({
                    "algorithm": algo_name,
                    "archetype": archetype,
                    "step": step,
                    "mean": mean.loc[step],
                    "std": std.loc[step],
                    "count": int(count.loc[step])
                })

        summary_df = pd.DataFrame(summary_rows)
        all_summaries[algo_name] = summary_df

        ax.set_title(algo_name, fontsize=18, fontweight="bold", pad=20)
        ax.set_xlabel("Simulation Step", fontsize=16, fontweight="bold")

        if idx == 0:
            ylabel = "Normalized Mean Accumulated Reward" if normalize else "Mittlerer kumulierter Reward"
            ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")
        else:
            ax.set_ylabel("")

        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=14)

    # ------------------------------------------------------------------
    # Shared legend
    # ------------------------------------------------------------------
    handles_labels = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            handles_labels[label] = handle

    fig.legend(
        handles_labels.values(),
        handles_labels.keys(),
        loc="lower center",
        ncol=min(len(handles_labels), 5),
        fontsize=14,
        bbox_to_anchor=(0.5, -0.02)
    )

    fig.suptitle(
        "Controlled Agent vs. Archetype Baselines",
        fontsize=20,
        fontweight="bold",
        y=1.02
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if filename is not None:
        # Save as PNG and PDF
        filepath_png = filename.replace(".png", "").replace(".pdf", "") + ".png"
        filepath_pdf = filename.replace(".png", "").replace(".pdf", "") + ".pdf"
        fig.savefig(
            filepath_png,
            dpi=600,
            bbox_inches="tight",
            facecolor=fig.get_facecolor()
        )
        fig.savefig(
            filepath_pdf,
            bbox_inches="tight",
            facecolor=fig.get_facecolor()
        )
        print(f"Saved plot to: {filepath_png}")
        print(f"Saved plot to: {filepath_pdf}")

    return fig, all_summaries, normalization_factor