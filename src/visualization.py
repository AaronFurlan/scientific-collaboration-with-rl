from collections import Counter, defaultdict
from src.env.area import Area
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import pandas as pd

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

def topic_area_per_project(projects, actions, area_pickle_file):
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

    # Custom visualization to highlight RL agent
    print("\n--- Publications per Archetype ---")
    arch_counts = Counter([p[2] for p in papers])
    for arch in ["careerist", "orthodox_scientist", "mass_producer", "rl_agent"]:
        print(f"{arch:18}: {arch_counts.get(arch, 0)}")

    visualize_knowledge_space(area, sampled_points=papers)


def visualize_knowledge_space(area, resolution=200, sampled_points=None, bounds=None):
    """
    Modified version of Area.visualize to better highlight the RL agent.
    """
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

    plt.figure(figsize=(12, 10))
    plt.imshow(
        Z,
        extent=(xmin, xmax, ymin, ymax),
        origin="lower",
        cmap="bwr",
        vmin=-1.0,
        vmax=1.0,
        alpha=0.3,  # Lighter background to make points pop
    )
    plt.colorbar(label="Scientific Value")

    if sampled_points is not None and len(sampled_points) > 0:
        has_category = len(sampled_points[0]) == 3

        if has_category:
            category_points = defaultdict(list)
            for px, py, cat in sampled_points:
                if xmin <= px <= xmax and ymin <= py <= ymax:
                    category_points[cat].append((px, py))

            # Definierte Farben für bekannte Archetypen
            colors = {
                "careerist": "blue",
                "orthodox_scientist": "green",
                "mass_producer": "orange",
                "rl_agent": "red",
                "unknown": "gray"
            }

            # Erst alle anderen plotten
            for cat in ["careerist", "orthodox_scientist", "mass_producer", "unknown"]:
                if cat not in category_points:
                    if cat != "unknown":
                        print(f"Note: No papers found for archetype '{cat}' in this simulation.")
                    continue

                pts = category_points[cat]
                xs, ys = zip(*pts)
                plt.scatter(
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
                plt.scatter(
                    xs, ys,
                    c="red",
                    s=80,
                    alpha=1.0,
                    edgecolors="black",
                    linewidth=1.5,
                    marker="*",
                    label="RL Agent (PPO)"
                )
        else:
            px, py = zip(*[(px, py) for (px, py) in sampled_points if xmin <= px <= xmax and ymin <= py <= ymax])
            plt.scatter(px, py, c="black", s=10, edgecolors="white", label="Papers")

    plt.legend(loc="upper left", bbox_to_anchor=(1.15, 1))
    plt.title("Knowledge Space with Published Papers")
    plt.xlabel("Topic Dimension 1")
    plt.ylabel("Topic Dimension 2")
    plt.tight_layout()
    plt.show()

def animate_knowledge_space(projects, actions, area_pickle_file, interval=200, steps_per_frame=5):
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

    colors = {"careerist": "blue", "orthodox_scientist": "green", "mass_producer": "orange", "rl_agent": "red",
              "unknown": "gray"}
    scatters = {}
    for arch, color in colors.items():
        if arch == "rl_agent":
            scatters[arch] = ax.scatter([], [], c=color, s=100, marker="*", edgecolors="black", label=arch, zorder=5)
        else:
            scatters[arch] = ax.scatter([], [], c=color, s=20, alpha=0.6, edgecolors="white", label=arch, zorder=3)

    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    title = ax.set_title("Knowledge Space Evolution - Step 0")

    max_time = max([p["time"] for p in paper_data]) if paper_data else 100
    frames = range(0, int(max_time) + 1, steps_per_frame)

    def update(frame):
        current_papers = [p for p in paper_data if p["time"] <= frame]

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
        return list(scatters.values()) + [title]

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
    plt.close()  # Verhindert doppelte Anzeige im Notebook
    return ani

def plot_histograms(projects, actions):
    contributors_per_project(projects)
    success_rate_over_time(projects)
    quality_metrics_over_time(projects)
    collab_metrics_over_time(projects)
    projects_per_contributor(projects)
    completion_per_project(projects)
    quality_per_project(projects)

def plot_mean_reward_trajectories_with_rl(dfs_heuristic, df_rl, strategy_name="by_effort", normalize=True):
    df_h, df_r = dfs_heuristic.copy(), df_rl.copy()
    all_archetypes = sorted(list(df_h["archetype"].unique()))
    palette = sns.color_palette("tab10", len(all_archetypes) + 1)
    color_map = {a: palette[i] for i, a in enumerate(all_archetypes)}
    color_map["rl_agent"] = "red"
    plt.figure(figsize=(12, 7))
    if normalize:
        max_val = max(df_h["mean_reward"].max(), df_r["mean_reward"].max())
        if max_val > 0:
            df_h["mean_reward"] /= max_val
            df_r["mean_reward"] /= max_val
    for archetype, group_arch in df_h.groupby("archetype"):
        summary_h = group_arch.groupby("step")["mean_reward"].mean().reset_index()
        plt.plot(summary_h["step"], summary_h["mean_reward"], label=archetype, color=color_map[archetype], lw=2, alpha=0.7)
    summary_rl = df_r.groupby("step").agg(mean_across_seeds=("mean_reward", "mean"), std_across_seeds=("mean_reward", "std")).reset_index()
    plt.plot(summary_rl["step"], summary_rl["mean_across_seeds"], label="rl_agent (PPO)", color=color_map["rl_agent"], lw=4, zorder=5)
    plt.fill_between(summary_rl["step"], summary_rl["mean_across_seeds"] - summary_rl["std_across_seeds"], summary_rl["mean_across_seeds"] + summary_rl["std_across_seeds"], color=color_map["rl_agent"], alpha=0.2, zorder=4)
    plt.title(f"Performance Comparison: RL_Agent vs. Heuristics ({strategy_name})")
    plt.ylabel("Normalized Accumulated Reward" if normalize else "Accumulated Reward")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()

def plot_rl_metrics_distribution(df_rl, strategy_name="by_effort"):
    final_states = df_rl[df_rl["archetype"] == "rl_agent"].groupby("seed").last().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.boxplot(y=final_states["h_index"], ax=axes[0], color="skyblue")
    sns.stripplot(y=final_states["h_index"], ax=axes[0], color="black", alpha=0.5)
    axes[0].set_title(f"RL Agent H-Index Distribution ({strategy_name})")
    sns.boxplot(y=final_states["age"], ax=axes[1], color="lightgreen")
    sns.stripplot(y=final_states["age"], ax=axes[1], color="black", alpha=0.5)
    axes[1].set_title(f"RL Agent Lifespan Distribution ({strategy_name})")
    plt.tight_layout()
    plt.show()

def aggregate_empirical_results(strat):
    df_h, df_rl_summary, df_rl_traj = get_path_for_strat(strat)
    if df_h is None or df_rl_summary is None or df_rl_traj is None:
        return

    # 1. Plot Performance Curves
    plot_mean_reward_trajectories_with_rl(df_h, df_rl_summary, strategy_name=strat)

    # 2. Plot Distribution Metrics
    plot_rl_metrics_distribution(df_rl_traj, strategy_name=strat)

    # 3. Statistical Summary
    final_step = df_rl_summary["step"].max()
    rl_final = df_rl_summary[df_rl_summary["step"] == final_step]["mean_reward"].mean()
    h_final = df_h[df_h["step"] == final_step].groupby("archetype")["mean_reward"].mean()

    print(f"\n--- Final Mean Reward (Step {final_step}) for {strat} ---")
    print(f"RL_Agent: {rl_final:.2f}")
    for arch, val in h_final.items():
        print(f"{arch:18}: {val:.2f}")

    best_h_val = h_final.max()
    diff = ((rl_final / best_h_val) - 1) * 100
    print(f"\nRL Agent is {diff:+.1f}% better than the best heuristic ({h_final.idxmax()}).")

def visualize_policy_population(simulation_steps):
    records = []
    for step_idx, step in enumerate(simulation_steps):
        for agent_id, agent in step.items():
            if agent is not None:
                records.append({"step": step_idx, "archetype": agent.get("archetype", "rl_agent")})
    df = pd.DataFrame(records)
    counts = df.groupby(["step", "archetype"]).size().reset_index(name="count")
    pivot = counts.pivot(index="step", columns="archetype", values="count").fillna(0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    pivot.plot.line(ax=axes[0])
    axes[0].set_title("Population Composition Over Time")
    axes[0].legend(title="Archetype", bbox_to_anchor=(1.05, 1), loc="upper left")
    pivot.sum(axis=1).plot(ax=axes[1], color="black")
    axes[1].set_ylabel("Total Agents")
    plt.tight_layout()
    plt.show()