from collections import Counter, defaultdict
from src.env.area import Area
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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