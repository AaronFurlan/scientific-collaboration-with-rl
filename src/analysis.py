from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def contributors_per_project(projects):
    num_contributors = [len(p["contributors"]) for p in projects]
    plt.figure(figsize=(8, 5))
    sns.histplot(num_contributors, bins=range(1, max(num_contributors) + 2), kde=False, discrete=True)
    plt.xlabel("Number of Contributors")
    plt.ylabel("Number of Projects")
    plt.title("Distribution of Contributors per Project")
    plt.tight_layout()
    plt.show()

def success_rate_over_time(projects):
    events = [(p["start_time"], p["finished"], p["final_reward"]) for p in projects]
    events.sort(key=lambda x: x[0])
    time_steps, success_rates = [], []
    finished, successful = 0, 0
    for t, is_finished, score in events:
        if is_finished:
            finished += 1
            if score > 0: successful += 1
        success_rates.append(successful / finished if finished > 0 else 0)
        time_steps.append(t)
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=time_steps, y=success_rates)
    plt.xlabel("Time Step")
    plt.ylabel("Success Rate")
    plt.title("Project Success Rate Over Time")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()

def quality_metrics_over_time(projects):
    projects_per_time = {p["start_time"]:[] for p in projects}
    for p in projects: projects_per_time[p["start_time"]].append(p)
    projects_sorted = sorted(projects_per_time.items(), key=lambda p: p[0])
    times = [t for t, _ in projects_sorted]
    quality = [np.mean([p["quality_score"] for p in pjs]) for _, pjs in projects_sorted]
    novelty = [np.mean([p["novelty_score"] for p in pjs]) for _, pjs in projects_sorted]
    value = [np.mean([p["societal_value_score"] for p in pjs]) for _, pjs in projects_sorted]
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=times, y=quality, label="Quality Score")
    sns.lineplot(x=times, y=novelty, label="Novelty")
    sns.lineplot(x=times, y=value, label="Societal Value")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title("Project Metrics Over Time")
    plt.legend()
    plt.show()

def collab_metrics_over_time(projects):
    projects_per_time = {p["start_time"]: [] for p in projects}
    for p in projects: projects_per_time[p["start_time"]].append(p)
    projects_sorted = sorted(projects_per_time.items(), key=lambda p: p[0])
    times = [t for t, _ in projects_sorted]
    effort = [np.mean([p["current_effort"] for p in pjs]) for _, pjs in projects_sorted]
    n_contributors = [np.mean([len(p["contributors"]) for p in pjs]) for _, pjs in projects_sorted]
    citations = [np.sum([len(p.get("citations", [])) for p in pjs]) for _, pjs in projects_sorted]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    sns.lineplot(x=times, y=effort, ax=axes[0])
    axes[0].set_title("Current Effort")
    sns.lineplot(x=times, y=n_contributors, ax=axes[1])
    axes[1].set_title("# Contributors per Paper")
    sns.lineplot(x=times, y=citations, ax=axes[2])
    axes[2].set_title("Citations")
    plt.tight_layout()
    plt.show()

def projects_per_contributor(projects):
    all_contributors = []
    for p in projects: all_contributors.extend(p["contributors"])
    author_counts = Counter(all_contributors)
    papers_per_author = list(author_counts.values())
    plt.figure(figsize=(8, 5))
    sns.histplot(papers_per_author, bins=range(1, max(papers_per_author) + 2), kde=False, discrete=True)
    plt.xlabel("Number of Papers per Author")
    plt.ylabel("Number of Authors")
    plt.title("Authors by Number of Papers")
    plt.tight_layout()
    plt.show()

def completion_per_project(projects):
    completion = [max(0, p["current_effort"] / p["required_effort"]) * 100 for p in projects]
    plt.figure(figsize=(8, 5))
    sns.histplot(completion, bins=range(0, 110, 10), kde=False)
    plt.xlabel("Completion (%)")
    plt.ylabel("Number of Projects")
    plt.title("Distribution of Project Completion")
    plt.tight_layout()
    plt.show()

def quality_per_project(projects):
    quality_scores = [p["quality_score"] for p in projects]
    plt.figure(figsize=(8, 5))
    sns.histplot(quality_scores, bins=[i/10 for i in range(0, 11)], kde=False)
    plt.xlabel("Quality Score")
    plt.ylabel("Number of Projects")
    plt.title("Distribution of Project Quality Scores")
    plt.tight_layout()
    plt.show()