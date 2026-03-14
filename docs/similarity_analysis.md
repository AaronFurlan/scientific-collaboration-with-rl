### Similarity Analysis of the RL Agent

This document explains how the similarity between the Reinforcement Learning (RL) agent and the predefined scientific archetypes (`careerist`, `orthodox_scientist`, `mass_producer`) is analyzed.

The main goal of the analysis is to understand which archetype the RL agent's behavior most closely resembles. For this purpose, a feature vector is created for each timestep and each agent, which is then compared with the vectors of other agents.

#### Creation of the Feature Vector

In each timestep, the feature vector for each agent is formed from its **action** and the associated **observation**.

Since actions are often context-dependent (e.g., "Invest in Project 1" means different things depending on the project), the analysis process uses the observation data to transform actions into a meaningful vector.

The final vector consists of three main categories:

1.  **Project Choice Features (`choose_project`)**
    When an agent chooses a new project, the following features are extracted:
    *   `choose_project_binary`: Was any project chosen at all? (0 or 1)
    *   `chosen_project_required_effort`: Required effort of the chosen project.
    *   `chosen_project_prestige`: Prestige value of the chosen project.
    *   `chosen_project_novelty`: Degree of novelty of the chosen project.
    *   `chosen_project_time_window`: Time window in which the project must be completed.

2.  **Collaboration Features (`collaborate_with`)**
    Analyzes who the agent collaborates with:
    *   `n_selected_collaborators`: Number of selected collaboration partners.
    *   `mean_selected_peer_reputation`: Average reputation of the selected partners.
    *   `mean_selected_peer_distance_to_self_centroid`: Average distance of the partners to the agent's own scientific focus (centroid) in the topic space.

3.  **Effort Allocation Features (`put_effort`)**
    When the agent invests effort into an ongoing project:
    *   `effort_is_none` / `effort_is_active`: Is the agent currently investing effort?
    *   `effort_target_required_effort`: Total required effort of the target project.
    *   `effort_target_prestige`: Prestige of the target project.
    *   `effort_target_novelty`: Novelty of the target project.
    *   `effort_target_time_left`: Remaining time for the project.
    *   `effort_target_current_effort`: Effort already invested in the project.
    *   `effort_target_n_contributors`: Number of current contributors.
    *   `effort_target_mean_peer_fit`: Average fit of the other contributors to the project.

#### Combination and Normalization

All the features mentioned above are combined into a single vector. To ensure stable comparability, the keys of the feature dicts are sorted alphabetically before being converted into a `numpy` array.

#### Comparison Metrics

To calculate the similarity between the RL agent ($v_{RL}$) and an archetype ($v_{Arch}$), two metrics are used:

1.  **Cosine Similarity:**
    Measures the angle between the vectors. A value of 1 means identical orientation (maximum similarity), 0 means orthogonality.
    $$\text{sim}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$

2.  **Euclidean Distance:**
    Measures the actual distance between points in n-dimensional space. A smaller value indicates higher similarity.
    $$d(A, B) = \sqrt{\sum (a_i - b_i)^2}$$

#### Analysis Process

The script `analyze_agent_similarity.py` performs the following steps:
1. Loads the log files (`actions.jsonl` and `observations.jsonl`).
2. Creates the feature vector for each agent per timestep.
3. Calculates the average similarity of the RL agent to all instances of the respective archetypes.
4. Generates time series diagrams and distribution boxplots in the `results/` folder.
