# Top-k Collaboration Mechanism

The Top-k collaboration mechanism is an ablation and constraint system used in the `RLLibSingleAgentWrapper`. It limits the number of collaboration partners an agent can select in a single simulation step to a maximum of `k`.

## Overview

In the *Game of Science* simulation, agents can propose projects and select collaborators. Without constraints, an agent could theoretically collaborate with an unlimited number of peers (limited only by the peer group size). The Top-k mechanism introduces a realistic capacity constraint or behavioral restriction.

## How it Works

The logic is implemented in the `_apply_topk_collaboration` and `_compute_peer_scores` methods within `rllib_single_agent_wrapper.py`.

### 1. Candidate Filtering
When an agent (either the controlled RL agent or a heuristic agent) selects a set of collaboration bits, the wrapper first cleans them against the environment's `action_mask` to ensure only valid peers are considered.

### 2. Scoring System
If the number of selected valid peers exceeds the threshold `k`, the wrapper calculates a **collaboration score** for every potential peer in the group. The score is a weighted combination of three factors extracted from the agent's observations:

*   **Reputation**: Peers with higher reputation are preferred.
*   **Distance**: Peers with smaller distance (thematic or spatial) are preferred.
*   **Same-group Bonus**: A bonus is applied if the peer belongs to the same group as the acting agent.

The score formula is:
```
Score = (w_rep * Norm_Reputation) - (w_dist * Norm_Distance) + (w_same * SameGroup)
```
Where:
*   `Norm_Reputation` and `Norm_Distance` are min-max normalized values (0 to 1) across the current group.
*   `Distance` is calculated as the Euclidean distance between the agent's `self_centroid` and the peer's `peer_centroids` (observable features).
*   `SameGroup` is 1 if in the same group, 0 otherwise. (Note: Currently, group IDs are not explicitly part of the observation, so this bonus defaults to 0 unless the environment is extended).

**Default Weights:**
*   `w_rep` (Reputation): 1.0
*   `w_dist` (Distance): 1.0
*   `w_same` (Same Group): 0.5

### 3. Selection & Pruning
1.  Candidates are sorted by their score in descending order.
2.  A small, deterministic noise (seeded) is added to ensure stable tie-breaking.
3.  Only the top `k` candidates are kept.
4.  The action bit-vector is modified: bits for the selected top `k` peers remain `1`, while all other collaboration bits are set to `0`.

## Configuration via CLI

In `run_policy_simulation_with_rlagent.py`, the mechanism can be controlled using the following arguments:

*   `--topk <n>`: Sets the value for `k`. If not provided (or `None`), the restriction is disabled.
*   `--topk-all-agents`: Applies the `k` limit to **all** agents in the environment (RL and heuristics). This is the default behavior if `--topk` is used.
*   `--no-topk-all-agents`: Applies the `k` limit **only** to the controlled RL agent. Heuristic agents will not be pruned by the wrapper.

## Purpose in Research
This mechanism allows researchers to study how agent policies adapt when they cannot collaborate with everyone they want. It forces the RL agent to learn to prioritize "high-value" collaborators based on the observable features (reputation, distance) rather than just choosing everyone available.
