### Collaboration Mechanism in the PeerGroupEnvironment

This document describes how agents in the `PeerGroupEnvironment` find each other, how the native logic of the environment forms cliques, and how different archetypes select their partners.

---

#### 1. The Core Process: Mutual Consent
Unlike many simpler RL environments, it is not enough for an agent to simply "want to work with someone". Collaboration is a **bi-directional process**:

1.  **Same Project:** Agent A and Agent B must choose exactly the same project via the `choose_project` action in the same timestep (`step`).
2.  **Mutual Interest:** 
    *   Agent A must mark the index of Agent B in its `collaborate_with` vector.
    *   Agent B must mark the index of Agent A in its `collaborate_with` vector.
3.  **Mutual Intent Filter:** The environment checks all received actions and only keeps the connections (edges in the graph) where this mutual interest exists (`intents & intents.T`).

#### 2. Clique Formation (Graph Theory)
After the environment has filtered the mutual interests, a graph is created for each project. The nodes are the agents who chose this project, and the edges represent the mutual interest.

*   **Algorithm:** The environment uses the **Bron-Kerbosch algorithm** (via `networkx.find_cliques`) to find "Maximal Cliques". A clique is a group of agents in which **everyone** has a mutual agreement with **every other** member of the group.
*   **Prioritization:** The largest found clique is processed first.
*   **Solo Fallback:** Agents who did not end up in any clique (e.g., because they didn't request anyone or their requests were not reciprocated) start the project alone as a solo project.

---

#### 3. Partner Selection of Archetypes
Each archetype follows its own strategy when selecting collaboration partners in the `collaborate_with` vector:

| Archetype | Partner Selection Strategy | Logic |
| :--- | :--- | :--- |
| **Careerist** | **Status-oriented** | Requests agents whose reputation is above the group average. If the Careerist themselves is the best, they request all active peers. |
| **Orthodox Scientist** | **Thematically focused** | Calculates the distance between their own research focus (Topic Centroid) and that of the peers. Requests only peers who are thematically closer than the average. |
| **Mass Producer** | **Quantity-oriented** | Requests basically **all** active peers within their range to maximize the probability of successful group formation. |

#### 4. The Role of the Top-K Wrapper
In the standard configuration (with RL training), the `RLLibSingleAgentWrapper` sits in front of the environment. If `topk_collab` is active, the agent's `collaborate_with` vector is filtered **before** the environment sees it.

This serves to keep the action space for the RL model small. The wrapper selects only the best $K$ partners based on a weighted score (reputation, distance, group membership), even if the agent has marked more. Without this wrapper (or if `topk_collab=None`), the agent's decision is passed unfiltered to the clique logic described above.
