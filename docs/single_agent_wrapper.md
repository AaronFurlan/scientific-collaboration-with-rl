### RLLib Single-Agent Wrapper

The `RLLibSingleAgentWrapper` is a crucial component in this project that translates the multi-agent environment (`PeerGroupEnvironment`) into a single-agent interface compatible with RLlib. This allows applying standard RL algorithms (such as PPO) to a single agent while the remaining agents in the environment are controlled by predefined heuristics or policies.

#### Key Tasks of the Wrapper

1.  **Interface Conversion:** Transformation of a PettingZoo-like parallel environment into a `gym.Env` (single-agent).
2.  **Action Space Abstraction (Macro-Actions):** Conversion of the environment's complex, hierarchical action space into a simple discrete action space.
3.  **Observation Processing (Flattening):** Transformation of nested dictionary observations into flat feature vectors.
4.  **Action Validation (Masking):** Ensuring the RL agent only performs valid actions by applying the environment's `action_mask`.
5.  **Top-K Collaboration Filtering:** An optional mechanism to limit the number of collaboration partners based on scientific metrics.

#### Action Space (Discrete Macro-Actions)

Since the environment requires simultaneous decisions on project choice, effort allocation, and collaboration partners, the wrapper uses an **action encoding process**. The resulting discrete action space has the size:
$N = N_{\text{projects}} \times N_{\text{effort}} \times 2^{N_{\text{peers}}}$

A single integer value is decoded as follows:
*   **Project Choice (`choose_project`):** Which new project should be started? (0 = none, 1-n = project slot)
*   **Effort (`put_effort`):** In which ongoing project should work be invested? (0 = none, 1-n = project slot)
*   **Collaboration (`collaborate_with`):** A bit vector indicating which peers are invited to collaborate.

#### Observation Space (Flattening)

RLlib models usually require flat tensors as input. The wrapper takes the rich, structured observations of the environment (information about own projects, available project opportunities, reputation of peers, etc.) and converts them into a continuous vector (`Box` space).

#### Top-K Collaboration Mechanism

To keep the action space manageable, a `topk_collab` restriction can be activated.
If an agent selects more than $K$ partners, the wrapper filters the selection based on a score composed of:
*   **Reputation** of the partner.
*   **Thematic Distance** (based on the centroid in the topic space).
*   **Group Membership** (bonus for members of the same working group).

This ensures that even if the RL agent selects "all" peers, only the $K$ most promising partners are actually requested.

#### Control of Other Agents

While the wrapper exposes a "controlled agent" for RL training, the other agents in the simulation must continue to act. This is done via the `other_policies` parameter. Archetypes (`careerist`, `orthodox_scientist`, etc.) can be assigned here, so the RL agent learns in a dynamic, populated world.

#### Use in Training

In the file `train_ppo_rllib.py`, the wrapper is used to prepare the environment for the PPO algorithm:

```python
env_config = {
    "env": peer_group_env,
    "controlled_agent": "agent_0",
    "other_policies": archetypes_mapping,
    "topk_collab": 5
}
wrapped_env = RLLibSingleAgentWrapper(**env_config)
```
