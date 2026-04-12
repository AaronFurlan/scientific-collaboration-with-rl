### Collaboration & Observation Masking

This document describes the mechanisms of collaboration and how invalid or empty interaction opportunities are handled during agent training.

---

#### 1. Collaboration Intent & Bits

Collaboration between agents occurs via a binary vector (MultiBinary), where each bit corresponds to a "peer slot" in the environment.

*   **Intent**: In each step, the agent's neural network outputs a list of bits (`1` for "wants to collaborate", `0` for "does not want to"). This is referred to in the debug log as `n_requested_collaborators`.
*   **Decisions**: Actual collaboration only occurs if:
    1.  **Mutual Interest**: Both agents (A and B) must have set the corresponding bit for each other.
    2.  **Project Matching**: Both agents must choose to start a new project in the same step (`choose_project`).

---

#### 2. Observation Masking (Zero-Padding)

The environment has a fixed number of slots for peers (defined by `max_peer_group_size`). Since actual peer groups are often smaller or agents drop out, these slots must be "masked."

*   **Why a fixed size?**: Neural networks require a constant input dimension. We cannot physically "cut out" slots from the data stream.
*   **Visibility**: For the agent, inactive or empty slots are made "invisible" by zeroing out all associated features (**Zero-Padding**):
    *   `peer_group`: `0` (Inactive/Empty) vs. `1` (Active).
    *   `peer_reputation`: Set to `0.0`.
    *   `peer_h_index`: Set to `0`.
    *   `peer_centroids`: Set to `[0.0, 0.0]`.
*   **Learning Effect**: Through this process, the model learns that actions targeting "null slots" have no effect on the reward and automatically ignores them over time.

---

#### 3. Action Masking in the Wrapper

In addition to masking in the observation (what the agent sees), there is also masking of actions (what the agent does) in the `RLLibSingleAgentWrapper`.

*   **Safety Rail**: If the agent attempts to set a bit for an empty slot, the wrapper corrects this to `0` before the action is passed to the environment.
*   **Training Efficiency**: This prevents the agent from being punished for "impossible" actions during exploration or the environment from entering invalid states. It massively accelerates training convergence by limiting the search space to valid interactions.

---

#### 4. Debug Logs

You can track this process in the debug logs (`--debug-effort`):

*   `_decode_action`: Shows the agent's raw intent (`n_requested_collaborators`).
*   `_apply_action_mask`: Shows how many of these intents had to be revised due to inactive/empty slots (`INACTIVE or EMPTY`).
