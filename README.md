# Scientific Knowledge Production Simulation - Quick Start

## Prerequisites
- Python 3.9+
- Use a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Simulation
Runs the simulation for one random seed and three reward functions and saves the results.

```bash
python run_policy_simulation.py
```



Outputs (written in the repo root and `log/`):
- `balanced_summary.json`: high-level results (steps, rewards, success rate, policy distribution)
- `log/balanced_actions.jsonl`: actions taken each step (JSONL)
- `log/balanced_observations.jsonl`: observations per step (JSONL)
- `log/balanced_projects.json`: final project states
- `log/balanced_area.pickle`: serialized environment state

Tune the call to `run_simulation_with_policies(...)` in `run_policy_simulation.py` to change:
- `n_agents`, `max_steps`, `n_groups`, `max_peer_group_size`
- `policy_distribution` (see `POLICY_CONFIGS` in the same file)
- `output_file_prefix` (affects filenames)

## Run Calibration

Run the calibration script with this command (to do this the real world data sets are necessary).
````bash
python calibrate.py
````

## Agent Policies (in `reputation-environment/agent_policies.py`)

- Careerist (`careerist`)
  - Picks high-prestige opportunities above a threshold.
  - Collaborates with active peers at/above the active-peer average reputation.
  - Effort goes to the closest-deadline running project that still needs work.

- Orthodox Scientist (`orthodox_scientist`)
  - Prefers lowest-novelty opportunities; ties break toward higher prestige.
  - Collaborates with all active peers who have close topic centroids.
  - Effort prioritizes projects already below 90% of required effort; otherwise best peer fit.

- Mass Producer (`mass_producer`)
  - Take project if the (effort × time window) is realtively low.
  - Collaborates with all active peers within the action mask.
  - Effort goes to the closest-deadline running project that still needs work.

Shared safety: if a running project risks missing its requirement given remaining time, policies skip selecting a new project that step.

## Inspect Results
Run `python process_results.py` to create reward trajectories for log files of simulation runs of 10 seeds. **Don't run this if you didn't run the full simulation runs for all 10 seeds before!**

To simply reproduce the figures you can use the saved results in `results`.

Run `visualizations.ipynb`to reproduce the figures.

## Tips
- JSONL logs can be large; use `jq`, `tail -f`, or sample lines.
- Commit parameter changes alongside their `*_summary.json` for reproducibility.

## Run rl agent training with Ray RLlib
Supported algorithms: `PPO`, `APPO`, `DREAMERV3`.

```bash
# Default: PPO
python train_rl_agent.py

# Using APPO
python train_rl_agent.py --algo APPO

# Using DreamerV3 (requires more resources, GPU recommended)
python train_rl_agent.py --algo DREAMERV3
```
## Run the simulation with a trained RL agent
```bash
python run_policiy_simulation_with_rl_agent.py `
  --checkpoint "checkpoints\09-04-2026\balanced_by_effort_iter0019_mrl50_09-04-01-52_eval4.87_best" `
  --seed 101 `
  --num-seeds 10 `
  --all-rewards
```
- `--checkpoint`: Path to the checkpoint directory containing the trained RL agent model. The checkpoint directory should contain the necessary files for loading the model, such as the model weights and configuration.
- `--seed`: Starting seed for the simulation.
- `--num-seeds`: Number of seeds to run the simulation with. Each seed will have a different random seed for reproducibility.
- `--all-rewards`: If set, the script will print all rewards for each episode instead of just the final reward.

## Overleaf workflow
- After writing in Overleaf pull the latest version from Overleaf with `git subtree pull --prefix thesis overleaf OVERLEAF_BRANCH --squash`
- then `git push origin main`