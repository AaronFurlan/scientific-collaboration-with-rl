# Documentation: process_rl_results.py

This script is used for the empirical evaluation of Reinforcement Learning (RL) simulation results. It aggregates data from multiple simulation runs (different seeds) and prepares them for statistical analysis.

## Functionality

The script processes JSONL log files generated during the evaluation of an RL agent. It calculates statistics on the collected rewards, agent age, and H-index, categorized by agent types (archetypes).

### 1. Data Sources
The script looks for files in the `log/` folder with the following naming scheme:
`rl_ppo_{strategy}_s{seed}_actions.jsonl`
`rl_ppo_{strategy}_s{seed}_observations.jsonl`

By default, **seeds 101 to 110** are processed for the strategies `multiply`, `evenly`, and `by_effort`.

### 2. RL Agent Identification
Since the controlled RL agent (usually `agent_0`) often does not carry a fixed archetype name in the simulation, the script automatically recognizes it and marks it as **`RL_Agent`** in the resulting data. This allows for a direct performance comparison between the learned agent and rule-based heuristics (e.g., Balanced, Greedy).

### 3. Generated Outputs (results/)
The results are saved in the `results/` folder in the efficient **Apache Parquet** format:

| File | Description |
| :--- | :--- |
| `rl_summary_by_archetype_{strategy}.parquet` | Aggregated statistics per time step: Mean (`mean_reward`), standard deviation (`std_reward`), and number of active agents per archetype. |
| `rl_trajectories_{strategy}.parquet` | Raw trajectories of all agents across all steps and seeds. Contains rewards, H-index, and age. |

## Workflow Usage

To generate the data correctly, the simulation logs must be saved with the appropriate prefix.

### Example: Run Evaluation (PowerShell)
```powershell
# Eval run for all strategies and seeds 101-110
foreach ($strat in "multiply", "evenly", "by_effort") {
    foreach ($s in 101..110) {
        python run_policy_simulation_with_rlagent.py `
            --checkpoint "checkpoints/best_model" `
            --reward-function $strat `
            --seed $s `
            --output-prefix "rl_ppo_${strat}_s${s}"
    }
}
```

### Example: Aggregate Results
```powershell
python process_rl_results.py
```

The generated Parquet files can then be loaded directly into Python (e.g., with Pandas) or into BI tools for final visualization (plots of learning curves, performance boxplots).
