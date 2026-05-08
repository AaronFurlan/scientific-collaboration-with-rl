# Reinforcement Learning in the Game of Science - Quick Start

Quick start guide for the Reinforcement Learning in the Game of Science. Train and evaluate RL agents in a simulated scientific environment, comparing against heuristic policies.

## Prerequisites
- Python 3.12
- Weights and Biases account and API key
- Use a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## General

All runnable scripts are stored in the folder scripts/. In addition, there are cli arguments for all scripts that can be used to configure the run. Use `--help` to see all the options.
For example:
```bash
python scripts/train_rl_agent.py --help
```

For the training and hyperparameter tuning scripts, there is a `
    --use-light-policy-obs` cli argument that can be used to enable a lightweight observation für the archetypes.
This increases training speed by about 30%. 

## Algorithms

At the moment, there are three supported algorithms:
- PPO
- APPO
- DreamerV3

## Hyperparameter Tuning
Hyperparameter tuning is supported for PPO and APPO via Weights and Biases bayesian optimization.

To run hyperparameter tuning run:
```bash
python scripts/wandb_sweep_rl_agent.py --algo "PPO" --count 10
```
The above command will run 10 hyperparameter sweeps for PPO.

## Training

### Checkpointing
All training checkpoints are saved to the `checkpoints/` directory.
Checkpoints are automatically saved after a specific number of training iterations or when achieving new best eval/reward.


### PPO / APPO
Train a PPO or APPO agent using Ray RLlib:

```bash
python scripts/train_rl_agent.py --algo PPO --iterations 100 --policy-config Balanced
```

**Key arguments:**
- `--algo`: Algorithm to use (`PPO` or `APPO`)
- `--iterations`: Number of training iterations
- `--policy-config`: Policy distribution for other agents (e.g., `Balanced`, `All Careerist`)
- `--total-env-steps`: Total environment steps (alternative to iterations)
- `--num-workers`: Number of parallel workers for data collection
- `--train-batch-size`: Training batch size
- `--lr`: Learning rate
- `--use-light-policy-obs`: Use lightweight observations (recommended for faster training)

**Example with custom hyperparameters:**
```bash
python scripts/train_rl_agent.py \
    --algo APPO \
    --total-env-steps 500000 \
    --lr 0.0001 \
    --num-workers 8 \
    --train-batch-size 10000 \
    --use-light-policy-obs
```

### DreamerV3
Train a DreamerV3 agent:

```bash
python scripts/train_dreamerv3.py --episodes 1000 --policy-config Balanced
```

**Key arguments:**
- `--episodes`: Number of training episodes
- `--policy-config`: Policy distribution for other agents
- `--batch-size`: Batch size for model training
- `--lr`: Learning rate

## Evaluation

Evaluate a trained RL agent checkpoint:

```bash
python scripts/eval_rl_agent.py --checkpoint checkpoints/01-05-2026/balanced_by_effort_iter0011_mrl50_44fwk28s_01-05-20-50_eval14.74_best
```

**Key arguments:**
- `--checkpoint`: Path to the trained model checkpoint
- `--policy-config`: Policy distribution for other agents (must match training)
- `--seed`: Random seed for reproducibility
- `--n-agents`: Number of agents in the simulation
- `--max-steps`: Maximum number of simulation steps

**Example:**
```bash
python scripts/eval_rl_agent.py \
    --checkpoint checkpoints/PPO_checkpoint_000100 \
    --policy-config Balanced \
    --seed 42 \
```

### Output

Evaluation outputs include:
- **JSON files**: Detailed episode data (observations, actions, rewards)
- Stored in directory `test_results/`

## Analysis

Analysis notebooks are available in the `notebook/` directory. Some notebooks need processed data.
Processing can be done wiht the scripts:

- `analysis/process_rl_results.py`
- `analysis/process_rl_actions_and_obs.py`

Notebooks:
- `aggregate_obs_actions_for_similarity_analysis.py`: Analyze observation and action patterns
- `benchmark_light_observation.py`: Compare performance of light vs. full observations
- `validate_light_observation.py`: Verify correctness of light observation wrapper




