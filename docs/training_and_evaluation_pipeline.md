# # Documentation: process_rl_results.py

1. Sweep / coarse search
   seed 42, 300k env steps

2. Validation / model selection
   Top configs auf seeds 101–110 evaluieren

3. Final training
   beste Config mit train seeds 1–5 trainieren
   je 100 Iterationen / ca. 1 Mio env steps

4. Final test
   jedes der 5 Modelle auf test seeds 201–210 testen

5. Reporting
   mean ± std über 5 × 10 = 50 Evaluations

## Overview

This project trains a **PPO reinforcement learning agent** in the Game of Science environment using RLlib.
Training is performed with multiple random seeds to evaluate robustness and reproducibility.
Each trained policy is then evaluated across multiple evaluation seeds.

The goal is to measure the expected performance of the RL agent under stochastic environment conditions and compare it against heuristic archetype agents.

## Training Procedure
### Algorithm

- Algorithm: **Proximal Policy Optimization (PPO)**
- Framework: **Ray RLlib**
- Backend: **PyTorch**

Only **one agent** in the environment is controlled by the RL policy.
All other agents follow fixed heuristic policies.

## Environment Setup
Typical configuration:
`start_agents = 60`
`max_agents = 64`
`max_steps = 500`

`n_groups = 8`
`max_peer_group_size = 8`

`n_projects_per_step = 1`
`max_projects_per_agent = 6`

`reward_mode = by_effort`
`acceptance_threshold = 0.5`

Episode termination occurs when:
- `max_steps` is reached
- the controlled agent dies
- environment termination conditions are triggered

## Training Seeds
To account for stochasticity in reinforcement learning, training is repeated with multiple seeds.
`training_seeds = [1,2,3,4,5,6,7,8,9,10]`

Each training seed produces an independent trained policy.
```
seed_1 → PPO training → policy_checkpoint_1
seed_2 → PPO training → policy_checkpoint_2
...
seed_10 → PPO training → policy_checkpoint_10
```

## Training Duration
Training is performed for a fixed number of PPO iterations.
Typical configuration:
`training_iterations = 600`
`train_batch_size = 32000`
`save_every_n_iters = 50`

Approximate runtime:
`~18 hours per training run (depending on hardware)`

## Logged Training Metrics
Each training run logs metrics to Weights & Biases (W&B).
Example metrics:
- `episode_reward_mean`
- `episode_len_mean`
- `policy_loss`
- `vf_loss`

## Evaluation Procedure
After training, the resulting policy is evaluated using fixed policy inference.
No learning occurs during evaluation.
Evaluation is performed using:
`run_policy_simulation_with_rlagent.py`

## Evaluation Seeds
Each trained policy (from one of the 10 training seeds) is evaluated across 10 independent evaluation seeds. 
Seeds **101 to 200** are reserved for evaluation purposes.

By default, the aggregation script `process_rl_results.py` processes **seeds 101 to 110** for each strategy.

Total evaluation runs for a complete empirical study:
```
training_seeds × evaluation_seeds per policy
= 10 × 10
= 100 simulations
```

