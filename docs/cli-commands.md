# CLI Commands

## Debugging `test_rl_agent.py`:

### Basic debug command:
```bash
python .\scripts\test_rl_agent.py --num-seeds 1 --seed 42 --debug-all --output-prefix "debug_run" --output-dir "test_results" --checkpoint "checkpoints/27-03-2026/balanced_by_effort_iter0007_mrl250_27-03-15-16_eval7.66_best"
```

*Note: Results are now saved in a unique timestamped subdirectory within `--output-dir`.*

### Control verbosity with `--debug-freq`:
- `--debug-freq 1` → Every step (very verbose)
- `--debug-freq 5` → Every 5 steps
- `--debug-freq 10` → Every 10 steps

## Debugging `train_rl_agent.py`
```bash
 python .\scripts\train_rl_agent.py --iterations 5 --wandb-mode disabled --train-batch-size 1000 --max-peer-group-size 10 --n-groups 10 --n-agents 100
```
## Debugging `train_dreamerv3.py`
```bash
python scripts\train_dreamerv3.py --total-env-steps 5000 --num-gpus 1 --training-ratio 64 --wandb-mode online --wandb-group "DreamerV3_Test"
```

### With Invalid Action Penalty (Recommended)
```bash
python scripts\train_dreamerv3.py --total-env-steps 5000 --wandb-mode online --wandb-group "DreamerV3_Test" --invalid-action-penalty 0.1
```

### With Action Mask Debugging
```bash
python scripts\train_dreamerv3.py --total-env-steps 500 --wandb-mode online --wandb-group "DreamerV3_Test" --use-light-policy-obs --debug-action-mask --debug-action-mask-steps 100 --debug-action-mask-interval 250 --debug-action-mask-jsonl debug_action_mask.jsonl --invalid-action-penalty 0.1
```

## Training `train_rl_agent.py`
````bash
python .\scripts\train_rl_agent.py --iterations 100 --seed 1 --wandb-group "Default_Setup1" 
````
### Continue Training
```bash
python .\scripts\train_rl_agent.py --iterations 39 --seed 1 --wandb-group "Default_Setup1" --wandb-run-id "eiy14h2k" --checkpoint "checkpoints/26-04-2026/balanced_by_effort_iter0069_mrl50_eiy14h2k_26-04-06-23_eval12.35_periodic" --use-light-policy-obs 
```

## Evaluations
```bash
python .\scripts\eval_rl_agent.py --output-dir "test_results/ppo_exp2" --output-prefix "ppo_exp2" --checkpoint "checkpoints/PPO_checkpoints/ppo_balanced_by_effort_iter0099_mrl50_vsy7vsv9_14-05-07-22_eval_na_periodic_seed5" --algo "PPO" --num-seeds 20 --seed 501 --reward-function "by_effort"
```

## Aggregate Actions and Observations for counterfactual analysis
````bash
python analysis/aggragate_actions_and_observation_over_seeds.py --input-dir test_results --output-dir results/aggregated_actions_and_obs --seed-start 501 --seed-end 520
````