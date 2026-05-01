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

## Training `train_rl_agent.py`
````bash
python .\scripts\train_rl_agent.py --iterations 100 --seed 1 --wandb-group "Default_Setup1" 
````
### Continue Training
```bash
python .\scripts\train_rl_agent.py --iterations 39 --seed 1 --wandb-group "Default_Setup1" --wandb-run-id "eiy14h2k" --checkpoint "checkpoints/26-04-2026/balanced_by_effort_iter0069_mrl50_eiy14h2k_26-04-06-23_eval12.35_periodic" --use-light-policy-obs 
```

## Testing
```bash
python .\scripts\test_rl_agent.py --num-seeds 20 --checkpoint "checkpoints/23-04-2026/balanced_by_effort_iter0099_mrl50_23-04-11-18_eval_na_periodic" --output-prefix "rl_agent_sim_config1_trainseed1" --all-rewards
```

## Aggregate Actions and Observations for counterfactual analysis
````bash
python analysis/aggragate_actions_and_observation_over_seeds.py --input-dir test_results --output-dir results/aggregated_actions_and_obs --seed-start 501 --seed-end 520
````