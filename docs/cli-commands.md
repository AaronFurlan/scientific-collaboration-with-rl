# CLI Commands

## Debugging `run_policy_simulation_with_rlagent.py`:

### Basic debug command:
```bash
python .\scripts\run_policy_simulation_with_rlagent.py --num-seeds 1 --seed 42 --debug-all --output-prefix "debugg projects" --checkpoint "checkpoints/27-03-2026/balanced_by_effort_iter0007_mrl250_27-03-15-16_eval7.66_best"
```

### Control verbosity with `--debug-freq`:
- `--debug-freq 1` → Every step (very verbose)
- `--debug-freq 5` → Every 5 steps
- `--debug-freq 10` → Every 10 steps

## Debugging `train_rl_agent.py`
```bash
 python .\scripts\train_rl_agent.py --iterations 5 --wandb-mode disabled --train-batch-size 1200 --max-peer-group-size 10 --n-groups 10 --n-agents 60
```

## Training `train_rl_agent.py`
````bash
python .\scripts\train_rl_agent.py --iterations 100 --seed 1 --wandb-group "Default_Setup1" 
````

