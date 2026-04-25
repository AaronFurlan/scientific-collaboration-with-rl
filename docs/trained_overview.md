## Config 1 (Default-Werte)

### Environment
| Parameter | Default |
|---|---:|
| `--n-agents` | `400` |
| `--start-agents` | `100` |
| `--max-steps` | `600` |
| `--max-rewardless-steps` | `50` |
| `--n-groups` | `10` |
| `--max-peer-group-size` | `40` |
| `--n-projects-per-step` | `1` |
| `--max-projects-per-agent` | `8` |
| `--max-agent-age` | `750` |

### Reward und Schwellenwerte
| Parameter | Default |
|---|---:|
| `--acceptance-threshold` | `0.44` |
| `--reward-function` | `"by_effort"` |
| `--prestige-threshold` | `0.29` |
| `--novelty-threshold` | `0.4` |
| `--effort-threshold` | `35` |

## Hyperparameter Config 1 (Default-Werte)

| Parameter | Default |
|---|---:|
| `--train-batch-size` | `10000` |
| `--gamma` | `0.9583432181048404` |
| `--lambda` (`lambda_`) | `0.9626992994491804` |
| `--lr` | `0.00020375077263171516` |
| `--num-epochs` | `3` |
| `--entropy-coeff` | `0.005515494202562797` |
| `--vf-loss-coeff` | `1.941963717117803` |
| `--grad-clip` | `0.5223688871667344` |

## Trained on found hyperparameters and config 1
`python .\scripts\train_rl_agent.py --iterations 100 --seed 2 --wandb-group "Default_Setup1"`
- Seed 1: 100 iterations, run ID: hqi8lme0 (`checkpoints/23-04-2026/balanced_by_effort_iter0099_mrl50_23-04-11-18_eval_na_periodic`)
- Seed 2: 100 iterations, run ID: nvnrxjxo (`checkpoints/24-04-2026/balanced_by_effort_iter0062_mrl50_24-04-21-59_eval17.93_best`)

## Continue Training
`python .\scripts\train_rl_agent.py --iterations 50 --seed 1 --wandb-group "Default_Setup1" --wandb-run-id "hqi8lme0"`
- Seed 1: 25 iterations, run ID: hqi8lme0, total 150 iterations