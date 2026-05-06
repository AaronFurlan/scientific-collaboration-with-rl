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
- Seed 2: 100 iterations, run ID: nvnrxjxo (`checkpoints/25-04-2026/balanced_by_effort_iter0099_mrl50_25-04-03-13_eval_na_periodic`)
- Seed 3: 100 iterations, run ID: eiy14h2k (`checkpoints/26-04-2026/balanced_by_effort_iter0069_mrl50_eiy14h2k_26-04-06-23_eval12.35_periodic`) # Crashed, continued training
- Seed 4: 100 iterations, run ID: aw841bxi (`checkpoints/26-04-2026/balanced_by_effort_iter0099_mrl50_aw841bxi_26-04-20-42_eval_na_periodic`)
- Seed 5: 100 iterations, run ID: dbdosvul (`checkpoints/27-04-2026/balanced_by_effort_iter0099_mrl50_dbdosvul_27-04-13-27_eval_na_periodic`)

- seed 1: best: `checkpoints/23-04-2026/balanced_by_effort_iter0068_mrl50_23-04-05-55_eval15.25_best`
- seed 2: best: `checkpoints/24-04-2026/balanced_by_effort_iter0062_mrl50_24-04-21-59_eval17.93_best`
- Seed 3: best: `checkpoints/26-04-2026/balanced_by_effort_iter0057_mrl50_eiy14h2k_26-04-05-02_eval16.29_best`

## Config 2 (Default-Werte + Episoden-Seeding)

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

## Trained on found hyperparameters and config 2
`python .\scripts\train_rl_agent.py --iterations 100 --seed 2 --wandb-group "Default_Setup1"`
- Seed 1: 100 iterations, run ID: a2s9m0hg (`checkpoints/28-04-2026/balanced_by_effort_iter0099_mrl50_a2s9m0hg_28-04-11-16_eval_na_periodic`)
- Seed 2: 100 iterations, run ID: 7kamraoi (`checkpoints/29-04-2026/balanced_by_effort_iter0099_mrl50_7kamraoi_29-04-08-06_eval_na_periodic`)
- Seed 3: 100 iterations, run ID: vurdac00 (`checkpoints/30-04-2026/balanced_by_effort_iter0099_mrl50_vurdac00_30-04-11-40_eval_na_periodic`)
- Seed 4: 100 iterations, run ID: pcudux1u (`checkpoints/01-05-2026/balanced_by_effort_iter0079_mrl50_pcudux1u_01-05-01-23_eval_na_periodic`) # Crashed, continued training
- Seed 5: 100 iterations, run ID: 44fwk28s (`checkpoints/02-05-2026/balanced_by_effort_iter0099_mrl50_44fwk28s_02-05-06-29_eval_na_periodic`)

- seed 1: best: ``
- seed 2: best: ``
- Seed 3: best: ``