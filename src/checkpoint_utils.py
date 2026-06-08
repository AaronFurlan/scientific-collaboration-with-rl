"""Utilities for building descriptive checkpoint paths for RLlib training."""

from __future__ import annotations

import math
import os
from datetime import datetime
from typing import Optional


def build_checkpoint_path(
    *,
    base_dir: str = "checkpoints",
    algo: str = "ppo",
    policy_config_name: str,
    reward_function: str,
    iteration: int,
    max_rewardless_steps: int,
    eval_return: Optional[float] = None,
    wandb_run_id: Optional[str] = None,
    tag: str = "",
) -> str:
    """
    Build checkpoint directory path with naming convention:
    <base_dir>/<dd-mm-yyyy>/<algo>_<policy>_<reward>_iter<N>_mrl<M>_<timestamp>_eval<X>[_<tag>]/

    Args:
        base_dir: Root checkpoint directory
        algo: Algorithm name (e.g., "ppo")
        policy_config_name: Policy distribution name
        reward_function: Reward scheme identifier
        iteration: Training iteration number
        max_rewardless_steps: Environment parameter
        eval_return: Evaluation return (None/NaN → "eval_na")
        tag: Optional suffix (e.g., "best", "periodic")

    Returns:
        Full filesystem path for checkpoint directory
    """
    now = datetime.now()
    day_folder = now.strftime("%d-%m-%Y")       # e.g. "10-03-2026"
    timestamp = now.strftime("%d-%m-%H-%M")     # e.g. "10-03-14-30"

    # Sanitize names for filesystem (spaces → underscores, lowercase)
    algo_safe = algo.replace(" ", "_").lower()
    policy_safe = policy_config_name.replace(" ", "_").lower()
    reward_safe = reward_function.replace(" ", "_").lower()

    # Format eval return (handle None / NaN)
    if eval_return is not None and not math.isnan(eval_return):
        eval_str = f"eval{eval_return:.2f}"
    else:
        eval_str = "eval_na"

    parts = [
        algo_safe,
        policy_safe,
        reward_safe,
        f"iter{iteration:04d}",
        f"mrl{max_rewardless_steps}",
    ]
    if wandb_run_id:
        parts.append(wandb_run_id)
    
    parts.extend([
        timestamp,
        eval_str,
    ])
    if tag:
        parts.append(tag)

    folder_name = "_".join(parts)
    full_path = os.path.join(base_dir, day_folder, folder_name)
    full_path = os.path.abspath(full_path)

    # Ensure daily subfolder exists
    os.makedirs(os.path.join(base_dir, day_folder), exist_ok=True)

    return full_path

