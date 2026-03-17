"""
checkpoint_utils.py

Utilities for building explicit, descriptive checkpoint paths for RLlib training runs.

Checkpoint naming convention:
    <policy>_<reward>_iter<N>_mrl<M>_<dd-mm-HH-MM>_eval<X>[_<tag>]

Directory structure:
    <base_dir>/<dd-mm-yyyy>/  (one subfolder per calendar day)

Example:
    checkpoints/10-03-2026/balanced_by_effort_iter0004_mrl500_10-03-14-30_eval12.35_best/
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from typing import Optional


def build_checkpoint_path(
    *,
    base_dir: str = "checkpoints",
    policy_config_name: str,
    reward_function: str,
    iteration: int,
    max_rewardless_steps: int,
    eval_return: Optional[float] = None,
    tag: str = "",
) -> str:
    """Build an explicit checkpoint directory path with dynamic naming.

    Structure:
        <base_dir>/<dd-mm-yyyy>/<policy>_<reward>_iter<N>_mrl<M>_<dd-mm-HH-MM>_eval<X>[_<tag>]/

    A daily subfolder is created automatically if it does not exist yet.

    Parameters
    ----------
    base_dir : str
        Root directory for all checkpoints (default: ``"checkpoints"``).
    policy_config_name : str
        Name of the policy distribution (e.g. ``"Balanced"``).
    reward_function : str
        Reward scheme identifier (e.g. ``"by_effort"``).
    iteration : int
        Current training iteration number.
    max_rewardless_steps : int
        Environment parameter – embedded in the folder name for traceability.
    eval_return : float or None
        Evaluation return to include in the name.  ``None`` or ``NaN`` → ``"eval_na"``.
    tag : str
        Optional suffix such as ``"best"`` or ``"periodic"`` to distinguish
        checkpoint types.

    Returns
    -------
    str
        Full filesystem path for the checkpoint directory.
        The parent (daily) directory is guaranteed to exist after this call.
    """
    now = datetime.now()
    day_folder = now.strftime("%d-%m-%Y")       # e.g. "10-03-2026"
    timestamp = now.strftime("%d-%m-%H-%M")     # e.g. "10-03-14-30"

    # Sanitize names for filesystem (spaces → underscores, lowercase)
    policy_safe = policy_config_name.replace(" ", "_").lower()
    reward_safe = reward_function.replace(" ", "_").lower()

    # Format eval return (handle None / NaN)
    if eval_return is not None and not math.isnan(eval_return):
        eval_str = f"eval{eval_return:.2f}"
    else:
        eval_str = "eval_na"

    parts = [
        policy_safe,
        reward_safe,
        f"iter{iteration:04d}",
        f"mrl{max_rewardless_steps}",
        timestamp,
        eval_str,
    ]
    if tag:
        parts.append(tag)

    folder_name = "_".join(parts)
    full_path = os.path.join(base_dir, day_folder, folder_name)
    full_path = os.path.abspath(full_path)

    # Ensure daily subfolder exists
    os.makedirs(os.path.join(base_dir, day_folder), exist_ok=True)

    return full_path

