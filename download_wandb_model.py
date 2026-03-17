"""
download_wandb_model.py

Downloads a trained PPO model artifact from Weights & Biases
using the public API (no active run needed, avoids Windows cache issues).

Usage:
    python download_wandb_model.py
    python download_wandb_model.py --artifact "rl_in_the_game_of_science/RL in the Game of Science/ppo-best-Balanced-s42:latest"
    python download_wandb_model.py --artifact "rl_in_the_game_of_science/RL in the Game of Science/ppo-best-Balanced-s42:v0"
    python download_wandb_model.py --output ./models
"""

from __future__ import annotations

import argparse
import os
import shutil

import wandb


DEFAULT_ARTIFACT = (
    "rl_in_the_game_of_science/RL in the Game of Science/"
    "ppo-best-Balanced-s42:latest"
)

DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models"
)


def clear_wandb_cache() -> None:
    """Remove the local W&B artifact cache to avoid Windows PermissionError on os.replace()."""
    cache_dir = os.path.join(
        os.path.expanduser("~"), "AppData", "Local", "wandb", "wandb", "Cache"
    )
    if os.path.isdir(cache_dir):
        print(f"Clearing W&B cache: {cache_dir}")
        shutil.rmtree(cache_dir, ignore_errors=True)


def download_artifact(artifact_name: str, output_dir: str) -> str:
    """Download a W&B model artifact using the public API and return the local directory path."""

    # Clear stale cache to prevent Windows PermissionError
    clear_wandb_cache()

    os.makedirs(output_dir, exist_ok=True)

    # Use the public API – no wandb.init() / run needed
    api = wandb.Api()
    artifact = api.artifact(artifact_name, type="model")

    print(f"Downloading artifact: {artifact.name} (version {artifact.version})")
    artifact_dir = artifact.download(root=output_dir, skip_cache=True)
    print(f"Artifact downloaded to: {artifact_dir}")

    # List downloaded files
    for root, dirs, files in os.walk(artifact_dir):
        for f in files:
            full = os.path.join(root, f)
            size = os.path.getsize(full)
            print(f"  {os.path.relpath(full, artifact_dir)}  ({size:,} bytes)")

    return artifact_dir


def main():
    parser = argparse.ArgumentParser(description="Download a model artifact from W&B")
    parser.add_argument(
        "--artifact",
        type=str,
        default=DEFAULT_ARTIFACT,
        help="Full artifact path (entity/project/name:version)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Local directory to download the artifact into (default: ./models)",
    )
    args = parser.parse_args()

    artifact_dir = download_artifact(args.artifact, args.output)
    print(f"\nModel ready at: {artifact_dir}")


if __name__ == "__main__":
    main()

