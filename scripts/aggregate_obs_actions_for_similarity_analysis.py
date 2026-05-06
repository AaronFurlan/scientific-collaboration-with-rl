#!/usr/bin/env python3
"""
Aggregate observations and actions from test_results subdirectories
for use in the archetype similarity analysis notebook.

This script:
1. Scans test_results/ for subdirectories with seeds 501-520
2. Loads actions and observations JSONL files
3. Extracts agent_0 (RL agent) data
4. Adds metadata (seed, step, episode)
5. Saves to parquet files for the similarity analysis notebook
"""

from pathlib import Path
import json
import pandas as pd
from typing import List, Dict, Any
import re


def extract_seed_from_dirname(dirname: str) -> int:
    """Extract seed number from directory name like 'rl_ppo_by_effort_s501_...'"""
    match = re.search(r'_s(\d+)_', dirname)
    if match:
        return int(match.group(1))
    return None


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load JSONL file and return list of records."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def process_actions(actions_data: List[Dict], seed: int, agent_id: str = "agent_0") -> List[Dict]:
    """
    Process actions JSONL data and extract agent_0 actions.

    Each line in actions.jsonl contains actions for all agents at one timestep.
    Format: {"agent_0": {...}, "agent_1": {...}, ...}

    Returns list of records with columns: seed, step, agent_id, action
    """
    records = []
    for step, timestep_data in enumerate(actions_data):
        if agent_id in timestep_data and timestep_data[agent_id] is not None:
            agent_action = timestep_data[agent_id]
            records.append({
                "seed": seed,
                "step": step,
                "agent_id": agent_id,
                "action": agent_action,
            })
    return records


def process_observations(obs_data: List[Dict], seed: int, agent_id: str = "agent_0") -> List[Dict]:
    """
    Process observations JSONL data and extract agent_0 observations.

    Each line in observations.jsonl contains observations for all agents at one timestep.
    Format: {"agent_0": {"observation": {...}, "action_mask": {...}}, "agent_1": {...}, ...}

    Returns list of records with columns: seed, step, agent_id, observation, action_mask
    """
    records = []
    for step, timestep_data in enumerate(obs_data):
        if agent_id in timestep_data and timestep_data[agent_id] is not None:
            agent_obs_data = timestep_data[agent_id]

            # Extract observation and action_mask
            observation = agent_obs_data.get("observation")
            action_mask = agent_obs_data.get("action_mask")

            records.append({
                "seed": seed,
                "step": step,
                "agent_id": agent_id,
                "observation": observation,
                "action_mask": action_mask,
            })
    return records


def find_result_directories(test_results_dir: Path, seed_range: tuple = (501, 520)) -> Dict[int, List[Path]]:
    """
    Find all result directories grouped by seed.

    Returns dict: {seed: [dir1, dir2, ...]}
    """
    seed_dirs = {}

    for subdir in test_results_dir.iterdir():
        if not subdir.is_dir():
            continue

        seed = extract_seed_from_dirname(subdir.name)
        if seed is None:
            continue

        if seed_range[0] <= seed <= seed_range[1]:
            if seed not in seed_dirs:
                seed_dirs[seed] = []
            seed_dirs[seed].append(subdir)

    return seed_dirs


def aggregate_all_data(test_results_dir: Path,
                       seed_range: tuple = (501, 520),
                       agent_id: str = "agent_0") -> tuple:
    """
    Aggregate all actions and observations from all seeds.

    Returns: (actions_df, observations_df)
    """
    all_actions = []
    all_observations = []

    seed_dirs = find_result_directories(test_results_dir, seed_range)

    print(f"Found {len(seed_dirs)} unique seeds in range {seed_range}")

    for seed in sorted(seed_dirs.keys()):
        dirs = seed_dirs[seed]
        print(f"\nProcessing seed {seed}: {len(dirs)} directories")

        # Use the most recent directory (last in sorted order)
        # Sort by directory name to get the most recent timestamp
        dirs_sorted = sorted(dirs, key=lambda x: x.name)
        selected_dir = dirs_sorted[-1]

        print(f"  Selected directory: {selected_dir.name}")

        # Find actions and observations files
        # Pattern: rl_ppo_by_effort_s{seed}_actions.jsonl
        actions_file = None
        observations_file = None

        for f in selected_dir.iterdir():
            if f.suffix == '.jsonl':
                if 'actions' in f.name:
                    actions_file = f
                elif 'observations' in f.name:
                    observations_file = f

        if actions_file is None:
            print(f"  WARNING: No actions file found in {selected_dir.name}")
            continue

        if observations_file is None:
            print(f"  WARNING: No observations file found in {selected_dir.name}")
            continue

        print(f"  Loading actions from: {actions_file.name}")
        actions_data = load_jsonl(actions_file)
        actions_records = process_actions(actions_data, seed, agent_id)
        all_actions.extend(actions_records)
        print(f"    Extracted {len(actions_records)} action records")

        print(f"  Loading observations from: {observations_file.name}")
        obs_data = load_jsonl(observations_file)
        obs_records = process_observations(obs_data, seed, agent_id)
        all_observations.extend(obs_records)
        print(f"    Extracted {len(obs_records)} observation records")

    # Convert to dataframes
    actions_df = pd.DataFrame(all_actions)
    observations_df = pd.DataFrame(all_observations)

    return actions_df, observations_df


def main():
    """Main execution function."""
    # Configuration
    TEST_RESULTS_DIR = Path("test_results")
    OUTPUT_DIR = Path("test_results")
    SEED_RANGE = (501, 520)
    AGENT_ID = "agent_0"

    print("="*80)
    print("AGGREGATING OBSERVATIONS AND ACTIONS FOR SIMILARITY ANALYSIS")
    print("="*80)
    print(f"Test results directory: {TEST_RESULTS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Seed range: {SEED_RANGE}")
    print(f"Agent ID: {AGENT_ID}")
    print()

    # Check if test_results exists
    if not TEST_RESULTS_DIR.exists():
        print(f"ERROR: Directory {TEST_RESULTS_DIR} does not exist!")
        return

    # Aggregate data
    print("Starting aggregation...")
    actions_df, observations_df = aggregate_all_data(
        TEST_RESULTS_DIR,
        SEED_RANGE,
        AGENT_ID
    )

    # Print summary
    print("\n" + "="*80)
    print("AGGREGATION SUMMARY")
    print("="*80)
    print(f"\nActions DataFrame:")
    print(f"  Total rows: {len(actions_df)}")
    print(f"  Columns: {list(actions_df.columns)}")
    print(f"  Unique seeds: {actions_df['seed'].nunique()}")
    print(f"  Seed values: {sorted(actions_df['seed'].unique())}")
    print(f"  Step range: {actions_df['step'].min()} - {actions_df['step'].max()}")

    print(f"\nObservations DataFrame:")
    print(f"  Total rows: {len(observations_df)}")
    print(f"  Columns: {list(observations_df.columns)}")
    print(f"  Unique seeds: {observations_df['seed'].nunique()}")
    print(f"  Seed values: {sorted(observations_df['seed'].unique())}")
    print(f"  Step range: {observations_df['step'].min()} - {observations_df['step'].max()}")

    # Save to parquet
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    actions_output = OUTPUT_DIR / "actions_all_seeds.parquet"
    observations_output = OUTPUT_DIR / "observations_all_seeds.parquet"

    print(f"\nSaving outputs...")
    print(f"  Actions -> {actions_output}")
    actions_df.to_parquet(actions_output, index=False)

    print(f"  Observations -> {observations_output}")
    observations_df.to_parquet(observations_output, index=False)

    print("\n" + "="*80)
    print("AGGREGATION COMPLETE!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - {actions_output}")
    print(f"  - {observations_output}")
    print(f"\nThese files can now be used in:")
    print(f"  notebooks/analyse_similarity_with_agent0_obs.ipynb")
    print()


if __name__ == "__main__":
    main()
