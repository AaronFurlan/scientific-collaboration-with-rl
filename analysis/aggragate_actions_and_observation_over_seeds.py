import json
import re
import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def extract_seed(path: Path) -> int | None:
    """
    Extract seed number from run folder name.

    Patterns supported:
    - rl_ppo_by_effort_s501_...
    - rl_ppo_multiply_s502_...

    Args:
        path: Path to the run folder or file

    Returns:
        Seed as integer, or None if not found
    """
    # Look in the parent folder name (run_id)
    folder_name = path.parent.name if path.is_file() else path.name

    # Pattern: s followed by digits
    match = re.search(r's(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return None


def extract_timestamp(run_id: str) -> str | None:
    """
    Extract timestamp from run folder name.

    Pattern: YYYYMMDD_HHMMSS

    Args:
        run_id: Run folder name

    Returns:
        Timestamp string or None if not found
    """
    match = re.search(r'(\d{8}_\d{6})', run_id)
    if match:
        return match.group(1)
    return None


def load_json_records(path: Path) -> list[dict]:
    """
    Load JSON records from a file.

    Supports:
    - Normal JSON array: [{"a": 1}, {"b": 2}]
    - JSON object with list: {"actions": [...]} or {"observations": [...]}
    - JSONL fallback: one JSON object per line

    Args:
        path: Path to JSON file

    Returns:
        List of record dictionaries
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Try parsing as normal JSON
        try:
            data = json.loads(content)

            # If it's already a list, return it
            if isinstance(data, list):
                return data

            # If it's a dict, look for common keys
            if isinstance(data, dict):
                # Try common keys
                for key in ['actions', 'observations', 'data', 'records']:
                    if key in data and isinstance(data[key], list):
                        return data[key]

                # If single object, wrap in list
                return [data]

        except json.JSONDecodeError:
            # Fall back to JSONL
            lines = content.strip().split('\n')
            records = []
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return records

    except Exception as e:
        print(f"  Warning: Failed to load {path}: {e}")
        return []


def normalize_records(records: list[dict], metadata: dict) -> pd.DataFrame:
    """
    Normalize records into a DataFrame with metadata columns.

    Args:
        records: List of record dictionaries
        metadata: Dict with run_id, seed, source_file, source_type

    Returns:
        DataFrame with flattened records and metadata
    """
    if not records:
        return pd.DataFrame()

    # Try to normalize nested structures
    try:
        df = pd.json_normalize(records, max_level=2)
    except Exception:
        # If normalization fails, convert to DataFrame directly
        df = pd.DataFrame(records)

    # Convert columns with complex objects to JSON strings
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if any value is dict or list
            sample = df[col].dropna().head(1)
            if len(sample) > 0:
                val = sample.iloc[0]
                if isinstance(val, (dict, list)):
                    df[col] = df[col].apply(
                        lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
                    )

    # Add metadata columns
    for key, value in metadata.items():
        df[key] = value

    return df


def aggregate_files(
    input_dir: Path,
    pattern: str,
    source_type: str,
    seed_start: int,
    seed_end: int
) -> pd.DataFrame:
    """
    Aggregate all files matching pattern within seed range.

    Args:
        input_dir: Directory to search
        pattern: Glob pattern (e.g., "*_actions.json")
        source_type: "actions" or "observations"
        seed_start: Minimum seed (inclusive)
        seed_end: Maximum seed (inclusive)

    Returns:
        Aggregated DataFrame
    """
    all_dfs = []
    found_files = []
    skipped_files = []

    # Recursively find all matching files
    for file_path in input_dir.rglob(pattern):
        # Extract seed
        seed = extract_seed(file_path)

        # Filter by seed range
        if seed is None:
            skipped_files.append((str(file_path), "No seed found"))
            continue

        if seed < seed_start or seed > seed_end:
            skipped_files.append((str(file_path), f"Seed {seed} out of range"))
            continue

        # Extract run metadata
        run_id = file_path.parent.name
        timestamp = extract_timestamp(run_id)

        # Load records
        records = load_json_records(file_path)

        if not records:
            skipped_files.append((str(file_path), "No records loaded"))
            continue

        # Prepare metadata
        metadata = {
            'run_id': run_id,
            'seed': seed,
            'source_file': str(file_path.relative_to(input_dir)),
            'source_type': source_type,
        }

        if timestamp:
            metadata['timestamp'] = timestamp

        # Normalize and add to collection
        df = normalize_records(records, metadata)

        if not df.empty:
            all_dfs.append(df)
            found_files.append((str(file_path), seed, len(df)))

    # Print summary for this source type
    print(f"\n{source_type.capitalize()} files:")
    print(f"  Found: {len(found_files)}")
    if found_files:
        seeds_included = sorted(set(seed for _, seed, _ in found_files))
        print(f"  Seeds: {seeds_included}")
        total_rows = sum(rows for _, _, rows in found_files)
        print(f"  Total rows: {total_rows:,}")

    if skipped_files:
        print(f"  Skipped: {len(skipped_files)}")
        for path, reason in skipped_files[:5]:  # Show first 5
            print(f"    - {Path(path).name}: {reason}")
        if len(skipped_files) > 5:
            print(f"    ... and {len(skipped_files) - 5} more")

    # Combine all DataFrames
    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Sort by relevant columns
    sort_cols = ['seed', 'run_id']

    # Add step-like columns if available
    for step_col in ['step', 'timestep', 'env_step', 'episode']:
        if step_col in combined.columns:
            sort_cols.append(step_col)
            break

    # Add agent_id if available
    if 'agent_id' in combined.columns:
        sort_cols.append('agent_id')

    combined = combined.sort_values(by=sort_cols, ignore_index=True)

    return combined


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Aggregate RL evaluation JSON files into parquet format."
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='test_results',
        help='Input directory to scan (default: test_results)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='test_results/aggregated',
        help='Output directory for parquet files (default: test_results/aggregated)'
    )
    parser.add_argument(
        '--seed-start',
        type=int,
        default=501,
        help='Minimum seed to include (default: 501)'
    )
    parser.add_argument(
        '--seed-end',
        type=int,
        default=520,
        help='Maximum seed to include (default: 520)'
    )

    args = parser.parse_args()

    # Convert to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Aggregating RL evaluation files")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Seed range: {args.seed_start} - {args.seed_end}")

    # Aggregate actions (support both .json and .jsonl)
    actions_df = aggregate_files(
        input_dir=input_dir,
        pattern='*_actions.json*',
        source_type='actions',
        seed_start=args.seed_start,
        seed_end=args.seed_end
    )

    # Aggregate observations (support both .json and .jsonl)
    observations_df = aggregate_files(
        input_dir=input_dir,
        pattern='*_observations.json*',
        source_type='observations',
        seed_start=args.seed_start,
        seed_end=args.seed_end
    )

    # Write output files
    print("\nWriting output files:")

    if not actions_df.empty:
        actions_file = output_dir / f'actions_{args.seed_start}_{args.seed_end}.parquet'
        actions_df.to_parquet(actions_file, index=False)
        print(f"  ✓ {actions_file}")
        print(f"    Rows: {len(actions_df):,}")
        print(f"    Columns: {len(actions_df.columns)}")
    else:
        print("  ⚠ No actions data to write")

    if not observations_df.empty:
        obs_file = output_dir / f'observations_{args.seed_start}_{args.seed_end}.parquet'
        observations_df.to_parquet(obs_file, index=False)
        print(f"  ✓ {obs_file}")
        print(f"    Rows: {len(observations_df):,}")
        print(f"    Columns: {len(observations_df.columns)}")
    else:
        print("  ⚠ No observations data to write")

    print("\nAggregation complete!")


if __name__ == '__main__':
    main()
