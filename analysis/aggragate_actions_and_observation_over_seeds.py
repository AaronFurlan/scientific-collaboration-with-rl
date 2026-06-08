import json
import re
import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def extract_seed(path: Path) -> int | None:
    """Extract seed number from run folder name (e.g., s501)."""
    folder_name = path.parent.name if path.is_file() else path.name
    match = re.search(r's(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return None


def extract_timestamp(run_id: str) -> str | None:
    """Extract timestamp from run folder name (YYYYMMDD_HHMMSS)."""
    match = re.search(r'(\d{8}_\d{6})', run_id)
    if match:
        return match.group(1)
    return None


def load_json_records(path: Path) -> list[dict]:
    """Load JSON records from file (supports array or object with list).
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


def normalize_records(records: list[dict], metadata: dict, controlled_agent: str = 'agent_0') -> pd.DataFrame:
    """
    Extract controlled agent data from records.

    Converts agent-keyed dictionaries, extracting only the controlled agent:
    {step: 0, agent_0: {...}, agent_1: {...}}
    → [(step=0, agent_id=agent_0, data={...})]

    Args:
        records: List of record dictionaries
        metadata: Dict with run_id, seed, source_file, source_type
        controlled_agent: Agent ID to extract (default: 'agent_0')

    Returns:
        DataFrame with controlled agent data only
    """
    if not records:
        return pd.DataFrame()

    rows = []

    for record in records:
        if not isinstance(record, dict):
            continue

        # Extract step/timestep if present
        step = record.get('step', record.get('timestep', None))

        # Check if controlled agent exists in this record
        if controlled_agent in record:
            agent_data = record[controlled_agent]

            # Skip if agent is inactive (None)
            if agent_data is None:
                continue

            row = {
                'agent_id': controlled_agent,
                'step': step,
            }

            # Add agent-specific data
            if isinstance(agent_data, dict):
                # Flatten nested dicts with limited depth
                for key, val in agent_data.items():
                    if isinstance(val, (dict, list)):
                        # Store complex objects as JSON strings
                        row[key] = json.dumps(val)
                    else:
                        row[key] = val
            else:
                # Scalar value (shouldn't happen, but handle it)
                row['data'] = agent_data

            # Add metadata
            for key, value in metadata.items():
                row[key] = value

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def aggregate_files(
    input_dir: Path,
    pattern: str,
    source_type: str,
    seed_start: int,
    seed_end: int,
    output_file: Path,
    batch_size: int = 10
) -> dict:
    """
    Aggregate all files matching pattern within seed range.
    Memory-efficient version that writes directly to parquet in batches.

    Args:
        input_dir: Directory to search
        pattern: Glob pattern (e.g., "*_actions.json")
        source_type: "actions" or "observations"
        seed_start: Minimum seed (inclusive)
        seed_end: Maximum seed (inclusive)
        output_file: Path to output parquet file
        batch_size: Number of files to process before writing to disk (default: 10)

    Returns:
        Dictionary with statistics
    """
    from tqdm import tqdm

    found_files = []
    skipped_files = []

    # First pass: collect all file paths
    print(f"\nScanning for {source_type} files...")
    all_file_paths = []
    for file_path in input_dir.rglob(pattern):
        seed = extract_seed(file_path)

        if seed is None:
            skipped_files.append((str(file_path), "No seed found"))
            continue

        if seed < seed_start or seed > seed_end:
            skipped_files.append((str(file_path), f"Seed {seed} out of range"))
            continue

        all_file_paths.append((file_path, seed))

    print(f"  Found {len(all_file_paths)} files to process")

    if not all_file_paths:
        return {'found': 0, 'skipped': len(skipped_files), 'total_rows': 0}

    # Process files in batches
    writer = None
    schema = None
    total_rows = 0
    batch_dfs = []

    for idx, (file_path, seed) in enumerate(tqdm(all_file_paths, desc=f"Processing {source_type}")):
        try:
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

            # Normalize
            df = normalize_records(records, metadata)

            if not df.empty:
                batch_dfs.append(df)
                found_files.append((str(file_path), seed, len(df)))
                total_rows += len(df)

                # Write batch to disk when batch is full or at the end
                if len(batch_dfs) >= batch_size or idx == len(all_file_paths) - 1:
                    batch_combined = pd.concat(batch_dfs, ignore_index=True)

                    # Convert to PyArrow table
                    table = pa.Table.from_pandas(batch_combined)

                    # Initialize writer on first batch
                    if writer is None:
                        schema = table.schema
                        writer = pq.ParquetWriter(output_file, schema)

                    # Write batch
                    writer.write_table(table)

                    # Clear batch from memory
                    batch_dfs.clear()
                    del batch_combined
                    del table

        except Exception as e:
            skipped_files.append((str(file_path), f"Error: {e}"))
            continue

    # Close writer
    if writer is not None:
        writer.close()

    # Print summary
    print(f"\n{source_type.capitalize()} summary:")
    print(f"  Processed: {len(found_files)} files")
    if found_files:
        seeds_included = sorted(set(seed for _, seed, _ in found_files))
        print(f"  Seeds: {len(seeds_included)} unique ({min(seeds_included)} - {max(seeds_included)})")
        print(f"  Total rows: {total_rows:,}")

    if skipped_files:
        print(f"  Skipped: {len(skipped_files)} files")
        if len(skipped_files) <= 5:
            for path, reason in skipped_files:
                print(f"    - {Path(path).name}: {reason}")
        else:
            for path, reason in skipped_files[:3]:
                print(f"    - {Path(path).name}: {reason}")
            print(f"    ... and {len(skipped_files) - 3} more")

    return {
        'found': len(found_files),
        'skipped': len(skipped_files),
        'total_rows': total_rows,
        'seeds': len(set(seed for _, seed, _ in found_files)),
    }


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

    print(f"Aggregating RL evaluation files (Memory-efficient mode)")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Seed range: {args.seed_start} - {args.seed_end}")

    # Define output files
    actions_file = output_dir / 'actions_all_seeds.parquet'
    obs_file = output_dir / 'observations_all_seeds.parquet'

    # Aggregate actions (support both .json and .jsonl)
    # Writes directly to parquet file in batches
    actions_stats = aggregate_files(
        input_dir=input_dir,
        pattern='*_actions.json*',
        source_type='actions',
        seed_start=args.seed_start,
        seed_end=args.seed_end,
        output_file=actions_file,
        batch_size=10  # Process 10 files at a time
    )

    # Aggregate observations (support both .json and .jsonl)
    obs_stats = aggregate_files(
        input_dir=input_dir,
        pattern='*_observations.json*',
        source_type='observations',
        seed_start=args.seed_start,
        seed_end=args.seed_end,
        output_file=obs_file,
        batch_size=10  # Process 10 files at a time
    )

    # Print final summary
    print("\n" + "="*60)
    print("AGGREGATION COMPLETE")
    print("="*60)

    if actions_stats['found'] > 0:
        print(f"\n✓ Actions: {actions_file}")
        print(f"    Files processed: {actions_stats['found']}")
        print(f"    Total rows: {actions_stats['total_rows']:,}")
        print(f"    Unique seeds: {actions_stats['seeds']}")
    else:
        print("\n⚠ No actions data written")

    if obs_stats['found'] > 0:
        print(f"\n✓ Observations: {obs_file}")
        print(f"    Files processed: {obs_stats['found']}")
        print(f"    Total rows: {obs_stats['total_rows']:,}")
        print(f"    Unique seeds: {obs_stats['seeds']}")
    else:
        print("\n⚠ No observations data written")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
