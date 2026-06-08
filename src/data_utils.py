"""
Data utility functions for saving and loading DataFrames.
"""
import os
from datetime import datetime
import pandas as pd
from typing import Optional


def save_df_to_parquet(
    df: pd.DataFrame,
    filename: str,
    output_dir: str = "results",
    timestamp: bool = True,
    create_dir: bool = True,
    use_project_root: bool = True,
) -> str:
    """
    Save DataFrame to Parquet file with optional timestamp.

    Args:
        df: DataFrame to save
        filename: Base filename (without extension)
        output_dir: Output directory (default: "results")
        timestamp: Add timestamp to filename
        use_project_root: Save relative to project root vs current dir

    Returns:
        Full path to saved file
    """
    # Remove .parquet extension if user provided it
    filename = filename.replace(".parquet", "")

    # Add timestamp if requested
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename}_{ts}"

    # Add extension
    filename = f"{filename}.parquet"

    # Handle output directory
    if use_project_root and not os.path.isabs(output_dir):
        # Find project root (directory containing src/)
        current_dir = os.path.abspath(os.getcwd())

        # Try to find project root by looking for src/ directory
        project_root = current_dir
        while project_root != os.path.dirname(project_root):  # Not at filesystem root
            if os.path.exists(os.path.join(project_root, "src")):
                break
            project_root = os.path.dirname(project_root)

        # If we found project root, use it
        if os.path.exists(os.path.join(project_root, "src")):
            output_dir = os.path.join(project_root, output_dir)
        # Otherwise, use current directory (fallback)

    # Create full path
    filepath = os.path.abspath(os.path.join(output_dir, filename))

    # Create directory if needed
    if create_dir:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save DataFrame
    df.to_parquet(filepath, index=False)

    print(f"Saved DataFrame to: {filepath}")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Size: {os.path.getsize(filepath) / 1024:.2f} KB")

    return filepath


def load_df_from_parquet(filepath: str) -> pd.DataFrame:
    """Load DataFrame from Parquet file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_parquet(filepath)

    print(f"Loaded DataFrame from: {filepath}")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    return df


def save_multiple_dfs(
    dfs: dict[str, pd.DataFrame],
    output_dir: str = "results",
    timestamp: bool = True,
) -> dict[str, str]:
    """Save multiple DataFrames to Parquet files."""
    paths = {}

    for name, df in dfs.items():
        path = save_df_to_parquet(
            df=df,
            filename=name,
            output_dir=output_dir,
            timestamp=timestamp,
            create_dir=True,
        )
        paths[name] = path

    return paths
