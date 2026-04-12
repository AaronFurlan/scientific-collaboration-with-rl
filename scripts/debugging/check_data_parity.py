import pandas as pd
import os
import json
import hashlib

def get_file_hash(file_path):
    """Calculates SHA256 hash of a file to check for exact identity."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def check_parity():
    strategies = ["multiply", "evenly", "by_effort"]
    results_dir = "results"
    log_dir = "log"
    seeds = range(101, 111)
    
    data = {}
    
    print("--- Checking Parquet Files (Summary) ---")
    for strat in strategies:
        file_path = os.path.join(results_dir, f"rl_summary_by_archetype_{strat}.parquet")
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            reward_sum = df["mean_reward"].sum()
            record_count = len(df)
            data[strat] = (reward_sum, record_count)
            print(f"Strategy: {strat:10} | Sum of Rewards: {reward_sum:12.4f} | Records: {record_count}")
        else:
            print(f"Strategy: {strat:10} | FILE NOT FOUND: {file_path}")

    print("\n--- Comparison Results (Parquets) ---")
    identical_strats = []
    if len(data) >= 2:
        strats = list(data.keys())
        for i in range(len(strats)):
            for j in range(i + 1, len(strats)):
                s1, s2 = strats[i], strats[j]
                if data[s1] == data[s2]:
                    print(f"ALERT: Parquets for {s1} and {s2} are IDENTICAL!")
                    if s1 not in identical_strats: identical_strats.append(s1)
                    if s2 not in identical_strats: identical_strats.append(s2)
                else:
                    diff = abs(data[s1][0] - data[s2][0])
                    print(f"OK: Parquets for {s1} and {s2} are different (Diff: {diff:.4f})")
    
    if not identical_strats:
        print("\nConclusion: Parquet files are already different. No need to check logs.")
        return

    print("\n--- Checking Source JSONL Logs (Actions) ---")
    # We check seed 101 as a representative sample
    sample_seed = 101
    log_hashes = {}
    
    for strat in strategies:
        # Construct the same filename as process_rl_results.py expects
        log_file = os.path.join(log_dir, f"rl_ppo_{strat}_s{sample_seed}_actions.jsonl")
        if os.path.exists(log_file):
            h = get_file_hash(log_file)
            size = os.path.getsize(log_file)
            log_hashes[strat] = h
            print(f"Strategy: {strat:10} | Seed: {sample_seed} | Size: {size:10} bytes | Hash: {h[:16]}...")
        else:
            print(f"Strategy: {strat:10} | Seed: {sample_seed} | LOG FILE NOT FOUND: {log_file}")

    print("\n--- Comparison Results (Logs) ---")
    log_strats = list(log_hashes.keys())
    identical_logs = False
    for i in range(len(log_strats)):
        for j in range(i + 1, len(log_strats)):
            s1, s2 = log_strats[i], log_strats[j]
            if log_hashes[s1] == log_hashes[s2]:
                print(f"ALERT: Log files for {s1} and {s2} are BINARY IDENTICAL!")
                identical_logs = True
            else:
                print(f"OK: Log files for {s1} and {s2} are different.")

    if identical_logs:
        print("\nFINAL CONCLUSION: The simulation is generating identical log files.")
        print("This means the '--reward-function' parameter is likely NOT being applied correctly")
        print("in 'run_policy_simulation_with_rlagent.py' despite the fix, or the environment")
        print("ignores it during the run.")
    else:
        print("\nFINAL CONCLUSION: Logs are different, but Parquets are identical.")
        print("This points to an issue in 'process_rl_results.py' (perhaps it always reads the same file?).")

if __name__ == "__main__":
    check_parity()
