import json
import pandas as pd
import os
import argparse
import re
from datetime import datetime
from tqdm import tqdm
import numpy as np

def find_latest_log_files(base_dir, strategy, seed):
    """Find latest actions and observations files for given strategy and seed."""
    pattern = re.compile(f"rl_ppo_{strategy}_s{seed}_(\\d{{8}}_\\d{{6}})_s{seed}")
    
    candidates = []
    if not os.path.exists(base_dir):
        return None, None

    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path):
            match = pattern.match(entry)
            if match:
                timestamp_str = match.group(1)
                try:
                    ts = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    candidates.append((ts, full_path))
                except ValueError:
                    continue
    
    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    latest_dir = candidates[0][1]
    
    actions_file = os.path.join(latest_dir, f"rl_ppo_{strategy}_s{seed}_actions.jsonl")
    obs_file = os.path.join(latest_dir, f"rl_ppo_{strategy}_s{seed}_observations.jsonl")
    
    if os.path.exists(actions_file) and os.path.exists(obs_file):
        return actions_file, obs_file
    
    return None, None

def flatten_observation(obs_dict):
    """Flatten observation structure for DataFrame format.
    Konzentriert sich auf skalare Werte und flacht Listen/Arrays ab.
    """
    if not obs_dict or not isinstance(obs_dict, dict):
        return {}
    
    flat = {}

    for k, v in obs_dict.items():
        if k == "observation" and isinstance(v, list):
            continue
        
        if isinstance(v, list) and len(v) == 1:
            flat[k] = v[0]
        elif isinstance(v, (int, float, str, bool)):
            flat[k] = v
    
    return flat

def build_combined_dataframe(observations, actions, seed, strategy):
    """Combine observations and actions into single DataFrame.
    Jede Zeile repräsentiert einen Agenten in einem Zeitschritt.
    """
    records = []
    
    # Anzahl der Schritte bestimmen (Minimum aus beiden, falls Diskrepanz)
    num_steps = min(len(observations), len(actions))
    
    for step_idx in range(num_steps):
        step_obs = observations[step_idx]
        step_act = actions[step_idx]
        
        # Wir gehen davon aus, dass beide Dicts die gleichen Agenten-IDs enthalten
        agents = set(step_obs.keys()) | set(step_act.keys())
        
        for agent_id in agents:
            obs_data = step_obs.get(agent_id)
            act_data = step_act.get(agent_id)
            
            # Nur aufzeichnen, wenn der Agent in diesem Schritt aktiv war (Daten vorhanden)
            if obs_data is None and act_data is None:
                continue
                
            record = {
                "step": step_idx,
                "agent_id": agent_id,
                "seed": seed,
                "strategy": strategy
            }
            
            # Observation hinzufügen
            if obs_data:
                flat_obs = flatten_observation(obs_data)
                record.update({f"obs_{k}": v for k, v in flat_obs.items()})
            
            # Action hinzufügen
            if act_data:
                # Actions sind oft flacher: {"choose_project": X, "put_effort": Y, "archetype": Z}
                for k, v in act_data.items():
                    if k == "archetype":
                        record["archetype"] = v
                    else:
                        record[f"act_{k}"] = v
            
            records.append(record)
            
    return pd.DataFrame(records)

def main():
    parser = argparse.ArgumentParser(description="Aggregiert RL-Aktionen und -Observationen in ein kombiniertes Parquet-File.")
    parser.add_argument("--log-dir", type=str, default="test_results", help="Pfad zum Log-Verzeichnis (default: 'test_results')")
    parser.add_argument("--out-base-dir", type=str, default="results", help="Basis-Ausgabeverzeichnis (default: 'results')")
    parser.add_argument("--start-seed", type=int, default=501, help="Start-Seed (default: 501)")
    parser.add_argument("--num-seeds", type=int, default=20, help="Anzahl der Seeds (default: 20)")
    
    args = parser.parse_args()

    seeds = range(args.start_seed, args.start_seed + args.num_seeds)
    strategies = ["multiply", "evenly", "by_effort"]
    
    log_dir = args.log_dir
    
    # Zeitstempel für den Ausgabeordner
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_base_dir, f"aggregation_combined_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Aggregration (Combined) gestartet. Ergebnisse unter: {out_dir}")

    for name in strategies:
        all_combined_data = []
        
        print(f"\nVerarbeite Strategie: {name}")
        
        for seed in tqdm(seeds, desc=f"Seeds ({name})", unit="seed"):
            actions_file, obs_file = find_latest_log_files(log_dir, name, seed)
            
            if not actions_file or not obs_file:
                continue
            
            try:
                with open(actions_file, "r") as f:
                    actions = [json.loads(line) for line in f]
                with open(obs_file, "r") as f:
                    observations = [json.loads(line) for line in f]
            except Exception as e:
                print(f"Fehler beim Lesen der Dateien für Seed {seed}: {e}")
                continue

            if not observations or not actions:
                continue

            df_combined = build_combined_dataframe(observations, actions, seed, name)
            if not df_combined.empty:
                all_combined_data.append(df_combined)
        
        if all_combined_data:
            df_final = pd.concat(all_combined_data, ignore_index=True)
            
            output_path = os.path.join(out_dir, f"rl_combined_obs_actions_{name}.parquet")
            df_final.to_parquet(output_path, index=False)
            
            print(f"  FERTIG: Kombiniertes File gespeichert unter {output_path}")
            print(f"  Datensätze: {len(df_final)}")
        else:
            print(f"  [Fehler] Keine Daten für Strategie {name} gefunden.")

    print(f"\nAlle Strategien verarbeitet. Finale Ergebnisse in: {out_dir}")

if __name__ == "__main__":
    main()
