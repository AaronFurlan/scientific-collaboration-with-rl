import wandb
import argparse
import sys

def check_sweep(sweep_path):
    api = wandb.Api()
    try:
        sweep = api.sweep(sweep_path)
        print(f"Sweep: {sweep_path}")
        print(f"Name: {sweep.name}")
        print(f"State: {sweep.state}")
        
        if sweep.state.lower() != "running":
            print("\n!!! WARNUNG !!!")
            print(f"Der Sweep-Status ist '{sweep.state}'.")
            print("Ein WandB-Agent kann nur mit Sweeps verbunden werden, die den Status 'RUNNING' haben.")
            print("Bitte setze den Sweep in der WandB-Benutzeroberfläche wieder auf 'Running'.")
            print(f"Link: https://wandb.ai/{sweep_path}")
        else:
            print("\nDer Sweep läuft. Der Agent sollte sich verbinden können.")
            
    except Exception as e:
        print(f"Fehler beim Abrufen des Sweeps: {e}")
        print("Stelle sicher, dass der Pfad das Format 'entity/project/sweep_id' hat.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_path", help="Format: entity/project/sweep_id")
    args = parser.parse_args()
    check_sweep(args.sweep_path)
