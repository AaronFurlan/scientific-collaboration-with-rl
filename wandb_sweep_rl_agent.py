import wandb
import os
import sys
import argparse
from train_rl_agent import main as train_main
import ray

def sweep_train():
    """
    Diese Funktion wird von wandb.agent() für jeden Run aufgerufen.
    Sie liest die Hyperparameter aus wandb.config und übergibt sie an das Hauptskript.
    """
    # 1. Initialisiere WandB für diesen spezifischen Run des Sweeps
    # Das Hauptskript train_rl_agent.py ruft intern wandb.init() auf, 
    # aber wenn wir es im Sweep-Modus betreiben, übernimmt wandb.agent() das Setup.
    # Wir übergeben wandb_mode="online" an train_main, damit es dort korrekt geloggt wird.
    
    with wandb.init() as run:
        config = run.config
        
        print(f"\n=== Starting Sweep Run: {run.name}")
        print(f"===Config: {config}\n")
        
        # 2. Standardparameter (Base) definieren
        # Diese Werte werden verwendet, wenn sie nicht im Sweep definiert sind.
        params = {
            "algo": "PPO",
            "iterations": 25,  # Genug Iterationen für Konvergenz-Tendenzen
            "framework": "torch",
            "policy_config_name": "Balanced",
            "group_policy_homogenous": False,
            "seed": 42,
            "n_agents": 100,
            "start_agents": 50,
            "max_steps": 100,
            "max_rewardless_steps": 100,
            "n_groups": 20,
            "max_peer_group_size": 20,
            "n_projects_per_step": 1,
            "max_projects_per_agent": 6,
            "max_agent_age": 750,
            "acceptance_threshold": 0.44,
            "reward_function": "by_effort",
            "prestige_threshold": 0.2,
            "novelty_threshold": 0.8,
            "effort_threshold": 22,
            "controlled_agent_id": "agent_0",
            "wandb_mode": "online",
            "train_batch_size": 2000,
            "num_workers": 10, # Reduziert für Sweeps (je nach CPU-Kernen)
            "evaluation_interval": 0,
            "save_every_n_iters": 0, # Keine Checkpoints während Sweeps (spart Platz)
            # RL training hyperparameters
            "gamma": 0.99,
            "lambda_": 0.95,
            "lr": 1e-4,
            "num_epochs": 8,
            "entropy_coeff": 0.01,
            "vf_loss_coeff": 1.0,
            "grad_clip": 0.5,
        }
        
        # 3. Parameter aus dem Sweep-Config überschreiben
        for key in config.keys():
            if key in params:
                params[key] = config[key]
        
        # 4. Training starten
        try:
            train_main(**params)
        except Exception as e:
            print(f"Error in sweep run {run.name}: {e}")
            # Ray Shutdown sicherstellen, falls ein Crash passiert
            if ray.is_initialized():
                ray.shutdown()
            raise e

def create_sweep_config(project_name="game-of-science-sweeps"):
    """
    Erstellt die Konfiguration für den WandB Sweep.
    """
    sweep_configuration = {
        "method": "grid",  # 'grid', 'random', oder 'bayes'
        "name": "RL Hyperparameter Search",
        "metric": {
            "name": "eval/episode_return_mean",
            "goal": "maximize"
        },
        "parameters": {
            "lr": {
                "values": [1e-4, 5e-5, 3e-4]
            },
            "entropy_coeff": {
                "values": [0.001, 0.01, 0.05]
            },
            "gamma": {
                "values": [0.95, 0.99]
            },
            "lambda_": {
                "values": [0.9, 0.95, 1.0]
            },
            "num_epochs": {
                "values": [5, 10, 15]
            },
            "train_batch_size": {
                "values": [4000, 8000, 16000]
            }
        }
    }
    
    sweep_id = wandb.sweep(sweep_configuration, project=project_name)
    return sweep_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WandB Sweep Wrapper für RL Training")
    parser.add_argument("--sweep_id", type=str, default=None, help="Existierende Sweep ID")
    parser.add_argument("--project", type=str, default="game-of-science-sweeps")
    parser.add_argument("--count", type=int, default=None, help="Anzahl der Runs (nur für random/bayes)")
    
    args = parser.parse_args()
    
    if args.sweep_id is None:
        # Erstelle neuen Sweep
        print("Creating new sweep...")
        sweep_id = create_sweep_config(project_name=args.project)
        print(f"Sweep created with ID: {sweep_id}")
    else:
        sweep_id = args.sweep_id
        print(f"Using existing sweep ID: {sweep_id}")
    
    # Starte den Agenten
    print(f"Starting wandb agent for sweep {sweep_id}...")
    wandb.agent(sweep_id, function=sweep_train, count=args.count, project=args.project)
