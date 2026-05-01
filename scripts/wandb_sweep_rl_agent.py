import wandb
import os
import sys
import argparse

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.train_rl_agent import main as train_main
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
    
    # Erhöht den HTTP-Timeout für GraphQL-Anfragen (Standard ist oft zu niedrig)
    os.environ["WANDB_HTTP_TIMEOUT"] = "180"

    with wandb.init() as run:
        config = run.config
        
        print(f"\n=== Starting Sweep Run: {run.name}")
        print(f"===Config: {config}\n")
        
        # 2. Standardparameter (Base) definieren
        # Diese Werte werden verwendet, wenn sie nicht im Sweep definiert sind.
        params = {
            "algo": "PPO",
            "iterations": 20,
            "framework": "torch",
            "policy_config_name": "Balanced",
            "group_policy_homogenous": False,
            "seed": 42,
            "n_agents": 400,
            "start_agents": 100,
            "max_steps": 600,
            "max_rewardless_steps": 50,
            "n_groups": 10,
            "max_peer_group_size": 40,
            "n_projects_per_step": 1,
            "max_projects_per_agent": 8,
            "max_agent_age": 750,
            "acceptance_threshold": 0.44,
            "reward_function": "by_effort",
            "prestige_threshold": 0.4,
            "novelty_threshold": 0.4,
            "effort_threshold": 38,
            "controlled_agent_id": "agent_0",
            "wandb_mode": "online",
            "train_batch_size": 2000,
            "num_workers": 6,
            "num_envs_per_worker": 1,
            "rollout_fragment_length": 200,
            "evaluation_interval": 0,
            "save_every_n_iters": 0,
            # RL training hyperparameters
            "gamma": 0.99,
            "lambda_": 0.95,
            "lr": 1e-4,
            "num_epochs": 8,
            "entropy_coeff": 0.01,
            "vf_loss_coeff": 1.0,
            "grad_clip": 0.5,
            "vf_share_layers": True,
            "total_env_steps": 300_000,
        }
        
        # 3. Parameter aus dem Sweep-Config überschreiben
        for key in config.keys():
            if key in params:
                params[key] = config[key]
        
        # 4. Training starten
        try:
            # We must ensure that iterations is large enough if total_env_steps is used.
            # If total_env_steps is set, we use it as the main stopping criterion.
            # We set iterations to a very large number so it doesn't stop prematurely,
            # unless the user explicitly provided it in the sweep.
            if "total_env_steps" in params and "iterations" not in config:
                params["iterations"] = 100000 
            
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
        "method": "bayes",  # Bayesianische Optimierung statt Grid Search
        "name": "RL Bayesian Hyperparameter Search Expanded",
        "metric": {
            "name": "train/episode_return_mean",
            "goal": "maximize"
        },
        "parameters": {
            "lr": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 5e-4
            },
            "entropy_coeff": {
                "distribution": "log_uniform_values",
                "min": 0.005,
                "max": 0.008
            },
            "vf_loss_coeff": {
                "distribution": "uniform",
                "min": 1.9,
                "max": 2.0
            },
            "grad_clip": {
                "distribution": "uniform",
                "min": 0.50,
                "max": 0.54
            },
            "vf_share_layers": {
                "values": [True]
            },
            "gamma": {
                "distribution": "uniform",
                "min": 0.958,
                "max": 0.96
            },
            "lambda_": {
                "distribution": "uniform",
                "min": 0.959,
                "max": 0.963
            },
            "num_epochs": {
                "values": [3, 4]
            },
            "train_batch_size": {
                "values": [8000, 10000, 12000]
            },
            "total_env_steps": {
                "value": 300000
            }
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 5,
            "eta": 2
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
    
    # Check if sweep_id is just an ID or a full path
    sweep_path = sweep_id
    if "/" not in sweep_id:
        sweep_path = f"{args.project}/{sweep_id}"
    
    api = wandb.Api()
    try:
        sweep = api.sweep(sweep_path)
        state = sweep.state.lower()

        if state in {"canceled", "cancelled", "stopped", "finished", "failed", "crashed"}:
            print(f"\n!!! ERROR: Sweep state is '{sweep.state}' !!!")
            if state in {"canceled", "cancelled"}:
                print("Dieser Sweep wurde endgültig abgebrochen und kann nicht fortgesetzt werden.")
                print("Bitte erstelle einen neuen Sweep, indem du das Skript ohne --sweep_id startest.")
            else:
                print("Dieser Sweep akzeptiert keine neuen Runs mehr.")
            sys.exit(1)

        if state in {"pending", "running"}:
            print(f"Sweep state is '{sweep.state}'. Agent can start.")
        else:
            print(f"Unbekannter Sweep-State '{sweep.state}'. Versuche trotzdem den Agent zu starten...")

    except Exception as e:
        print(f"Could not verify sweep status: {e}. Proceeding anyway...")

    wandb.agent(sweep_id, function=sweep_train, count=args.count, project=args.project)
