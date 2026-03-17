import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Füge das aktuelle Verzeichnis zum Pfad hinzu, damit train_ppo_rllib importiert werden kann
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_ppo_rllib import main

class TestTrainPPORllibGuard(unittest.TestCase):
    
    @patch('train_ppo_rllib.ray.init')
    @patch('train_ppo_rllib.tune.register_env')
    @patch('train_ppo_rllib.PPOConfig')
    @patch('train_ppo_rllib.seed_everything')
    @patch('train_ppo_rllib.torch.use_deterministic_algorithms')
    def test_guard_fails_when_group_size_too_large(self, mock_det, mock_seed, mock_ppo, mock_reg, mock_ray):
        """Testet, ob der Guard einen ValueError wirft, wenn n_agents / n_groups > max_peer_group_size."""
        
        # Konfiguration, die fehlschlagen sollte: 2000 / 20 = 100 > 13
        params = {
            "iterations": 1,
            "framework": "torch",
            "policy_config_name": "Balanced",
            "group_policy_homogenous": False,
            "seed": 42,
            "n_agents": 2000,
            "start_agents": 200,
            "max_steps": 600,
            "max_rewardless_steps": 500,
            "n_groups": 20,
            "max_peer_group_size": 13,
            "n_projects_per_step": 1,
            "max_projects_per_agent": 6,
            "max_agent_age": 750,
            "acceptance_threshold": 0.44,
            "reward_function": "by_effort",
            "prestige_threshold": 0.2,
            "novelty_threshold": 0.8,
            "effort_threshold": 22,
            "controlled_agent_id": "agent_0",
            "wandb_mode": "disabled"
        }
        
        with self.assertRaisesRegex(ValueError, "CONFIGURATION ERROR"):
            main(**params)

    @patch('train_ppo_rllib.ray.init')
    @patch('train_ppo_rllib.tune.register_env')
    @patch('train_ppo_rllib.PPOConfig')
    @patch('train_ppo_rllib.seed_everything')
    @patch('train_ppo_rllib.torch.use_deterministic_algorithms')
    def test_guard_passes_when_group_size_is_ok(self, mock_det, mock_seed, mock_ppo, mock_reg, mock_ray):
        """Testet, ob der Guard bei einer validen Konfiguration nicht abbricht."""
        
        # Mocking der Config-Kette, damit main() nicht wirklich versucht zu trainieren
        mock_ppo_instance = MagicMock()
        mock_ppo.return_value = mock_ppo_instance
        mock_ppo_instance.api_stack.return_value = mock_ppo_instance
        mock_ppo_instance.debugging.return_value = mock_ppo_instance
        mock_ppo_instance.environment.return_value = mock_ppo_instance
        mock_ppo_instance.framework.return_value = mock_ppo_instance
        mock_ppo_instance.training.return_value = mock_ppo_instance
        mock_ppo_instance.env_runners.return_value = mock_ppo_instance
        mock_ppo_instance.callbacks.return_value = mock_ppo_instance
        mock_ppo_instance.evaluation.return_value = mock_ppo_instance
        
        # Wir stoppen den Test vor dem Aufruf von config.build_algo()
        mock_ppo_instance.build_algo.side_effect = Exception("Stopped after guard check")
        
        # Konfiguration, die ok sein sollte: 100 / 10 = 10 <= 13
        params = {
            "iterations": 1,
            "framework": "torch",
            "policy_config_name": "Balanced",
            "group_policy_homogenous": False,
            "seed": 42,
            "n_agents": 100,
            "start_agents": 20,
            "max_steps": 100,
            "max_rewardless_steps": 50,
            "n_groups": 10,
            "max_peer_group_size": 13,
            "n_projects_per_step": 1,
            "max_projects_per_agent": 6,
            "max_agent_age": 750,
            "acceptance_threshold": 0.44,
            "reward_function": "by_effort",
            "prestige_threshold": 0.2,
            "novelty_threshold": 0.8,
            "effort_threshold": 22,
            "controlled_agent_id": "agent_0",
            "wandb_mode": "disabled"
        }
        
        try:
            main(**params)
        except Exception as e:
            if str(e) == "Stopped after guard check":
                # Guard wurde passiert, Test erfolgreich
                pass
            else:
                self.fail(f"main() raised unexpected exception: {e}")

    @patch('train_ppo_rllib.seed_everything')
    def test_guard_fails_when_max_peer_group_size_too_large_for_macro_action(self, mock_seed):
        """Testet, ob der Guard einen ValueError wirft, wenn max_peer_group_size > 16."""
        
        params = {
            "iterations": 1,
            "framework": "torch",
            "policy_config_name": "Balanced",
            "group_policy_homogenous": False,
            "seed": 42,
            "n_agents": 100,
            "start_agents": 20,
            "max_steps": 100,
            "max_rewardless_steps": 50,
            "n_groups": 10,
            "max_peer_group_size": 17, # > 16
            "n_projects_per_step": 1,
            "max_projects_per_agent": 6,
            "max_agent_age": 750,
            "acceptance_threshold": 0.44,
            "reward_function": "by_effort",
            "prestige_threshold": 0.2,
            "novelty_threshold": 0.8,
            "effort_threshold": 22,
            "controlled_agent_id": "agent_0",
            "wandb_mode": "disabled"
        }
        
        with self.assertRaisesRegex(ValueError, "too large for the current macro-action wrapper approach"):
            main(**params)

if __name__ == "__main__":
    unittest.main()
