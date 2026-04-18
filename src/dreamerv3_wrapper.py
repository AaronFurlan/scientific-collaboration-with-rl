import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete
from typing import Any, Callable, Dict, Optional, Union, List

class DreamerV3SingleAgentWrapper(gym.Env):
    """
    Spezialisierter Single-Agent Wrapper für DreamerV3.
    
    Ziele:
    1. Konvertierung des komplexen Action-Dict-Spaces in einen einzelnen Discrete(N) Space.
    2. Flache np.float32-Observation für DreamerV3 (Vermeidung von Dict-Spaces).
    3. Robuste Dekodierung der diskreten Aktion in strukturierte Aktionen für die Umgebung.
    """

    def __init__(
        self,
        env: gym.Env,
        controlled_agent: str = "agent_0",
        other_policies: Optional[Dict[str, Callable[[Any], Any]]] = None,
        max_peer_group_size: int = 10,
        debug_effort: bool = False,
    ):
        super().__init__()
        self.env = env
        self.controlled_agent = controlled_agent
        self.other_policies = other_policies or {}
        self.max_peer_group_size = max_peer_group_size
        self.debug_effort = debug_effort

        # Die zugrunde liegende Umgebung liefert bestimmte Limits
        # Wir fixieren max_peer_group_size=10 wie angefordert.
        # Action Space Komponenten aus der Umgebung:
        # - choose_project: Discrete(n_projects_per_step + 1) -> meist 2
        # - put_effort: Discrete(max_projects_per_agent + 1) -> meist 9
        # - collaborate_with: MultiBinary(max_peer_group_size) -> 2^10 = 1024
        
        # Annahme: n_projects_per_step = 1, max_projects_per_agent = 8
        self.n_choose = 2 # 0: kein Projekt, 1: Projekt 0
        self.n_effort = 9 # 0: kein Effort, 1-8: Effort in Projekt i-1
        self.n_collab = 2 ** self.max_peer_group_size # 1024
        
        self.total_actions = self.n_choose * self.n_effort * self.n_collab
        self.action_space = Discrete(self.total_actions)

        # Observation Space Definition (Flach, float32)
        # Wir berechnen die Größe basierend auf der ursprünglichen Observation
        sample_obs = self.env.observation_space(self.controlled_agent)
        self.flat_obs_dim = self._calculate_flat_obs_dim(sample_obs)
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.flat_obs_dim,), 
            dtype=np.float32
        )

        self._last_obs = None

    def _calculate_flat_obs_dim(self, space: gym.Space) -> int:
        if isinstance(space, gym.spaces.Dict):
            return sum(self._calculate_flat_obs_dim(s) for s in space.values())
        elif isinstance(space, Box):
            return int(np.prod(space.shape))
        elif isinstance(space, Discrete):
            return 1
        elif isinstance(space, gym.spaces.MultiBinary):
            return int(np.prod(space.shape))
        else:
            # Fallback für andere Spaces, falls vorhanden
            return int(np.prod(space.shape)) if hasattr(space, 'shape') else 1

    def _flatten_observation(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Konvertiert die verschachtelte Dict-Observation in einen flachen float32-Vektor.
        Stellt sicher, dass die Länge konsistent mit dem Observation Space ist,
        indem fehlende Projekte (opportunities/running) mit Nullen aufgefüllt werden.
        """
        # Wir nutzen die rekursive Padding-Logik, die den Observation Space als Template verwendet.
        # Dies ist robust gegenüber fehlenden Keys in den Projekt-Dictionaries.
        return self._flatten_with_padding(obs, self.env.observation_space(self.controlled_agent))

    def _flatten_with_padding(self, data: Any, space: gym.Space) -> np.ndarray:
        if isinstance(space, gym.spaces.Dict):
            parts = []
            # Sortierte Keys für Konsistenz
            for key in sorted(space.spaces.keys()):
                space_child = space.spaces[key]
                if key in data:
                    parts.append(self._flatten_with_padding(data[key], space_child))
                else:
                    # Fehlendes Element im Dict -> mit Nullen der passenden Größe auffüllen
                    size = self._calculate_flat_obs_dim(space_child)
                    parts.append(np.zeros(size, dtype=np.float32))
            return np.concatenate(parts)
        elif isinstance(data, np.ndarray):
            return data.flatten().astype(np.float32)
        elif isinstance(data, (int, float, bool, np.bool_)):
            return np.array([data], dtype=np.float32)
        else:
            # Fallback
            return np.array([data], dtype=np.float32).flatten()

    def decode_action(self, action_id: int) -> Dict[str, Any]:
        """
        Dekodiert die diskrete action_id in die strukturierte Aktion.
        """
        # Mapping: action_id = (choose * n_effort * n_collab) + (effort * n_collab) + collab
        
        collab_id = action_id % self.n_collab
        remaining = action_id // self.n_collab
        
        effort_id = remaining % self.n_effort
        choose_id = remaining // self.n_effort
        
        # Konvertiere collab_id in Binär-Array
        collab_bits = np.zeros(self.max_peer_group_size, dtype=np.int8)
        for i in range(self.max_peer_group_size):
            if (collab_id >> i) & 1:
                collab_bits[i] = 1
                
        return {
            "choose_project": int(choose_id),
            "put_effort": int(effort_id),
            "collaborate_with": collab_bits
        }

    def encode_action(self, choose_id: int, effort_id: int, collab_bits: np.ndarray) -> int:
        """
        Helper zum Enkodieren (z.B. für Tests).
        """
        collab_id = 0
        for i, bit in enumerate(collab_bits):
            if bit:
                collab_id |= (1 << i)
        
        return int(choose_id * self.n_effort * self.n_collab + effort_id * self.n_collab + collab_id)

    def reset(self, *, seed=None, options=None):
        obs_dict, info = self.env.reset(seed=seed, options=options)
        self._last_obs = obs_dict
        
        controlled_obs = obs_dict[self.controlled_agent]
        return self._flatten_observation(controlled_obs), info

    def step(self, action: int):
        # 1. Dekodiere die Aktion
        try:
            decoded_action = self.decode_action(int(action))
        except Exception:
            # Fallback bei ungültigen Aktionen (sollte bei Discrete Space nicht passieren)
            decoded_action = {
                "choose_project": 0,
                "put_effort": 0,
                "collaborate_with": np.zeros(self.max_peer_group_size, dtype=np.int8)
            }

        # 2. Sammle Aktionen aller Agenten
        actions_dict = {}
        
        # Der gesteuerte Agent
        actions_dict[self.controlled_agent] = decoded_action
        
        # Alle anderen Agenten (Heuristiken)
        for agent_id in self.env.possible_agents:
            if agent_id == self.controlled_agent:
                continue
            
            # Nur aktive Agenten bekommen eine Aktion
            if agent_id in self._last_obs:
                if agent_id in self.other_policies:
                    # Heuristik-Policies erwarten oft (obs, mask)
                    # PeerGroupEnvironment liefert die Maske oft als Teil der Observation oder separat.
                    # RLLibSingleAgentWrapper hat hier eine Logik, wir versuchen es simpel:
                    # Die meisten Heuristik-Policies in agent_policies.py erwarten (obs, action_mask).
                    # PeerGroupEnvironment._get_action_mask liefert die Maske.
                    
                    agent_obs = self._last_obs[agent_id]
                    
                    # Robustheit: Sicherstellen, dass agent_obs ein gültiges Dict ist und project_opportunities enthält.
                    # Dies verhindert den AttributeError in mass_producer_policy (und anderen).
                    if not isinstance(agent_obs, dict) or agent_obs.get("project_opportunities") is None:
                        actions_dict[agent_id] = {
                            "choose_project": 0,
                            "put_effort": 0,
                            "collaborate_with": np.zeros(self.max_peer_group_size, dtype=np.int8)
                        }
                        continue

                    agent_mask = self.env._get_action_mask(agent_id)
                    
                    # Policy-Input-Format konsistent mit train_dreamerv3.py _policy_from_name
                    policy_input = {
                        "observation": agent_obs,
                        "action_mask": agent_mask
                    }
                    action = self.other_policies[agent_id](policy_input)
                    if action is not None:
                        actions_dict[agent_id] = action
                    else:
                        # No-Op Fallback (nicht None an step() senden)
                        actions_dict[agent_id] = {
                            "choose_project": 0,
                            "put_effort": 0,
                            "collaborate_with": np.zeros(self.max_peer_group_size, dtype=np.int8)
                        }
                else:
                    # No-Op
                    actions_dict[agent_id] = {
                        "choose_project": 0,
                        "put_effort": 0,
                        "collaborate_with": np.zeros(self.max_peer_group_size, dtype=np.int8)
                    }

        # 3. Environment Step
        obs_dict, rewards, terminations, truncations, infos = self.env.step(actions_dict)
        self._last_obs = obs_dict

        # 4. Rückgabe für den gesteuerten Agenten
        # Wenn der gesteuerte Agent nicht mehr in obs_dict ist (z.B. gestorben), liefere Nullen
        if self.controlled_agent in obs_dict:
            flat_obs = self._flatten_observation(obs_dict[self.controlled_agent])
            reward = float(rewards.get(self.controlled_agent, 0.0))
            terminated = terminations.get(self.controlled_agent, False)
            truncated = truncations.get(self.controlled_agent, False)
            info = infos.get(self.controlled_agent, {})
        else:
            # Agent ist ausgeschieden
            flat_obs = np.zeros(self.flat_obs_dim, dtype=np.float32)
            reward = float(rewards.get(self.controlled_agent, 0.0))
            terminated = True # Wenn nicht in obs, dann meist terminiert
            truncated = False
            info = {}

        return flat_obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
