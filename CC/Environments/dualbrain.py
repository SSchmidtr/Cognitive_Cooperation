import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from gymnasium import spaces
from Environments.simplegridv2 import SimpleEnv
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava, Floor
from minigrid.manual_control import ManualControl

class DualBrainSimpleEnv(SimpleEnv):
    def __init__(self, render_mode="human"):
        super().__init__(render_mode=render_mode)
        
        self.observation_space = spaces.Dict({
            'brain1': spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8),
            'brain2': spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)
        })
        self.brain1_cumulative_reward = 0
        self.brain2_cumulative_reward = 0

    def get_dual_obs(self):
        grid = self.grid.encode()
        
        brain1_obs = np.zeros_like(grid)
        brain2_obs = np.zeros_like(grid)
        
        for i in range(self.width):
            for j in range(self.height):
                cell = self.grid.get(i, j)
                if cell is not None:
                    if isinstance(cell, (Lava, Floor)):
                        brain1_obs[i, j] = grid[i, j]
                    elif isinstance(cell, (Wall, Goal)):
                        brain2_obs[i, j] = grid[i, j]
        
        agent_pos = self.agent_pos
        brain1_obs[agent_pos[0], agent_pos[1]] = np.array([10, 0, self.agent_dir])
        brain2_obs[agent_pos[0], agent_pos[1]] = np.array([10, 0, self.agent_dir])
        
        return {
            'brain1': brain1_obs,
            'brain2': brain2_obs
        }

    def reset(self, seed=None, options=None):
        self.stepped_floors = set()  # Inicializar el conjunto de pisos visitados
        observation, info = super().reset(seed=seed, options=options)
        self._place_agent()  # Colocar al agente en una nueva posición aleatoria
        self.brain1_cumulative_reward = 0
        self.brain2_cumulative_reward = 0
        dual_obs = self.get_dual_obs()
        return dual_obs, info

    def step(self, action):
        # Llamamos al método step de la clase padre, pero ignoramos el reward
        obs, _, terminated, truncated, info = super().step(action)
        
        dual_obs = self.get_dual_obs()
        agent_cell = self.grid.get(*self.agent_pos)
        brain1_reward = 0
        brain2_reward = 0

        # Recompensas y penalizaciones basadas en la celda actual
        if agent_cell is not None and agent_cell.type == 'lava':
            brain1_reward -= 1  # Penalización por lava
            terminated = True
            print(f"Stepped on Lava! Negative reward: {brain1_reward}")
        elif agent_cell is not None and agent_cell.type == 'floor' and agent_cell.color == 'blue':
            if self.agent_pos not in self.stepped_floors:
                brain1_reward += 0.1  # Recompensa única por pisar un Floor
                self.stepped_floors.add(self.agent_pos)
                print(f"Stepped on new Floor! Positive reward: {brain1_reward}")
            else:
                print("Stepped on already visited Floor. No additional reward.")
        elif agent_cell is not None and agent_cell.type == 'goal':
            if len(self.stepped_floors) == len(self.key_positions):
                brain2_reward += 1  # Recompensa por alcanzar la meta con todos los Floors visitados
                terminated = True
                print(f"Reached Goal with all Floors visited! Reward: {brain2_reward}")
            else:
                print("Reached Goal, but not all Floors collected.")

        # Actualizar recompensas acumuladas e información
        self.brain1_cumulative_reward += brain1_reward
        self.brain2_cumulative_reward += brain2_reward
        info['brain1_reward'] = brain1_reward
        info['brain2_reward'] = brain2_reward
        info['cumulative_brain1_reward'] = self.brain1_cumulative_reward
        info['cumulative_brain2_reward'] = self.brain2_cumulative_reward
        
        total_reward = brain1_reward + brain2_reward
        reward = total_reward  # Asignamos nuestra recompensa total

        # Imprimir información para depuración
        print("\n--- Step Information ---")
        print(f"Action taken: {action}")
        print(f"Agent position: {self.agent_pos}")
        print(f"Agent direction: {self.agent_dir}")
        print(f"Agent cell: {agent_cell}")
        print(f"Stepped floors: {self.stepped_floors}")
        print(f"Number of keys (Floors): {len(self.key_positions)}")
        print(f"Number of floors stepped: {len(self.stepped_floors)}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print(f"\nBrain 1 Reward: {brain1_reward}")
        print(f"Brain 2 Reward: {brain2_reward}")
        print(f"Cumulative Brain 1 Reward: {self.brain1_cumulative_reward}")
        print(f"Cumulative Brain 2 Reward: {self.brain2_cumulative_reward}")
        print(f"Total Reward: {total_reward}")
        
        return dual_obs, reward, terminated, truncated, info

class CustomManualControl(ManualControl):
    def __init__(self, env, seed=None):
        super().__init__(env, seed)
        self.reset(seed)

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.step_count = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        print(f"\nStep: {self.step_count}, Total Reward: {reward:.2f}")
        print(f"Brain 1 Reward: {info['brain1_reward']:.2f}")
        print(f"Brain 2 Reward: {info['brain2_reward']:.2f}")
        print(f"Cumulative Brain 1 Reward: {info['cumulative_brain1_reward']:.2f}")
        print(f"Cumulative Brain 2 Reward: {info['cumulative_brain2_reward']:.2f}")

        if terminated or truncated:
            print("\n--- Episode Ended ---")
            print(f"Total Steps: {self.step_count}")
            print(f"Final Cumulative Brain 1 Reward: {info['cumulative_brain1_reward']:.2f}")
            print(f"Final Cumulative Brain 2 Reward: {info['cumulative_brain2_reward']:.2f}")
            self.reset(seed=self.seed)

        return obs, reward, terminated, truncated, info

# Test the environment
if __name__ == "__main__":
    env = DualBrainSimpleEnv()
    manual_control = CustomManualControl(env, seed=42)
    manual_control.start()
