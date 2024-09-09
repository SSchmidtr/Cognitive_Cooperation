import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from gymnasium import spaces
from Environments.simplegridv2 import SimpleEnv
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava, Floor
from minigrid.manual_control import ManualControl

#TODO: El Floor solamente puede dar 1 reward por episodio, no por cada vez que se pise
#TODO: Revisar que la asiganción y el action space tenga la asignación correcta
#TODO: ¿Agregar reward negativo a la pared?

class DualBrainSimpleEnv(SimpleEnv):
    def __init__(self, render_mode="human"):
        super().__init__(render_mode=render_mode)
        
        self.observation_space = spaces.Dict({
            'brain1': spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8),
            'brain2': spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)
        })
        self.cumulative_reward_brain1 = 0
        self.cumulative_reward_brain2 = 0

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
        observation, info = super().reset(seed=seed, options=None)
        dual_obs = self.get_dual_obs()
        self.cumulative_reward_brain1 = 0
        self.cumulative_reward_brain2 = 0
        print("\n--- Environment Reset ---")
        print("Brain 1 Observation:")
        print(dual_obs['brain1'])
        print("\nBrain 2 Observation:")
        print(dual_obs['brain2'])
        return dual_obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        
        obs = self.get_dual_obs()
        
        agent_cell = self.grid.get(*self.agent_pos)
        
        brain1_reward = 0
        brain2_reward = 0
        
        if isinstance(agent_cell, Lava):
            brain1_reward = -1  # Penalización por lava
        elif isinstance(agent_cell, Floor):
            brain1_reward = 0.1  # Pequeña recompensa positiva por moverse en el suelo
        elif isinstance(agent_cell, Goal):
            brain2_reward = 1  # Recompensa por alcanzar la meta

        self.cumulative_reward_brain1 += brain1_reward
        self.cumulative_reward_brain2 += brain2_reward

        info['brain1_reward'] = brain1_reward
        info['brain2_reward'] = brain2_reward
        info['cumulative_brain1_reward'] = self.cumulative_reward_brain1
        info['cumulative_brain2_reward'] = self.cumulative_reward_brain2
        
        total_reward = brain1_reward + brain2_reward
        
        print("\n--- Step Information ---")
        print(f"Action taken: {action}")
        print("Brain 1 Observation:")
        print(obs['brain1'])
        print("\nBrain 2 Observation:")
        print(obs['brain2'])
        print(f"\nBrain 1 Reward: {brain1_reward}")
        print(f"Brain 2 Reward: {brain2_reward}")
        print(f"Cumulative Brain 1 Reward: {self.cumulative_reward_brain1}")
        print(f"Cumulative Brain 2 Reward: {self.cumulative_reward_brain2}")
        print(f"Total Reward: {total_reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        
        return obs, total_reward, terminated, truncated, info

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