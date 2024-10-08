import numpy as np
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Wall, Goal, Lava, Floor
from minigrid.minigrid_env import MiniGridEnv
import random

class CombinedEnv(MiniGridEnv):
    """
    Este entorno tiene un solo agente controlado por dos cerebros. Los cerebros decidirán
    los movimientos del agente en direcciones diferentes (norte-sur y este-oeste).
    """

    def __init__(self, size=15, max_steps=None, render_mode=None, **kwargs):
        self.size = size
        self.key_positions = []
        self.lava_positions = []

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 10 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            render_mode=render_mode,
            **kwargs,
        )

        # Definir espacios de acción para cada cerebro
        self.brain1_action_space = spaces.Discrete(4)  # Norte, sur, este, oeste
        self.brain2_action_space = spaces.Discrete(4)  # Norte, sur, este, oeste

        # Definir espacio de observación
        self.observation_space = spaces.Dict({
            "brain1": spaces.Box(low=0, high=255, shape=(self.size, self.size, 3), dtype=np.uint8),
            "brain2": spaces.Box(low=0, high=255, shape=(self.size, self.size, 3), dtype=np.uint8)
        })

        # Crear espacio combinado de acciones
        self.combined_actions = [
            (a1, a2) for a1 in range(self.brain1_action_space.n) for a2 in range(self.brain2_action_space.n)
        ]
        self.action_space = spaces.Discrete(len(self.combined_actions))

    def gen_obs(self):
        """
        Generar observaciones personalizadas para cada cerebro.
        """
        obs_brain1 = self.get_brain1_obs()
        obs_brain2 = self.get_brain2_obs()
        return {"brain1": obs_brain1, "brain2": obs_brain2}

    def get_brain1_obs(self):
        """
        Generar observación para el cerebro 1: solo ve paredes y la meta.
        """
        obs = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        for x in range(self.size):
            for y in range(self.size):
                obj = self.grid.get(x, y)
                if isinstance(obj, Wall) or isinstance(obj, Goal):
                    obs[x, y, :] = obj.encode()
        return obs

    def get_brain2_obs(self):
        """
        Generar observación para el cerebro 2: solo ve lava y llaves.
        """
        obs = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        for x in range(self.size):
            for y in range(self.size):
                obj = self.grid.get(x, y)
                if isinstance(obj, Lava) or isinstance(obj, Floor):
                    obs[x, y, :] = obj.encode()
        return obs

    def _gen_grid(self, width, height):
        """
        Generar el grid inicial. Esto debe coincidir con lo que el espacio de observación espera.
        """
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Colocamos algunas paredes, lava, y llaves para tener un entorno adecuado
        for y in range(1, height-1):
            self.put_obj(Wall(), 3, y)
            self.put_obj(Wall(), 7, y)

        self.lava_positions = [(4, 1), (6, 6)]
        for x, y in self.lava_positions:
            self.put_obj(Lava(), x, y)

        self.key_positions = [(1, 5), (5, 9)]
        for x, y in self.key_positions:
            self.put_obj(Floor('blue'), x, y)

        # Colocar al agente en una posición inicial aleatoria
        self._place_agent()

        # Colocamos la meta
        self.put_obj(Goal(), width - 2, height - 2)

        self.mission = "Evitar la lava, recoger llaves, y llegar a la meta"

    def _place_agent(self):
        """
        Coloca al agente en una posición aleatoria dentro del grid, asegurándose de que no esté sobre una pared, llave, lava, o meta.
        """
        while True:
            x = random.randint(1, self.size - 2)
            y = random.randint(1, self.size - 2)
            if self.grid.get(x, y) is None:  # Asegúrate de que la celda esté vacía
                self.agent_pos = (x, y)
                self.agent_dir = random.randint(0, 3)  # Dirección aleatoria (norte, sur, este, oeste)
                break

    def step(self, action):
        brain1_action, brain2_action = self.combined_actions[action]
        move_vector = self.get_move_vector(brain1_action, brain2_action)
        new_pos = (self.agent_pos[0] + move_vector[0], self.agent_pos[1] + move_vector[1])

        reward = 0
        terminated = False
        truncated = False
        info = {}

        if 0 <= new_pos[0] < self.grid.width and 0 <= new_pos[1] < self.grid.height:
            cell = self.grid.get(*new_pos)
            if cell is None or cell.can_overlap():
                self.agent_pos = new_pos
                self.agent_dir = 0
        else:
            reward -= 0.5
            terminated = True

        current_cell = self.grid.get(*self.agent_pos)

        if isinstance(current_cell, Lava):
            reward -= 0.3
            terminated = True
        elif isinstance(current_cell, Floor):
            if self.agent_pos not in self.stepped_floors:
                reward += 2
                self.stepped_floors.add(self.agent_pos)
        elif isinstance(current_cell, Goal):
            if len(self.stepped_floors) == len(self.key_positions):
                reward += 10
                terminated = True
            else:
                reward += 5
                terminated = True

        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True

        obs = self.gen_obs()
        return obs, reward, terminated, truncated, info

    def get_move_vector(self, brain1_action, brain2_action):
        dx = 0
        dy = 0

        # Cerebro 1
        if brain1_action == 0:
            dy = -1
        elif brain1_action == 1:
            dy = 1
        elif brain1_action == 2:
            dx = 1
        elif brain1_action == 3:
            dx = -1

        # Cerebro 2
        if brain2_action == 0:
            dy = -1
        elif brain2_action == 1:
            dy = 1
        elif brain2_action == 2:
            dx = 1
        elif brain2_action == 3:
            dx = -1

        return dx, dy

    def reset(self, **kwargs):
        self.step_count = 0
        self.stepped_floors = set()
        obs = super().reset(**kwargs)
        return obs

    def close(self):
        if self.window:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None
        super().close()

    @staticmethod
    def _gen_mission():
        return "Evitar la lava, recoger llaves, y llegar a la meta"
