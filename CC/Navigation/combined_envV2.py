import random  # Para generar números aleatorios
from typing import Optional  # Para definir tipos opcionales de variables
from gymnasium import spaces  # Librería para definir espacios de acción y observación
from minigrid.core.grid import Grid  # Para crear el grid en el que el agente se moverá
from minigrid.core.mission import MissionSpace  # Para definir misiones (tareas) en el entorno
from minigrid.core.world_object import Wall, Goal, Lava, Floor  # Objetos que podemos colocar en el grid
from minigrid.minigrid_env import MiniGridEnv  # Clase base para entornos de MiniGrid
import numpy as np  # Librería para operaciones matemáticas

class CombinedEnv(MiniGridEnv):
    """
    Este entorno tiene un solo agente controlado por dos cerebros. Cada cerebro ve diferentes objetos
    en el entorno y recibe recompensas específicas basadas en lo que observa.
    """

    def __init__(
        self,
        size=15,  # Tamaño del grid (15x15 por defecto)
        max_steps: Optional[int] = None,  # Máximo número de pasos permitidos (opcional)
        render_mode=None,  # Modo de renderizado (por defecto, no se renderiza)
        **kwargs,  # Cualquier argumento adicional
    ):
        self.size = size  # Guardamos el tamaño del grid
        self.key_positions = []  # Lista de posiciones donde colocaremos los pisos (llaves)
        self.lava_positions = []  # Lista de posiciones donde colocaremos lava

        # Definimos el espacio de misión usando la función '_gen_mission'
        mission_space = MissionSpace(mission_func=self._gen_mission)

        # Si no se ha definido un máximo de pasos, lo calculamos basado en el tamaño del grid
        if max_steps is None:
            max_steps = 10 * size**2  # Máximo de pasos = 10 * (tamaño del grid al cuadrado)

        # Llamamos al constructor de la clase base 'MiniGridEnv'
        super().__init__(
            mission_space=mission_space,
            grid_size=size,  # Tamaño del grid
            see_through_walls=True,  # Permitir ver a través de las paredes
            max_steps=max_steps,  # Máximo de pasos permitidos
            render_mode=render_mode,  # Modo de renderizado
            **kwargs,  # Argumentos adicionales
        )

        # Definir el espacio de acciones para cada cerebro
        self.brain1_action_space = spaces.Discrete(4)  # Norte, sur, este, oeste
        self.brain2_action_space = spaces.Discrete(4)  # Norte, sur, este, oeste

        # Combinamos las acciones de ambos cerebros en un solo espacio de acciones
        self.combined_actions = [
            (a1, a2) for a1 in range(self.brain1_action_space.n) for a2 in range(self.brain2_action_space.n)
        ]
        self.action_space = spaces.Discrete(len(self.combined_actions))

        # Definir el espacio de observaciones para cada cerebro
        self.observation_space = spaces.Dict({
            "brain1": spaces.Box(low=0, high=255, shape=(self.size, self.size, 3), dtype=np.uint8),  # Cerebro 1 ve paredes y meta
            "brain2": spaces.Box(low=0, high=255, shape=(self.size, self.size, 3), dtype=np.uint8),  # Cerebro 2 ve lava y pisos
        })

    # Función que genera la misión o tarea del entorno.
    @staticmethod
    def _gen_mission():
        return "Evitar la lava, pasar por los pisos, llegar a la meta"

    # Función que genera el grid (la cuadrícula donde el agente se moverá)
    def _gen_grid(self, width, height):
        # Creamos un grid vacío con el ancho y alto especificado
        self.grid = Grid(width, height)
        # Rodeamos el grid con paredes
        self.grid.wall_rect(0, 0, width, height)

        # Colocamos paredes verticales en posiciones fijas
        for y in range(1, height-1):
            self.put_obj(Wall(), 3, y)
            self.put_obj(Wall(), 7, y)
            self.put_obj(Wall(), 11, y)

        # Colocamos 5 posiciones fijas de lava en el grid
        self.lava_positions = [(4, 1), (6, 6), (10, 11), (4, 12), (12, 4)]
        for x, y in self.lava_positions:
            self.put_obj(Lava(), x, y)  # Colocamos objetos de lava en las posiciones asignadas

        # Colocamos 5 tiles fijos de piso de color azul (simulando llaves)
        self.key_positions = [(1, 5), (5, 9), (9, 13), (13, 1), (9, 5)]
        for x, y in self.key_positions:
            self.put_obj(Floor('blue'), x, y)  # Colocamos las "llaves" como tiles de color azul

        # Colocamos la meta (goal) en la esquina inferior derecha
        self.goal_pos = (width - 2, height - 2)
        self.put_obj(Goal(), *self.goal_pos)

        # Colocamos al agente en una posición inicial aleatoria
        self._place_agent()

        # Definimos la misión del agente
        self.mission = "Evitar la lava, pasar por los pisos, llegar a la meta"

    # Función que coloca al agente en una posición aleatoria del grid
    def _place_agent(self):
        while True:
            # Elegimos posiciones aleatorias dentro de los límites del grid
            x = random.randint(1, self.size - 2)
            y = random.randint(1, self.size - 2)
            pos = (x, y)

            # Aseguramos que la posición esté vacía y no tenga obstáculos
            if (
                self.grid.get(*pos) is None and  # Que no haya objetos en esa posición
                pos not in self.lava_positions and  # Que no haya lava en esa posición
                pos not in self.key_positions and  # Que no haya llaves en esa posición
                pos != self.goal_pos  # Que no sea la posición de la meta
            ):
                self.agent_pos = pos  # Guardamos la posición del agente
                self.agent_dir = random.randint(0, 3)  # Asignamos una dirección aleatoria (norte, sur, este, oeste)
                break  # Salimos del bucle una vez que encontramos una posición válida

    # Función para generar observaciones separadas para cada cerebro
    def gen_obs(self):
        obs_brain1 = self.get_brain1_obs()  # Cerebro 1 ve paredes y meta
        obs_brain2 = self.get_brain2_obs()  # Cerebro 2 ve lava y pisos
        return {"brain1": obs_brain1, "brain2": obs_brain2}

    # Generar observación para el cerebro 1: solo ve paredes y la meta
    def get_brain1_obs(self):
        obs = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        for x in range(self.size):
            for y in range(self.size):
                obj = self.grid.get(x, y)
                if isinstance(obj, Wall) or isinstance(obj, Goal):
                    obs[x, y, :] = obj.encode()
        return obs

    # Generar observación para el cerebro 2: solo ve lava y pisos
    def get_brain2_obs(self):
        obs = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        for x in range(self.size):
            for y in range(self.size):
                obj = self.grid.get(x, y)
                if isinstance(obj, Lava) or isinstance(obj, Floor):
                    obs[x, y, :] = obj.encode()
        return obs

    # Función que procesa un paso del agente en el entorno
    def step(self, action):
        # Dividimos la acción combinada en las acciones de los dos cerebros
        brain1_action, brain2_action = self.combined_actions[action]

        # Mapear las acciones a un vector de movimiento
        move_vector = self.get_move_vector(brain1_action, brain2_action)

        # Calculamos la nueva posición del agente según el movimiento
        new_pos = (self.agent_pos[0] + move_vector[0], self.agent_pos[1] + move_vector[1])

        # Inicializamos las recompensas y los estados de terminado
        reward_brain1 = 0  # Recompensa para el cerebro 1 (paredes y meta)
        reward_brain2 = 0  # Recompensa para el cerebro 2 (lava y pisos)
        terminated = False
        truncated = False
        info = {}

        # Verificamos si la nueva posición está dentro de los límites del grid
        if 0 <= new_pos[0] < self.grid.width and 0 <= new_pos[1] < self.grid.height:
            cell = self.grid.get(*new_pos)
            if cell is None or cell.can_overlap():
                self.agent_pos = new_pos  # Movemos el agente a la nueva posición
                self.agent_dir = 0  # Actualizamos la dirección del agente 
            '''else:
                reward_brain1 -= 0.1  # Penalizamos a Cerebro 1 por chocar con paredes u objetos no atravesables'''

        # Obtenemos el objeto en la celda actual donde está el agente
        current_cell = self.grid.get(*self.agent_pos)

        # Lógica para recompensas basadas en el tipo de celda en la que está el agente
        if isinstance(current_cell, Lava):  # Si el agente pisa lava
            reward_brain2 -= 0.3  # Penalizamos a Cerebro 2 por pisar lava
            terminated = True  # Terminamos el episodio
        elif isinstance(current_cell, Floor):  # Si el agente pisa un tile de Floor
            if self.agent_pos not in self.stepped_floors:  # Si no ha pisado este tile antes
                reward_brain2 += 2  # Recompensa para Cerebro 2 por pisar un nuevo tile de Floor
                self.stepped_floors.add(self.agent_pos)  # Añadimos esta posición al set de tiles ya pisados
        elif isinstance(current_cell, Goal):  # Si el agente llega a la meta
            if len(self.stepped_floors) == len(self.key_positions):  # Si ha pisado todos los tiles de Floor
                reward_brain1 += 10  # Recompensa para Cerebro 1 por completar la misión
                terminated = True  # Terminamos el episodio
            else:
                reward_brain1 += 5  # Recompensa menor si llega a la meta sin haber pisado todos los tiles
                terminated = True  # Terminamos el episodio

        # Incrementamos el contador de pasos
        self.step_count += 1
        if self.step_count >= self.max_steps:  # Si alcanzamos el límite de pasos
            truncated = True  # Truncamos el episodio

        obs = self.gen_obs()  # Generamos la nueva observación
        return obs, (reward_brain1, reward_brain2), terminated, truncated, info  # Retornamos los resultados del paso

    # Función para obtener el vector de movimiento basado en las acciones de los dos cerebros
    def get_move_vector(self, brain1_action, brain2_action):
        dx = 0  # Movimiento en el eje x (este-oeste)
        dy = 0  # Movimiento en el eje y (norte-sur)

        # Acciones del Cerebro 1 (north=0, south=1, east=2, west=3)
        if brain1_action == 0:
            dy = -1  # norte (disminuir y)
        elif brain1_action == 1:
            dy = 1   # sur (aumentar y)
        elif brain1_action == 2:
            dx = 1   # este (aumentar x)
        elif brain1_action == 3:
            dx = -1  # oeste (disminuir x)

        # Acciones del Cerebro 2 (north=0, south=1, east=2, west=3)
        if brain2_action == 0:
            dy = -1  # norte (disminuir y)
        elif brain2_action == 1:
            dy = 1   # sur (aumentar y)
        elif brain2_action == 2:
            dx = 1   # este (aumentar x)
        elif brain2_action == 3:
            dx = -1  # oeste (disminuir x)

        return dx, dy  # Retornamos el vector de movimiento

    # Función que reinicia el entorno, se llama al inicio de cada episodio
    def reset(self, **kwargs):
        self.step_count = 0  # Reiniciamos el contador de pasos
        self.stepped_floors = set()  # Reiniciamos el conjunto de pisos ya pisados
        obs = super().reset(**kwargs)  # Llamamos a la función reset de la clase base
        return obs  # Retornamos la observación inicial del entorno
