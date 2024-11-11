
# Importamos las bibliotecas necesarias
import random  # Para generar números aleatorios
from typing import Optional  # Para definir tipos opcionales de variables
from gymnasium import spaces  # Librería para definir espacios de acción y observación
from minigrid.core.grid import Grid  # Para crear el grid en el que el agente se moverá
from minigrid.core.mission import MissionSpace  # Para definir misiones (tareas) en el entorno
from minigrid.core.world_object import Wall, Goal, Lava, Floor  # Objetos que podemos colocar en el grid
from minigrid.minigrid_env import MiniGridEnv  # Clase base para entornos de MiniGrid
from minigrid.manual_control import ManualControl  # Para controlar manualmente el entorno

# Definimos una nueva clase 'CombinedEnv' que hereda de 'MiniGridEnv', lo que significa
# que estamos creando un nuevo entorno basado en las funcionalidades del entorno original.
class CombinedEnv(MiniGridEnv):
    """
    Este entorno tiene un solo agente controlado por dos cerebros. Los cerebros decidirán
    los movimientos del agente en direcciones diferentes (norte-sur y este-oeste).
    """

    # Constructor de la clase CombinedEnv, se ejecuta cuando se crea una instancia del entorno
    def __init__(
        self,
        size=15,  # Tamaño del grid (15x15 por defecto)
        max_steps: Optional[int] = None,  # Máximo número de pasos permitidos (opcional)
        render_mode="Human",  # Modo de renderizado (por defecto, no se renderiza)
        **kwargs,  # Cualquier argumento adicional
    ):
        self.size = size  # Guardamos el tamaño del grid
        self.key_positions = []  # Lista de posiciones donde colocaremos las llaves
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

        #TODO: Aqui el action space acumulado debería de tener movimientos en las 4 direcciones además de diagonales
        #Cada acción es una combinación de dos acciones es suena buena idea
        # Definimos el espacio de acciones para cada cerebro
        self.brain1_action_space = spaces.Discrete(4)  # 0: Norte, 1: Sur, 2: Este, 3: Oeste
        self.brain2_action_space = spaces.Discrete(4)  # 0: Norte, 1: Sur, 2: Este, 3: Oeste

        # Combinamos las acciones de ambos cerebros
        self.combined_actions = [(a1, a2) for a1 in range(4) for a2 in range(4)]
        self.action_space = spaces.Discrete(len(self.combined_actions))

    # Función que genera la misión o tarea del entorno.
    @staticmethod
    def _gen_mission():
        return "Evitar la lava, pasar por los pisos, llegar a la meta"  # Misión del entorno

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
        
        # Colocamos paredes horizontales en posiciones fijas
        for x in range(1, width-1):
            self.put_obj(Wall(), x, 3)
            self.put_obj(Wall(), x, 7)
        
        # Creamos aperturas en las paredes para que el agente pueda pasar
        openings = [(3, 2), (3, 6), (3, 10), 
                    (7, 4), (7, 8), (7, 12),
                    (11, 2), (11, 6), (11, 10),
                    (2, 3), (6, 3), (10, 3),
                    (4, 7), (8, 7), (12, 7),
                    (2, 11), (6, 11), (10, 11)]
        for x, y in openings:
            self.grid.set(x, y, None)  # Quitamos paredes en estas posiciones para crear las aperturas
        
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
            #x = random.randint(1, self.size - 2) #TODO: Poner un valor fijo en lugar de un random para que el agente salga de una posición fija
            #y = random.randint(1, self.size - 2)
            x = 12
            y = 12
            self.agent_pos = (x, y)

            # Aseguramos que la posición esté vacía y no tenga obstáculos
            '''
            if (
                self.grid.get(*pos) is None and  # Que no haya objetos en esa posición
                pos not in self.lava_positions and  # Que no haya lava en esa posición
                pos not in self.key_positions and  # Que no haya llaves en esa posición
                pos != self.goal_pos  # Que no sea la posición de la meta
            ):
                self.agent_pos = pos  # Guardamos la posición del agente
                self.agent_dir = random.randint(0, 3)  # Asignamos una dirección aleatoria (norte, sur, este, oeste)
                break  # Salimos del bucle una vez que encontramos una posición válida'''

    # Función que reinicia el entorno, se llama al inicio de cada episodio
    def reset(self, **kwargs):
        self.step_count = 0  # Reiniciamos el contador de pasos
        self.stepped_floors = set()  # Reiniciamos el conjunto de llaves recogidas
        obs = super().reset(**kwargs)  # Llamamos a la función reset de la clase base
        return obs  # Retornamos la observación inicial del entorno

    # Función que procesa un paso del agente en el entorno
    def step(self, action):
        self.render_mode = "human"  # Modo de renderizado para visualizar el entorno
        # Dividimos la acción combinada en las acciones de los dos cerebros
        brain1_action, brain2_action = self.combined_actions[action]

        # Mapear las acciones a un vector de movimiento
        move_vector = self.get_move_vector(brain1_action, brain2_action)
        print(f"Env Step - Brain1 Action: {brain1_action}, Brain2 Action: {brain2_action}, Move Vector: {move_vector}")

        # Calculamos la nueva posición del agente según el movimiento
        new_pos = (self.agent_pos[0] + move_vector[0], self.agent_pos[1] + move_vector[1])

        # Inicializamos la recompensa, el estado de terminado, y truncado
        reward = 0
        terminated = False
        truncated = False
        info = {}

        # Verificamos si la nueva posición está dentro de los límites del grid
        if 0 <= new_pos[0] < self.grid.width and 0 <= new_pos[1] < self.grid.height:
            cell = self.grid.get(*new_pos)  # Obtenemos el objeto en la nueva posición
            if cell is None or cell.can_overlap():  # Si es un espacio vacío o puede ser atravesado
                self.agent_pos = new_pos  # Movemos el agente a la nueva posición
                self.agent_dir = 0  # Actualizamos la dirección del agente 
            else:
                reward -= 0.1  # Penalizamos por chocar con paredes u objetos no atravesables
        else:
            reward -= 0.5  # Penalizamos por intentar salir de los límites del grid
            terminated = True  # Terminamos el episodio si se sale del grid

        # Obtenemos el objeto en la celda actual donde está el agente
        current_cell = self.grid.get(*self.agent_pos)

        # Lógica para recompensas basadas en el tipo de celda en la que está el agente
        if isinstance(current_cell, Lava):  # Si el agente pisa lava
            reward -= 0.3  # Penalizamos por pisar lava
            terminated = True  # Terminamos el episodio
            print(f"Stepped on Lava! Negative reward: {reward}")
        elif isinstance(current_cell, Floor):  # Si el agente pisa una llave (tile azul)
            if self.agent_pos not in self.stepped_floors:  # Si no ha pisado esta llave antes
                reward += 2  # Recompensa por recoger una llave nueva
                self.stepped_floors.add(self.agent_pos)  # Añadimos esta llave a la lista de recogidas
                print(f"Stepped on new Floor! Positive reward: {reward}")
        elif isinstance(current_cell, Goal):  # Si el agente llega a la meta
            if len(self.stepped_floors) == len(self.key_positions):  # Si ha recogido todas las llaves
                reward += 10  # Recompensa máxima por completar la misión
                terminated = True  # Terminamos el episodio
                print(f"Reached Goal with all keys! Reward: {reward}")
            else:
                reward += 5  # Recompensa menor si llega a la meta sin todas las llaves
                terminated = True  # Terminamos el episodio
                print("Reached Goal, but not all keys collected.")

        if brain1_action == brain2_action:
            reward_brain1 += 0.2
            reward_brain2 += 0.2
            print("Brains agreed on action. Small positive reward added.")


        # Incrementamos el contador de pasos
        self.step_count += 1
        if self.step_count >= self.max_steps:  # Si alcanzamos el límite de pasos
            truncated = True  # Truncamos el episodio

        obs = self.gen_obs()  # Generamos la nueva observación
        return obs, reward, terminated, truncated, info  # Retornamos los resultados del paso

    def get_move_vector(self, brain1_action, brain2_action):
        action_to_vector = {
            0: (0, -1),  # Norte
            1: (0, 1),   # Sur
            2: (1, 0),   # Este
            3: (-1, 0),  # Oeste
        }

        dx1, dy1 = action_to_vector[brain1_action]
        dx2, dy2 = action_to_vector[brain2_action]

        # Sumar los vectores de movimiento
        dx = dx1 + dx2
        dy = dy1 + dy2

        # Limitar el movimiento a [-1, 0, 1] en cada eje
        dx = max(-1, min(1, dx))
        dy = max(-1, min(1, dy))

        move_vector = (dx, dy)
        return move_vector

    # Función para cerrar el entorno y limpiar recursos
    def close(self):
        if self.window:  # Si existe una ventana abierta
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None
        super().close()

    # Función principal que inicia el entorno
    def main():
        env = CombinedEnv(render_mode="human")  # Creamos el entorno con modo de renderizado humano
        manual_control = ManualControl(env, seed=42)  # Control manual para probar el entorno
        manual_control.start()  # Iniciamos el control manual

if __name__ == "__main__":
    CombinedEnv.main()  # Si ejecutamos el script directamente, se inicia el entorno