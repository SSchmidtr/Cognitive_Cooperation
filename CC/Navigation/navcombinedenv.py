import numpy as np  # Biblioteca para manejar arrays y operaciones numéricas
import matplotlib.pyplot as plt  # Para graficar los resultados
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Añadimos la ruta del archivo
from Environments.combined_env import CombinedEnv  # Importamos el entorno que hemos definido

# Clase que define una política de un cerebro basado en aprendizaje por refuerzo
class BrainPolicy:
    def __init__(self, action_space, learning_rate=0.01, gamma=0.99, epsilon_start=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.action_space = action_space  # El espacio de acciones del cerebro
        self.learning_rate = learning_rate  # Tasa de aprendizaje (qué tanto ajustar la política después de cada episodio)
        self.gamma = gamma  # Factor de descuento (qué tan importante es el futuro respecto al presente)
        self.epsilon = epsilon_start  # Valor inicial de epsilon (para controlar la exploración)
        self.epsilon_decay = epsilon_decay  # Tasa de decaimiento de epsilon (disminuir la exploración a lo largo del tiempo)
        self.epsilon_min = epsilon_min  # Valor mínimo de epsilon (cuando parar de explorar)
        self.policy = {}  # Inicializamos la política como un diccionario vacío
        self.loss_history = []  # Para almacenar la pérdida (loss) en cada episodio

    # Función que obtiene una acción basado en el estado actual y la política
    def get_action(self, state):
        state_key = tuple(state.flatten())  # Convertimos el estado en una clave manejable (un tuple)
        if state_key not in self.policy:
            self.policy[state_key] = np.zeros(self.action_space.n)  # Inicializamos la política para este estado si no existe
        
        # Aplicamos la estrategia epsilon-greedy para decidir si explorar o explotar
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space.n)  # Exploración: seleccionamos una acción aleatoria
        else:
            probs = self.softmax(self.policy[state_key])  # Explotación: seleccionamos la mejor acción según la política
            return np.random.choice(self.action_space.n, p=probs)

    # Función que aplica softmax para convertir las preferencias en probabilidades
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))  # Restamos el máximo valor de 'x' para estabilidad numérica
        return exp_x / exp_x.sum()  # Retornamos las probabilidades normalizadas

    # Función que actualiza la política del cerebro después de cada episodio
    def update_policy(self, episode):
        discounted_rewards = []  # Lista para almacenar las recompensas descontadas
        # Calculamos las recompensas descontadas (suma de recompensas futuras con un factor de descuento)
        for t in range(len(episode)):
            Gt = sum([r * (self.gamma ** i) for i, (_, _, r) in enumerate(episode[t:])])
            discounted_rewards.append(Gt)
        discounted_rewards = np.array(discounted_rewards)  # Convertimos a array

        # Calculamos el baseline (promedio de las recompensas descontadas)
        baseline = np.mean(discounted_rewards)
        
        # Calculamos las ventajas (recompensas descontadas menos el baseline)
        advantages = discounted_rewards - baseline

        loss = 0  # Inicializamos la pérdida para este episodio

        # Recorremos cada paso del episodio para actualizar la política
        for t, (state, action, _) in enumerate(episode):
            state_key = tuple(state.flatten())  # Convertimos el estado a una clave única
            if state_key not in self.policy:
                self.policy[state_key] = np.zeros(self.action_space.n)  # Inicializamos la política para este estado si no existe

            probs = self.softmax(self.policy[state_key])  # Calculamos las probabilidades actuales
            grad = -probs  # Calculamos el gradiente negativo
            grad[action] += 1  # Ajustamos el gradiente para la acción tomada

            # Calculamos la pérdida (logaritmo de la probabilidad ponderada por la ventaja)
            loss -= np.log(probs[action]) * advantages[t]

            # Actualizamos la política usando la ventaja
            self.policy[state_key] += self.learning_rate * advantages[t] * grad

        # Almacenamos la pérdida de este episodio
        self.loss_history.append(loss)

        # Reducimos epsilon después de cada episodio (menos exploración a medida que entrenamos más)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # Función que grafica la pérdida acumulada a lo largo de los episodios
    def plot_loss(self):
        plt.figure()
        plt.plot(self.loss_history)  # Graficamos la pérdida
        plt.title("Policy Loss over Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.show()

# Clase que representa al agente combinado (controlado por dos cerebros)
class CombinedAgent:
    def __init__(self, env):
        self.env = env  # Asignamos el entorno en el que entrenará el agente
        self.brain1 = BrainPolicy(env.brain1_action_space)  # Cerebro 1 (controla norte-sur)
        self.brain2 = BrainPolicy(env.brain2_action_space)  # Cerebro 2 (controla este-oeste)

    # Función que combina las acciones de ambos cerebros en una acción combinada
    def get_combined_action(self, state):
        action1 = self.brain1.get_action(state)  # Cerebro 1 elige una acción
        action2 = self.brain2.get_action(state)  # Cerebro 2 elige una acción
        combined_action = self.env.combined_actions.index((action1, action2))  # Encontramos el índice de la acción combinada
        return combined_action  # Retornamos la acción combinada

    # Función para entrenar al agente durante un número de episodios
    def train(self, num_episodes, test_ratio):
        training_episodes = int(num_episodes * (1 - test_ratio))  # Calculamos el número de episodios de entrenamiento
        test_episodes = num_episodes - training_episodes  # Calculamos los episodios de prueba
        reward_history = []  # Lista para almacenar las recompensas por episodio
        
        # Fase de entrenamiento
        for episode in range(training_episodes):
            obs, info = self.env.reset()  # Reiniciamos el entorno al inicio del episodio
            state = obs['image']  # Obtenemos la observación inicial
            done = False  # Variable para verificar si el episodio ha terminado
            episode_memory1 = []  # Memoria del Cerebro 1 para este episodio
            episode_memory2 = []  # Memoria del Cerebro 2 para este episodio
            total_reward = 0  # Recompensa total acumulada durante el episodio
            while not done:
                combined_action = self.get_combined_action(state)  # Obtenemos la acción combinada
                next_obs, reward, terminated, truncated, _ = self.env.step(combined_action)  # Ejecutamos la acción en el entorno
                next_state = next_obs['image']  # Actualizamos el estado
                done = terminated or truncated  # Verificamos si el episodio ha terminado
                # Guardamos las experiencias para cada cerebro
                brain1_action, brain2_action = self.env.combined_actions[combined_action]  # Separar las acciones combinadas
                episode_memory1.append((state, brain1_action, reward))  # Guardar experiencia para cerebro 1
                episode_memory2.append((state, brain2_action, reward))  # Guardar experiencia para cerebro 2
                state = next_state  # Actualizamos el estado actual
                total_reward += reward  # Sumamos la recompensa obtenida
            # Actualizamos las políticas de ambos cerebros
            self.brain1.update_policy(episode_memory1)
            self.brain2.update_policy(episode_memory2)
            reward_history.append(total_reward)  # Guardamos la recompensa total obtenida en el episodio
            if (episode + 1) % 100 == 0:  # Imprimir cada 100 episodios
                print(f"Episode {episode + 1}/{training_episodes}, Total Reward: {total_reward}")
        
        print("Training completed.")  # Indicamos que ha terminado el entrenamiento

        # Graficamos la pérdida de ambos cerebros
        self.brain1.plot_loss()
        self.brain2.plot_loss()

        # Fase de pruebas
        test_rewards = []  # Lista para almacenar las recompensas de prueba
        for test_episode in range(test_episodes):
            obs, info = self.env.reset()  # Reiniciamos el entorno para cada episodio de prueba
            state = obs['image']  # Obtenemos la observación inicial
            done = False  # Variable para verificar si el episodio ha terminado
            total_reward = 0  # Recompensa total para este episodio de prueba
            while not done:
                combined_action = self.get_combined_action(state)  # Obtenemos la acción combinada
                next_obs, reward, terminated, truncated, _ = self.env.step(combined_action)  # Ejecutamos la acción
                state = next_obs['image']  # Actualizamos el estado
                done = terminated or truncated  # Verificamos si el episodio ha terminado
                total_reward += reward  # Sumamos la recompensa
            test_rewards.append(total_reward)  # Guardamos la recompensa de este episodio de prueba
            print(f"Test Episode {test_episode + 1}/{test_episodes}, Total Test Reward: {total_reward}")

        # Calculamos el promedio de las recompensas de prueba
        avg_test_reward = np.mean(test_rewards)
        print(f"Average Test Reward: {avg_test_reward}")
        
        return reward_history, avg_test_reward  # Retornamos el historial de recompensas y la recompensa promedio de prueba

# Función que grafica la tendencia de las recompensas durante los episodios de entrenamiento
def plot_trend_line(rewards):
    plt.figure()
    plt.plot(rewards, label="Reward")  # Graficamos las recompensas
    
    # Calculamos una línea de tendencia usando el promedio móvil
    window_size = 50  # Tamaño de la ventana para el promedio móvil
    moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
    
    plt.plot(range(len(moving_avg)), moving_avg, color='red', label="Trend (Moving Average)")
    plt.title("Rewards over Training Episodes with Trend")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.show()

# Función principal para entrenar y probar al agente
def main():
    env = CombinedEnv(render_mode='human')  # Creamos el entorno
    agent = CombinedAgent(env)  # Creamos el agente combinado
    num_episodes = 5000  # Definimos el número total de episodios (entrenamiento + pruebas)
    reward_history, avg_test_reward = agent.train(num_episodes=num_episodes, test_ratio=0.01)  # Entrenamos al agente
    
    # Graficamos los rewards y la línea de tendencia
    plot_trend_line(reward_history)

    # Mostramos el promedio de recompensas de las pruebas
    print(f"Average Test Reward over {int(num_episodes * 0.2)} episodes: {avg_test_reward}")

    # Probamos al agente entrenado en un episodio
    obs, info = env.reset()  # Reiniciamos el entorno
    state = obs['image']  # Obtenemos la observación inicial
    done = False  # Verificamos si el episodio ha terminado
    total_reward = 0  # Inicializamos la recompensa total
    while not done:
        combined_action = agent.get_combined_action(state)  # Obtenemos la acción combinada
        next_obs, reward, terminated, truncated, _ = env.step(combined_action)  # Ejecutamos la acción en el entorno
        state = next_obs['image']  # Actualizamos el estado
        done = terminated or truncated  # Verificamos si el episodio ha terminado
        total_reward += reward  # Sumamos la recompensa
        env.render()  # Renderizamos el entorno en cada paso
    print(f"Total reward in test episode: {total_reward}")

if __name__ == "__main__":
    main()  # Ejecutamos la función principal si este archivo se ejecuta directamente
