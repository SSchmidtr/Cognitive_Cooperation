import sys
import os
import pickle  # Importamos pickle para guardar y cargar las políticas

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Añadimos la ruta del archivo
from Environments.combined_env import CombinedEnv  # Importamos el entorno que hemos definido
from Navigation.navcombinedenv import CombinedAgent  # Importamos el agente que hemos definido


# Función para cargar la política desde un archivo
def load_policy(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)  # Retorna la política cargada


def main():
    env = CombinedEnv(render_mode='human')  # Creamos el entorno
    agent = CombinedAgent(env)  # Creamos el agente combinado

    # Cargamos las políticas para brain1 y brain2
    agent.brain1.policy = load_policy('brain1_policy.pkl')
    agent.brain2.policy = load_policy('brain2_policy.pkl')
    print("Policies loaded.")

    # Probamos al agente entrenado en varios episodios en modo de prueba
    for i in range(10):
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

        print(f"Total reward in test episode {i + 1}: {total_reward}")

if __name__ == "__main__":
    main()  # Ejecutamos la función principal si este archivo se ejecuta directamente
